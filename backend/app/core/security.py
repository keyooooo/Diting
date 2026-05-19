import base64
import hashlib
import hmac
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from threading import Lock
from dotenv import load_dotenv

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models import User

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY 环境变量未设置")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "1440"))
ADMIN_INVITE_CODE = os.getenv("ADMIN_INVITE_CODE", "")
PBKDF2_ROUNDS = int(os.getenv("PASSWORD_PBKDF2_ROUNDS", "310000"))

# Dummy hash for timing blinding — prevents username enumeration via timing side-channel.
# Generated once at module load with the same round count as real password hashes.
_DUMMY_SALT = base64.b64encode(os.urandom(16)).decode("ascii")
_DUMMY_DIGEST = base64.b64encode(os.urandom(32)).decode("ascii")
_DUMMY_HASH = f"pbkdf2_sha256${PBKDF2_ROUNDS}${_DUMMY_SALT}${_DUMMY_DIGEST}"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class RateLimiter:
    """In-memory sliding-window rate limiter — swap to Redis for multi-process."""

    def __init__(self, max_attempts: int, window_seconds: int) -> None:
        self._max = max_attempts
        self._window = window_seconds
        self._attempts: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_limited(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            cutoff = now - self._window
            self._attempts[key] = [t for t in self._attempts[key] if t > cutoff]
            if len(self._attempts[key]) >= self._max:
                return True
            self._attempts[key].append(now)
            return False


login_limiter = RateLimiter(max_attempts=5, window_seconds=900)  # 5 次 / 15 分钟


def check_login_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    if login_limiter.is_limited(client_ip):
        raise HTTPException(status_code=429, detail="登录尝试过于频繁，请 15 分钟后再试")


def verify_password(plain_password: str, password_hash: str) -> bool:
    if not plain_password or not password_hash:
        return False

    # New format: pbkdf2_sha256$<rounds>$<salt_b64>$<digest_b64>
    if password_hash.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt_b64, digest_b64 = password_hash.split("$", 3)
            salt = base64.b64decode(salt_b64.encode("ascii"))
            expected = base64.b64decode(digest_b64.encode("ascii"))
            calculated = hashlib.pbkdf2_hmac(
                "sha256",
                plain_password.encode("utf-8"),
                salt,
                int(rounds),
            )
            return hmac.compare_digest(calculated, expected)
        except Exception:
            return False

    # Backward compatibility for legacy passlib/bcrypt hashes.
    if password_hash.startswith("$2") or password_hash.startswith("$bcrypt"):
        try:
            from passlib.context import CryptContext

            legacy_context = CryptContext(schemes=["bcrypt_sha256", "bcrypt"], deprecated="auto")
            return legacy_context.verify(plain_password, password_hash)
        except Exception:
            return False

    return False


def get_password_hash(password: str) -> str:
    if not password:
        raise ValueError("password is required")

    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ROUNDS,
    )
    salt_b64 = base64.b64encode(salt).decode("ascii")
    digest_b64 = base64.b64encode(digest).decode("ascii")
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt_b64}${digest_b64}"


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": username,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        # Run dummy pbkdf2 to blind timing — prevents username enumeration
        verify_password(password, _DUMMY_HASH)
        return None
    if not verify_password(password, user.password_hash):
        return None
    return user


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的认证令牌",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员权限不足")
    return current_user


def resolve_role(requested_role: str | None, admin_code: str | None) -> str:
    role = (requested_role or "user").strip().lower()
    if role != "admin":
        return "user"
    if ADMIN_INVITE_CODE and admin_code == ADMIN_INVITE_CODE:
        return "admin"
    raise HTTPException(status_code=403, detail="管理员邀请码错误")
