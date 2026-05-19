from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user, get_db, require_admin
from app.rag.milvus_client import MilvusManager
from app.models import KnowledgeBase, User
from app.schemas import KnowledgeBaseCreate, KnowledgeBaseInfo, KnowledgeBaseListResponse

milvus_manager = MilvusManager()

router = APIRouter()


@router.post("/knowledge-bases", response_model=KnowledgeBaseInfo)
async def create_knowledge_base(
    body: KnowledgeBaseCreate,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    kb = KnowledgeBase(
        name=body.name,
        description=body.description or "",
        created_by=current_user.id,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return KnowledgeBaseInfo(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        created_by=kb.created_by,
        created_at=kb.created_at.isoformat(),
    )


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    _: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kbs = db.query(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).all()
    return KnowledgeBaseListResponse(
        knowledge_bases=[
            KnowledgeBaseInfo(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                created_by=kb.created_by,
                created_at=kb.created_at.isoformat(),
            )
            for kb in kbs
        ]
    )


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: int,
    _: User = Depends(require_admin),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")
    db.delete(kb)
    db.commit()

    try:
        milvus_manager.drop_collection(f"kb_{kb_id}")
    except Exception:
        pass

    return {"message": f"知识库 {kb.name} 已删除"}
