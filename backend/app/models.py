from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy import JSON, Text, UniqueConstraint
from sqlalchemy import Column as SAColumn
from sqlalchemy import ForeignKey as SAForeignKey
from sqlalchemy import Integer as SAInteger
from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    username: str = Field(max_length=100, unique=True, index=True)
    password_hash: str = Field(max_length=255)
    role: str = Field(default="user", max_length=20)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    sessions: List["ChatSession"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ChatSession(SQLModel, table=True):
    __tablename__ = "chat_sessions"
    __table_args__ = (UniqueConstraint("user_id", "session_id", name="uq_user_session"),)

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    user_id: int = Field(
        sa_column=SAColumn(
            SAInteger,
            SAForeignKey("users.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    session_id: str = Field(max_length=120, index=True)
    metadata_json: dict[str, Any] = Field(default_factory=dict, sa_column=SAColumn(JSON, nullable=False))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    user: "User" = Relationship(back_populates="sessions")
    messages: List["ChatMessage"] = Relationship(
        back_populates="session",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class ChatMessage(SQLModel, table=True):
    __tablename__ = "chat_messages"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    session_ref_id: int = Field(
        sa_column=SAColumn(
            SAInteger,
            SAForeignKey("chat_sessions.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    message_type: str = Field(max_length=20)
    content: str = Field(sa_type=Text)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    rag_trace: Optional[dict[str, Any]] = Field(default=None, sa_column=SAColumn(JSON, nullable=True))

    session: "ChatSession" = Relationship(back_populates="messages")


class ParentChunk(SQLModel, table=True):
    __tablename__ = "parent_chunks"

    chunk_id: str = Field(max_length=512, primary_key=True)
    text: str = Field(sa_type=Text)
    filename: str = Field(max_length=255, index=True)
    file_type: str = Field(default="", max_length=50)
    file_path: str = Field(default="", max_length=1024)
    page_number: int = Field(default=0)
    parent_chunk_id: str = Field(default="", max_length=512)
    root_chunk_id: str = Field(default="", max_length=512)
    chunk_level: int = Field(default=0)
    chunk_idx: int = Field(default=0)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KnowledgeBase(SQLModel, table=True):
    __tablename__ = "knowledge_bases"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    name: str = Field(max_length=100, index=True)
    description: str = Field(default="", max_length=500)
    created_by: int = Field(
        sa_column=SAColumn(
            SAInteger,
            SAForeignKey("users.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    creator: "User" = Relationship()
