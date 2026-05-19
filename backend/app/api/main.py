from fastapi import APIRouter

from app.api.routes import auth, chat, documents, knowledge_bases, sessions

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(sessions.router)
api_router.include_router(chat.router)
api_router.include_router(documents.router)
api_router.include_router(knowledge_bases.router)
