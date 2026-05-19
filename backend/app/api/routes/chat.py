import json
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.agent import chat_with_agent, chat_with_agent_stream
from app.api.deps import _collection_name
from app.core.security import get_current_user
from app.models import User
from app.schemas import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = chat_with_agent(request.message, current_user.username, session_id, collection_name=_collection_name(request.kb_id), enable_rag=request.enable_rag)
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "上游模型服务触发限流/额度限制（429）。请检查账号额度/模型状态。\n"
                        f"原始错误：{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """跟 Agent 对话 (流式)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in chat_with_agent_stream(request.message, current_user.username, session_id, collection_name=_collection_name(request.kb_id), enable_rag=request.enable_rag):
                yield chunk
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
''' 
Header	作用
Cache-Control: no-cache	禁止中间代理缓存 SSE 流
Connection: keep-alive	SSE 协议要求
X-Accel-Buffering: no	关键：禁止 Nginx 缓冲 SSE，否则用户看到的是卡顿后的整段文本 
'''