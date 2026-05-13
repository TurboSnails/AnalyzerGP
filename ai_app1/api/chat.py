"""
Chat API 路由 — 基于 rag_framework 的流式对话接口

职责：
  1. 接收用户消息
  2. 通过 RAGContainer 调用流式对话
  3. 返回 StreamingResponse

所有业务逻辑（检索、LLM、会话管理）均在框架层处理。
"""
import asyncio
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import chat_logger

router = APIRouter()

# 全局容器（进程级单例）
_container: RAGContainer | None = None


def get_container() -> RAGContainer:
    global _container
    if _container is None:
        _container = RAGContainer.from_settings(get_settings())
    return _container


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


@router.post("/chat")
async def chat(req: ChatRequest, container: RAGContainer = Depends(get_container)):
    """
    流式聊天接口。

    请求：{ "message": "...", "user_id": "..." }
    响应：text/plain 流式输出
    """
    req_id = id(req)
    chat_logger.info(f"[{req_id}] 收到请求: message={req.message[:50]!r}, user={req.user_id}")
    start = time.monotonic()

    async def content_generator() -> AsyncIterator[str]:
        t_before = time.monotonic()
        first_chunk_logged = False

        async for chunk in container.chat_stream(req.message, req.user_id):
            if chunk:
                if not first_chunk_logged:
                    chat_logger.info(
                        f"[{req_id}] 收到 LLM 首字: "
                        f"TTFT={1000*(time.monotonic()-t_before):.0f}ms"
                    )
                    first_chunk_logged = True
                yield chunk

        elapsed = time.monotonic() - start
        chat_logger.info(f"[{req_id}] 流式传输完成: 总耗时={elapsed:.2f}s")

    return StreamingResponse(content_generator(), media_type="text/plain")
