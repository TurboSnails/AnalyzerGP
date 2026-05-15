"""
Chat API 路由 — 基于 rag_framework 的流式对话接口

职责：
  1. 接收用户消息
  2. 通过 RAGContainer 调用流式对话
  3. 返回 StreamingResponse

容器从 app.state 获取，零全局状态，测试时可注入 mock。
"""
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_framework.container import RAGContainer
from rag_framework.core.logger import chat_logger

router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"


def get_container(request: Request) -> RAGContainer:
    """从 app.state 读取容器（由 lifespan 注入）。"""
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("RAGContainer 未初始化，请检查 lifespan 是否已执行")
    return container


@router.post("/chat")
async def chat(req: ChatRequest, container: RAGContainer = Depends(get_container)):
    """
    流式聊天接口。

    请求：{ "message": "...", "user_id": "..." }
    响应：text/plain 流式输出
    """
    req_id = id(req)
    chat_logger.info(
        f"[{req_id}] 收到请求: message={req.message[:50]!r}, user={req.user_id}"
    )
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

    return StreamingResponse(content_generator(), media_type="text/event-stream")
