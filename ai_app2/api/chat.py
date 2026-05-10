"""
Chat API 路由：处理 POST /chat 请求。

与 ai_app1 的区别：
- 使用 LangGraph 的 state graph 替代手写的顺序函数调用
- 流式响应：Graph 执行完成后逐字 yield 回复，保持与 ai_app1 相同的 API 契约
- 会话状态由 checkpointer 管理，无需手写字典
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_app2.core.config import DEFAULT_TOKEN_BUDGET
from ai_app2.core.logger import chat_logger
from ai_app2.graph.builder import graph

router = APIRouter()


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    message: str


def _make_thread_id(user_id: str) -> str:
    """简单映射：每个 user_id 对应一个线程 ID"""
    return f"thread_{user_id}"


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    流式处理用户聊天请求。

    流程：
        1. 从 checkpointer 加载当前线程状态
        2. 将用户消息追加到 history，组装 input_state
        3. 调用 graph.ainvoke 执行完整 Graph
        4. 获取最终 reply，逐字符流式 yield 给客户端
        5. Graph 内部已自动完成：retrieve → build_messages → llm → save_reply → summarize? → trim
    """
    req_id = id(req)
    user_id = "default_user"
    thread_id = _make_thread_id(user_id)
    config = {"configurable": {"thread_id": thread_id}}

    chat_logger.info(f"[{req_id}] 收到请求: message={req.message[:50]!r}, thread={thread_id}")
    start = time.monotonic()

    # ── 1. 从 checkpoint 加载当前状态 ────────────────────────────────────────
    current_state = graph.get_state(config)
    if current_state and current_state.values:
        vals = current_state.values
        history = list(vals.get("history", []))
        summary = vals.get("summary", "")
        token_budget = vals.get("token_budget", DEFAULT_TOKEN_BUDGET)
        trimmed = list(vals.get("trimmed", []))
    else:
        history, summary, token_budget, trimmed = [], "", DEFAULT_TOKEN_BUDGET, []

    # 追加本轮用户消息
    history.append({"role": "user", "content": req.message})

    input_state = {
        "user_message": req.message,
        "history": history,
        "summary": summary,
        "token_budget": token_budget,
        "retrieved_context": None,
        "messages": [],
        "reply": "",
        "trimmed": trimmed,
    }

    chat_logger.debug(
        f"[{req_id}] 初始 state: history={len(history)}, summary={'有' if summary else '无'}, "
        f"budget={token_budget}"
    )

    # ── 2. 流式生成器 ───────────────────────────────────────────────────────
    async def _stream_generator() -> AsyncGenerator[str, None]:
        try:
            # 执行完整 Graph（retrieve → build_messages → llm → save_reply → summarize? → trim）
            result = await graph.ainvoke(input_state, config=config)
            reply = result.get("reply", "") or ""

            if not reply:
                chat_logger.warning(f"[{req_id}] Graph 返回空 reply")
                yield "抱歉，没有生成回复。"
                return

            # 模拟流式输出（与 ai_app1 的 StreamingResponse 契约保持一致）
            chunk_size = 2
            for i in range(0, len(reply), chunk_size):
                yield reply[i : i + chunk_size]
                if i < len(reply) - chunk_size:
                    await asyncio.sleep(0.005)

            elapsed = time.monotonic() - start
            final_history = result.get("history", [])
            final_summary = result.get("summary", "")
            chat_logger.info(
                f"[{req_id}] 请求完成: reply_len={len(reply)}, history={len(final_history)}, "
                f"summary={'有' if final_summary else '无'}, 耗时={elapsed:.2f}s"
            )

        except Exception as e:
            chat_logger.error(f"[{req_id}] Graph 执行异常: {type(e).__name__}: {e}")
            yield f"Error: {e}"

    return StreamingResponse(_stream_generator(), media_type="text/plain")
