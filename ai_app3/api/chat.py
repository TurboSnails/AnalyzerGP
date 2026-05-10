"""
Chat API 路由：处理 POST /chat 请求。

与 ai_app2 的区别：
- 第三代 Agentic RAG：意图分析 → 查询分解 → 迭代检索 → 评估 → 改写/KG扩展 → 生成
- SSE 流式响应：支持 trace（执行轨迹）+ content（回复内容）双通道
- 真流式：graph.ainvoke 完成后先推送 trace，再逐字推送 content
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_app3.core.config import DEFAULT_TOKEN_BUDGET
from ai_app3.core.logger import chat_logger
from ai_app3.graph.builder import graph

router = APIRouter()


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    message: str


def _make_thread_id(user_id: str) -> str:
    return f"thread_{user_id}"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    流式处理用户聊天请求（SSE）。

    流程：
        1. 从 checkpointer 加载当前线程状态
        2. 组装 input_state
        3. 调用 graph.ainvoke 执行完整 Agentic RAG Graph
        4. 先推送 trace（Agentic 执行轨迹）
        5. 逐字流式推送 reply
        6. 推送 done
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

    history.append({"role": "user", "content": req.message})

    input_state = {
        "user_message": req.message,
        "intent": None,
        "sub_queries": [],
        "retrieved_context": None,
        "kg_context": None,
        "confidence": 0.0,
        "evaluation_result": None,
        "retrieval_iterations": 0,
        "history": history,
        "summary": summary,
        "token_budget": token_budget,
        "messages": [],
        "reply": "",
        "trimmed": trimmed,
        "trace": [],
        "needs_tool": False,
        "tool_results": [],
    }

    chat_logger.debug(
        f"[{req_id}] 初始 state: history={len(history)}, summary={'有' if summary else '无'}, "
        f"budget={token_budget}"
    )

    # ── 2. SSE 生成器 ───────────────────────────────────────────────────────
    async def _stream_generator() -> AsyncGenerator[str, None]:
        try:
            # 执行完整 Graph
            result = await graph.ainvoke(input_state, config=config)
            reply = result.get("reply", "") or ""
            trace = result.get("trace", [])

            # 推送 trace
            if trace:
                yield _sse({"type": "trace", "payload": trace})
                await asyncio.sleep(0.01)

            if not reply:
                chat_logger.warning(f"[{req_id}] Graph 返回空 reply")
                yield _sse({"type": "content", "payload": "抱歉，没有生成回复。"})
                yield _sse({"type": "done"})
                return

            # 逐字流式推送（模拟 token 级流式，chunk_size=2 字符）
            chunk_size = 2
            for i in range(0, len(reply), chunk_size):
                yield _sse({"type": "content", "payload": reply[i : i + chunk_size]})
                if i < len(reply) - chunk_size:
                    await asyncio.sleep(0.005)

            elapsed = time.monotonic() - start
            final_history = result.get("history", [])
            final_summary = result.get("summary", "")
            chat_logger.info(
                f"[{req_id}] 请求完成: reply_len={len(reply)}, history={len(final_history)}, "
                f"summary={'有' if final_summary else '无'}, trace_steps={len(trace)}, 耗时={elapsed:.2f}s"
            )

            yield _sse({"type": "done", "payload": {"elapsed_sec": round(elapsed, 2)}})

        except Exception as e:
            chat_logger.error(f"[{req_id}] Graph 执行异常: {type(e).__name__}: {e}")
            yield _sse({"type": "error", "payload": str(e)})
            yield _sse({"type": "done"})

    return StreamingResponse(
        _stream_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
