"""
ai_app4 Chat API 路由。

提供：
  - POST /chat        — SSE 流式响应（trace + content + done）
  - POST /chat/json   — 非流式 JSON 响应
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ai_app4.core.config import CS4Settings
from ai_app4.core.container import CS4Container
from ai_app4.graph.builder import graph

router = APIRouter()


class ChatRequest(BaseModel):
    """POST /chat 请求体"""
    message: str
    user_id: str = "default_user"
    tenant_id: str = "default"


def _make_thread_id(user_id: str) -> str:
    return f"cs4_thread_{user_id}"


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(req: ChatRequest):
    """
    流式处理用户聊天请求（SSE）。

    流程：
        1. 从 checkpointer 加载当前线程状态
        2. 组装 input_state
        3. 调用 graph.ainvoke 执行完整 CS4 Graph
        4. 先推送 trace（Agentic 执行轨迹）
        5. 逐字流式推送 reply
        6. 推送 done
    """
    req_id = id(req)
    thread_id = _make_thread_id(req.user_id)
    config = {"configurable": {"thread_id": thread_id}}

    # 获取容器和设置（优先从全局上下文获取，避免循环导入）
    from ai_app4.core import context
    container: CS4Container = context.get_container()
    settings: CS4Settings = context.get_settings()
    if container is None:
        container = CS4Container.from_settings(CS4Settings())
        context.set_container(container)
    if settings is None:
        settings = CS4Settings()
        context.set_settings(settings)

    # 从 checkpoint 加载状态
    current_state = graph.get_state(config)
    if current_state and current_state.values:
        vals = current_state.values
        history = list(vals.get("history", []))
        summary = vals.get("summary", "")
        token_budget = vals.get("token_budget", settings.default_token_budget)
        trimmed = list(vals.get("trimmed", []))
    else:
        history, summary, token_budget, trimmed = [], "", settings.default_token_budget, []

    input_state = {
        "user_message": req.message,
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "intent": "",
        "intent_score": 0.0,
        "sentiment": "neutral",
        "sentiment_score": 0.5,
        "entities": [],
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
        "escalation_triggered": False,
        "escalation_reason": "",
        "agent_id": None,
        "trace": [],
        "needs_tool": False,
        "tool_results": [],
    }

    # 执行 Graph
    start = time.monotonic()
    result = await graph.ainvoke(input_state, config=config)
    latency_ms = (time.monotonic() - start) * 1000

    reply = result.get("reply", "")
    trace = result.get("trace", [])

    async def _stream() -> AsyncGenerator[str, None]:
        # 1. 推送 trace
        yield _sse({"type": "trace", "data": trace, "latency_ms": round(latency_ms, 1)})
        await asyncio.sleep(0.01)

        # 2. 逐字推送 content
        for char in reply:
            yield _sse({"type": "content", "data": char})
            await asyncio.sleep(0.005)  # 模拟打字机效果

        # 3. 推送 done
        yield _sse({"type": "done"})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/chat/json")
async def chat_json(req: ChatRequest):
    """非流式 JSON 响应（用于调试或低延迟场景）。"""
    thread_id = _make_thread_id(req.user_id)
    config = {"configurable": {"thread_id": thread_id}}

    from ai_app4.core import context
    container: CS4Container = context.get_container()
    settings: CS4Settings = context.get_settings()
    if container is None:
        container = CS4Container.from_settings(CS4Settings())
        context.set_container(container)
    if settings is None:
        settings = CS4Settings()
        context.set_settings(settings)

    current_state = graph.get_state(config)
    if current_state and current_state.values:
        vals = current_state.values
        history = list(vals.get("history", []))
        summary = vals.get("summary", "")
        token_budget = vals.get("token_budget", settings.default_token_budget)
        trimmed = list(vals.get("trimmed", []))
    else:
        history, summary, token_budget, trimmed = [], "", settings.default_token_budget, []

    input_state = {
        "user_message": req.message,
        "tenant_id": req.tenant_id,
        "user_id": req.user_id,
        "intent": "",
        "intent_score": 0.0,
        "sentiment": "neutral",
        "sentiment_score": 0.5,
        "entities": [],
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
        "escalation_triggered": False,
        "escalation_reason": "",
        "agent_id": None,
        "trace": [],
        "needs_tool": False,
        "tool_results": [],
    }

    start = time.monotonic()
    result = await graph.ainvoke(input_state, config=config)
    latency_ms = (time.monotonic() - start) * 1000

    return {
        "reply": result.get("reply", ""),
        "intent": result.get("intent", ""),
        "sentiment": result.get("sentiment", ""),
        "entities": result.get("entities", []),
        "escalation_triggered": result.get("escalation_triggered", False),
        "trace": result.get("trace", []),
        "latency_ms": round(latency_ms, 1),
    }
