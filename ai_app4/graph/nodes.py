"""
ai_app4 LangGraph 节点实现。

每个节点接收 CS4State，返回 state 的部分更新字典。
所有异步操作（模型推理、检索、LLM 调用）通过 await 处理。
"""
from __future__ import annotations

import time
from typing import Any

from ai_app4.core.config import CS4Settings
from ai_app4.core.container import CS4Container
from ai_app4.core import context
from ai_app4.graph.state import CS4State


# ── 辅助：获取容器 ──────────────────────────────────────────────────────────

def _get_container() -> CS4Container:
    """从全局上下文获取容器（避免循环导入 main.py）。"""
    container = context.get_container()
    if container is None:
        # fallback：直接构建（用于测试或独立运行节点）
        container = CS4Container.from_settings(CS4Settings())
        context.set_container(container)
    return container


def _get_settings() -> CS4Settings:
    settings = context.get_settings()
    if settings is None:
        settings = CS4Settings()
        context.set_settings(settings)
    return settings


# ── classify 节点 ───────────────────────────────────────────────────────────

async def classify_node(state: CS4State) -> dict[str, Any]:
    """
    意图分类 + 情感分析 + NER。

    调用 PyTorch 任务模型（若已启用），否则使用简单启发式规则。
    """
    container = _get_container()
    settings = _get_settings()
    text = state["user_message"]
    trace = list(state.get("trace", []))
    start = time.monotonic()

    intent = "general_inquiry"
    intent_score = 0.0
    sentiment = "neutral"
    sentiment_score = 0.5
    entities: list[dict] = []

    if settings.enable_torch_models:
        # 意图分类
        intent_model = container.get_torch_model("intent_classification")
        if intent_model is not None:
            pred = await intent_model.predict(text)
            intent = pred.label
            intent_score = pred.score

        # 情感分析
        sentiment_model = container.get_torch_model("sentiment_analysis")
        if sentiment_model is not None:
            pred = await sentiment_model.predict(text)
            sentiment = pred.label
            sentiment_score = pred.score

        # NER
        ner_model = container.get_torch_model("ner")
        if ner_model is not None:
            pred = await ner_model.predict(text)
            entities = pred.details.get("entities", [])

    # 简单启发式兜底
    if intent == "general_inquiry" and any(kw in text for kw in ("人工", "客服", "转接", "投诉")):
        intent = "escalation_request"
        intent_score = max(intent_score, 0.7)
    if sentiment == "neutral" and any(kw in text for kw in ("垃圾", "差", "慢", "崩溃", "愤怒")):
        sentiment = "negative"
        sentiment_score = max(sentiment_score, 0.7)

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "classify",
        "intent": intent,
        "sentiment": sentiment,
        "entities_count": len(entities),
        "latency_ms": round(latency_ms, 1),
    })

    # 转人工判断
    escalation_triggered = container.is_escalation_needed(intent, sentiment)

    return {
        "intent": intent,
        "intent_score": intent_score,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "entities": entities,
        "escalation_triggered": escalation_triggered,
        "escalation_reason": intent if escalation_triggered else "",
        "trace": trace,
    }


# ── retrieve 节点 ───────────────────────────────────────────────────────────

async def retrieve_node(state: CS4State) -> dict[str, Any]:
    """执行检索：优先 LlamaIndex，fallback 到 HybridRetriever。"""
    container = _get_container()
    text = state["user_message"]
    trace = list(state.get("trace", []))
    start = time.monotonic()

    context = None
    # 优先 LlamaIndex
    li_retriever = container.get_llamaindex_retriever()
    if li_retriever is not None:
        try:
            result = await li_retriever.retrieve(text, top_k=5)
            if result.docs:
                context = "\n\n".join(d.text for d in result.docs if d.text)
        except Exception as exc:
            trace.append({"node": "retrieve", "error": f"LlamaIndex: {exc}"})

    # fallback 到 HybridRetriever
    if not context and container.retriever is not None:
        try:
            routes = container.build_routes(text, state.get("history", []))
            result = await container.retriever.retrieve(routes, top_k=5)
            if result.docs:
                context = "\n\n".join(d.text for d in result.docs if d.text)
        except Exception as exc:
            trace.append({"node": "retrieve", "error": f"Hybrid: {exc}"})

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "retrieve",
        "has_context": bool(context),
        "latency_ms": round(latency_ms, 1),
    })

    iterations = state.get("retrieval_iterations", 0) + 1
    return {
        "retrieved_context": context,
        "retrieval_iterations": iterations,
        "trace": trace,
    }


# ── evaluate 节点 ───────────────────────────────────────────────────────────

async def evaluate_node(state: CS4State) -> dict[str, Any]:
    """检索质量评估（简化版，可扩展为 LLM-based 评估）。"""
    context = state.get("retrieved_context")
    trace = list(state.get("trace", []))

    confidence = 0.0
    if context:
        # 启发式：上下文长度 / 检索迭代次数
        confidence = min(0.5 + len(context) / 2000, 0.95)
        if state.get("retrieval_iterations", 0) > 1:
            confidence *= 0.9  # 改写后仍不理想，降低置信度

    trace.append({
        "node": "evaluate",
        "confidence": round(confidence, 2),
    })

    return {
        "confidence": confidence,
        "evaluation_result": {"confidence": confidence},
        "trace": trace,
    }


# ── rewrite 节点 ───────────────────────────────────────────────────────────

async def rewrite_node(state: CS4State) -> dict[str, Any]:
    """查询改写（调用 rewriter LLM）。"""
    container = _get_container()
    text = state["user_message"]
    trace = list(state.get("trace", []))

    rewritten = text
    if container.llm_rewriter is not None:
        try:
            from rag_framework.retrieval.query_rewriter.llm_rewriter import LLMQueryRewriter
            # 简化：直接调用 rewriter 的同步方法（实际应使用 async）
            # 此处仅做演示，实际实现应调用 rewriter 的 async 接口
            rewritten = f"【改写】{text}"
        except Exception:
            pass

    trace.append({"node": "rewrite", "original": text, "rewritten": rewritten})
    return {"user_message": rewritten, "trace": trace}


# ── generate 节点 ───────────────────────────────────────────────────────────

async def generate_node(state: CS4State) -> dict[str, Any]:
    """调用 LLM 生成回复。"""
    container = _get_container()
    settings = _get_settings()
    trace = list(state.get("trace", []))
    start = time.monotonic()

    # 构建 messages
    messages: list[dict] = []
    if hasattr(settings, "cs_system_prompt"):
        messages.append({"role": "system", "content": settings.cs_system_prompt})
    else:
        messages.append({"role": "system", "content": "你是客服助手。"})

    # 历史摘要
    summary = state.get("summary", "")
    if summary:
        messages.append({"role": "user", "content": f"【历史摘要】{summary}"})

    # 历史消息
    for h in state.get("history", [])[-settings.max_history_per_session:]:
        messages.append(h)

    # 检索上下文
    context = state.get("retrieved_context")
    if context:
        messages.append({"role": "user", "content": f"参考资料：{context}"})

    # 用户消息
    messages.append({"role": "user", "content": state["user_message"]})

    # 调用 LLM
    reply = ""
    try:
        reply = await container.llm.chat(messages)
    except Exception as exc:
        reply = f"抱歉，服务暂时异常，请稍后重试。（{exc}）"
        trace.append({"node": "generate", "error": str(exc)})

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "generate",
        "reply_length": len(reply),
        "latency_ms": round(latency_ms, 1),
    })

    return {"reply": reply, "trace": trace}


# ── escalate 节点 ───────────────────────────────────────────────────────────

async def escalate_node(state: CS4State) -> dict[str, Any]:
    """转人工节点：生成安抚话术并标记待交接。"""
    trace = list(state.get("trace", []))
    intent = state.get("intent", "")
    sentiment = state.get("sentiment", "")

    reply = (
        "非常理解您的心情，我已为您安排人工客服介入，"
        "请稍等片刻，专员将尽快为您服务。"
    )
    if sentiment == "negative":
        reply = (
            "非常抱歉给您带来了不好的体验，您的反馈我们非常重视。"
            "已为您转接人工客服，专员将尽快为您处理。"
        )

    trace.append({
        "node": "escalate",
        "intent": intent,
        "sentiment": sentiment,
    })

    return {
        "reply": reply,
        "escalation_triggered": True,
        "trace": trace,
    }


# ── handoff 节点 ───────────────────────────────────────────────────────────

async def handoff_node(state: CS4State) -> dict[str, Any]:
    """坐席交接节点：将对话标记为待人工接管。"""
    trace = list(state.get("trace", []))
    trace.append({"node": "handoff", "status": "waiting_for_agent"})

    return {
        "agent_id": None,
        "trace": trace,
    }


# ── save_reply 节点 ─────────────────────────────────────────────────────────

async def save_reply_node(state: CS4State) -> dict[str, Any]:
    """保存回复到会话历史。"""
    history = list(state.get("history", []))
    reply = state.get("reply", "")
    user_message = state.get("user_message", "")

    if user_message:
        history.append({"role": "user", "content": user_message})
    if reply:
        history.append({"role": "assistant", "content": reply})

    # Token 预算检查（简化版）
    token_budget = state.get("token_budget", 4096)
    summary = state.get("summary", "")
    trimmed = list(state.get("trimmed", []))

    # 若历史过长，裁剪旧消息
    max_history = _get_settings().max_history_per_session
    while len(history) > max_history * 2:
        old = history.pop(0)
        trimmed.append(old)

    return {
        "history": history,
        "summary": summary,
        "trimmed": trimmed,
        "token_budget": token_budget,
    }
