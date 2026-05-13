"""
LangGraph 节点定义：将 ai_app1 的业务逻辑包装为 Graph 节点。

每个节点接收 state，返回需要更新的字段字典。
所有组件通过 ai_app2.core.container.get_app_container() 懒加载，
确保容器在 main.py startup 初始化后可用。
"""
from __future__ import annotations

import re

from ai_app2.core.config import MAX_STEPS
from ai_app2.core.container import get_app_container
from ai_app2.core.logger import graph_logger, retrieve_logger
from ai_app2.service.retriever import query_db


# ═════════════════════════════════════════════════════════════════════════════
#  Node 1: retrieve_node  —— 混合检索（复用 rag_framework 完整管道）
# ═════════════════════════════════════════════════════════════════════════════
async def retrieve_node(state: dict) -> dict:
    """
    基于 user_message 执行多路混合检索（async）。
    复用 rag_framework 的 HybridRetriever：
    Rewrite → Classify → Dense → HyDE → BM25 → RRF → Rerank → Lost-in-Middle
    """
    query = state.get("user_message", "")
    history = state.get("history", [])
    if not query:
        retrieve_logger.warning("retrieve_node: user_message 为空，跳过检索")
        return {"retrieved_context": None}

    context = await query_db(query, history)
    if context:
        retrieve_logger.info(f"检索完成: query={query[:30]!r}, len={len(context)}")
    else:
        retrieve_logger.info(f"未检索到结果: query={query[:30]!r}")

    return {"retrieved_context": context}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 2: build_messages_node  —— 构建 LLM messages（OpenAI 格式）
# ═════════════════════════════════════════════════════════════════════════════
def build_messages_node(state: dict) -> dict:
    """
    将 state 中的系统提示、摘要、历史、检索结果组装为 OpenAI 格式消息列表。
    """
    container = get_app_container()
    domain = container.domain

    messages: list[dict] = []

    # System prompt
    messages.append({"role": "system", "content": domain.system_prompt})

    # 历史摘要
    summary = state.get("summary", "")
    if summary:
        messages.append({"role": "user", "content": f"【历史摘要】{summary}"})

    # 原始对话历史
    for h in state.get("history", []):
        messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})

    # 参考资料
    context = state.get("retrieved_context")
    if context:
        messages.append({"role": "user", "content": f"参考资料：{context}"})

    # 当前用户消息（若未在历史中则追加）
    user_msg = state.get("user_message", "")
    history = state.get("history", [])
    if user_msg and (not history or history[-1].get("content") != user_msg):
        messages.append({"role": "user", "content": user_msg})

    graph_logger.debug(f"build_messages: {len(messages)} 条消息")
    return {"messages": messages}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 3: llm_node  —— LLM 调用 + 工具执行（复用 OpenAILLMClient.run_agent）
# ═════════════════════════════════════════════════════════════════════════════
async def llm_node(state: dict) -> dict:
    """
    调用 LLM，处理 tool calling 多轮循环。
    复用 rag_framework.llm.openai_client.OpenAILLMClient.run_agent，
    支持非流式 tool calling 循环（Schema 由 tool_registry 统一管理）。
    """
    container = get_app_container()
    messages = list(state.get("messages", []))
    run_id = id(messages)
    graph_logger.info(f"[{run_id}] llm_node 启动: messages={len(messages)}")

    try:
        reply = await container.llm.run_agent(messages)
        graph_logger.info(f"[{run_id}] llm_node 结束: reply_len={len(reply)}")
        return {"reply": reply}
    except Exception as e:
        graph_logger.error(f"[{run_id}] LLM 调用异常: {e}")
        return {"reply": f"Error: {e}"}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 4: save_reply_node  —— 将 AI 回复追加到 history
# ═════════════════════════════════════════════════════════════════════════════
def save_reply_node(state: dict) -> dict:
    """
    将 llm_node 生成的 reply 追加到 history。
    该节点位于 llm 之后、should_summarize 之前，确保 summarize/trim
    能看到包含本轮 assistant 回复的完整 history。
    """
    reply = state.get("reply", "")
    history = list(state.get("history", []))
    if reply:
        history.append({"role": "assistant", "content": reply})
        graph_logger.debug(f"save_reply: assistant 消息入栈, history={len(history)}")
    return {"history": history}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 5: summarize_node  —— 历史摘要压缩
# ═════════════════════════════════════════════════════════════════════════════
async def summarize_node(state: dict) -> dict:
    """
    将当前 history（含本轮 user + assistant）压缩为摘要，更新 summary。
    复用 OpenAILLMClient.summarize 封装。
    """
    container = get_app_container()
    history = state.get("history", [])
    if not history:
        graph_logger.debug("summarize_node: history 为空，跳过")
        return {"summary": state.get("summary", "")}

    try:
        summary = await container.llm.summarize(history)
        graph_logger.info(f"summarize 完成: len={len(summary)}")
        return {"summary": summary}
    except Exception as e:
        graph_logger.error(f"summarize 异常: {e}")
        return {"summary": state.get("summary", "")}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 6: trim_node  —— 裁剪历史到 max_history 条
# ═════════════════════════════════════════════════════════════════════════════
def trim_node(state: dict) -> dict:
    """
    将 history 裁剪到 max_history 条，被裁掉的消息存入 trimmed。
    """
    container = get_app_container()
    max_h = container.settings.max_history

    history = state.get("history", [])
    trimmed = state.get("trimmed", [])

    if len(history) <= max_h:
        graph_logger.debug(f"无需裁剪: history_len={len(history)} <= {max_h}")
        return {}

    trimmed_count = len(history) - max_h
    new_trimmed = history[:trimmed_count]
    new_history = history[trimmed_count:]

    graph_logger.info(
        f"裁剪历史: 裁掉 {trimmed_count} 条, 剩余 {len(new_history)} 条, "
        f"trimmed +{len(new_trimmed)}"
    )

    return {
        "history": new_history,
        "trimmed": trimmed + new_trimmed,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  条件边: should_summarize
# ═════════════════════════════════════════════════════════════════════════════
def should_summarize(state: dict) -> str:
    """
    根据 token 预算判断是否需要 summarize。
    在 AI 回复后调用，因此 state 中的 history 已包含本轮对话。
    """
    container = get_app_container()
    settings = container.settings
    domain = container.domain

    # 构建预估消息列表（system + summary + history + context）
    est_messages: list[dict] = [{"role": "system", "content": domain.system_prompt}]
    if state.get("summary"):
        est_messages.append({"role": "user", "content": f"【历史摘要】{state['summary']}"})
    est_messages.extend(state.get("history", []))
    if state.get("retrieved_context"):
        est_messages.append({"role": "user", "content": f"参考资料：{state['retrieved_context']}"})

    total_tokens = 0
    for m in est_messages:
        text = m.get("content", "") or ""
        total_tokens += domain.estimate_tokens(text)

    token_budget = state.get("token_budget", settings.default_token_budget)
    result = total_tokens >= token_budget

    if result:
        graph_logger.info(f"触发 summarize: tokens={total_tokens}, budget={token_budget}")
        return "summarize"
    else:
        graph_logger.debug(f"跳过 summarize: tokens={total_tokens}, budget={token_budget}")
        return "trim"
