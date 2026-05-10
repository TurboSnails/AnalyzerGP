"""
LangGraph 节点定义：将 ai_app1 的业务逻辑包装为 Graph 节点。

每个节点接收 state，返回需要更新的字段字典。
"""
from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ai_app2.core.config import SYSTEM_PROMPT, DEFAULT_TOKEN_BUDGET, MAX_HISTORY
from ai_app2.core.logger import graph_logger, retrieve_logger
from ai_app2.service.retriever import query_db
from ai_app2.service.tools import TOOLS


# ── 工具映射（供 Agent node 执行）───────────────────────────────────────────
_TOOL_MAP = {t.name: t for t in TOOLS}


def _estimate_tokens(messages: list[dict | Any]) -> int:
    """
    估算 token 数（复用 ai_app1 的加权字符数策略）。
    支持 dict 和 LangChain Message 对象。
    """
    total = 0
    for m in messages:
        if isinstance(m, dict):
            text = m.get("content", "") or ""
        else:
            text = getattr(m, "content", "") or ""
        cn_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other_chars = len(text) - cn_chars
        total += int(cn_chars * 1.5 + other_chars * 0.5)
    return total


# ═════════════════════════════════════════════════════════════════════════════
#  Node 1: retrieve_node  —— 混合检索（复用 ai_app1 完整管道）
# ═════════════════════════════════════════════════════════════════════════════
def retrieve_node(state: dict) -> dict:
    """
    基于 user_message 执行多路混合检索。
    复用 ai_app1 的 query_db：Dense → HyDE → BM25 → RRF → Rerank → Lost-in-Middle
    """
    query = state.get("user_message", "")
    if not query:
        retrieve_logger.warning("retrieve_node: user_message 为空，跳过检索")
        return {"retrieved_context": None}

    context = query_db(query)
    if context:
        retrieve_logger.info(f"检索完成: query={query[:30]!r}, len={len(context)}")
    else:
        retrieve_logger.info(f"未检索到结果: query={query[:30]!r}")

    return {"retrieved_context": context}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 2: build_messages_node  —— 构建 LLM messages
# ═════════════════════════════════════════════════════════════════════════════
def build_messages_node(state: dict) -> dict:
    """
    将 state 中的系统提示、摘要、历史、检索结果组装为 LangChain Message 列表。
    """
    messages: list[Any] = []

    # System prompt
    messages.append(SystemMessage(content=SYSTEM_PROMPT))

    # 历史摘要
    summary = state.get("summary", "")
    if summary:
        messages.append(HumanMessage(content=f"【历史摘要】{summary}"))

    # 原始对话历史（dict 格式 → HumanMessage / AIMessage）
    history = state.get("history", [])
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # 参考资料
    context = state.get("retrieved_context")
    if context:
        messages.append(HumanMessage(content=f"参考资料：{context}"))

    # 当前用户消息（检索节点中未加入，这里追加）
    user_msg = state.get("user_message", "")
    if user_msg:
        # 检查最后一条是否已经是当前用户消息（可能已在 history 中）
        if not history or history[-1].get("content") != user_msg:
            messages.append(HumanMessage(content=user_msg))

    graph_logger.debug(f"build_messages: {len(messages)} 条消息")
    return {"messages": messages}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 3: llm_node  —— LLM 调用 + 工具执行（手工 ReAct 循环）
# ═════════════════════════════════════════════════════════════════════════════
async def llm_node(state: dict, llm: Any) -> dict:
    """
    调用 LLM，处理 tool calling 多轮循环。
    由于 MiniMax API 的流式与工具调用格式兼容性问题，此处使用非流式调用，
    最终返回完整 reply 文本。
    """
    from ai_app2.core.config import MAX_STEPS

    messages = list(state.get("messages", []))
    run_id = id(messages)
    graph_logger.info(f"[{run_id}] llm_node 启动: messages={len(messages)}")

    for step in range(MAX_STEPS):
        graph_logger.debug(f"[{run_id}] Step {step + 1}/{MAX_STEPS}: 发送请求")

        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            graph_logger.error(f"[{run_id}] LLM 调用异常: {e}")
            return {"reply": f"Error: {e}"}

        content = response.content or ""
        tool_calls = getattr(response, "tool_calls", None) or []

        graph_logger.debug(
            f"[{run_id}] Step {step + 1} 响应: content_len={len(content)}, "
            f"tool_calls={len(tool_calls)}"
        )

        if not tool_calls:
            graph_logger.info(f"[{run_id}] llm_node 结束（无 tool_calls）: step={step + 1}")
            return {"reply": content}

        # ── 执行工具 ──
        graph_logger.info(f"[{run_id}] Step {step + 1} 工具调用: {len(tool_calls)} 个")

        # 追加 assistant 的 tool_calls 消息
        messages.append(response)

        for tc in tool_calls:
            func_name = tc.get("name") if isinstance(tc, dict) else tc.name
            func_args = tc.get("args") if isinstance(tc, dict) else tc.args

            graph_logger.debug(f"  - 工具: {func_name}, 参数: {func_args}")

            tool = _TOOL_MAP.get(func_name)
            if tool is None:
                result = f"Unknown tool: {func_name}"
                graph_logger.warning(f"[{run_id}] 未知工具: {func_name}")
            else:
                try:
                    result = await tool.ainvoke(func_args) if hasattr(tool, "ainvoke") else tool.invoke(func_args)
                    graph_logger.debug(f"[{run_id}] 工具执行成功: {func_name}({func_args}) = {result}")
                except Exception as e:
                    result = f"Error: {e}"
                    graph_logger.error(f"[{run_id}] 工具异常: {func_name}, error={e}")

            tool_call_id = tc.get("id") if isinstance(tc, dict) else tc.id
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

    graph_logger.warning(f"[{run_id}] llm_node 达到最大步数限制: {MAX_STEPS}")
    return {"reply": "Agent stopped: max steps reached"}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 4: summarize_node  —— 历史摘要压缩
# ═════════════════════════════════════════════════════════════════════════════
async def summarize_node(state: dict, llm: Any) -> dict:
    """
    将当前 history（含本轮 user + assistant）压缩为摘要，更新 summary。
    """
    history = state.get("history", [])
    if not history:
        graph_logger.debug("summarize_node: history 为空，跳过")
        return {"summary": state.get("summary", "")}

    history_text = "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in history
    )

    prompt = [
        SystemMessage(content="请总结以下对话的关键信息，用于后续对话参考。保持简洁。"),
        HumanMessage(content=history_text),
    ]

    try:
        response = await llm.ainvoke(prompt)
        summary = response.content or ""
        graph_logger.info(f"summarize 完成: len={len(summary)}")
        return {"summary": summary}
    except Exception as e:
        graph_logger.error(f"summarize 异常: {e}")
        return {"summary": state.get("summary", "")}


# ═════════════════════════════════════════════════════════════════════════════
#  Node 5: save_reply_node  —— 将 AI 回复追加到 history
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
#  Node 6: trim_node  —— 裁剪历史到 MAX_HISTORY 条
# ═════════════════════════════════════════════════════════════════════════════
def trim_node(state: dict) -> dict:
    """
    将 history 裁剪到 MAX_HISTORY 条，被裁掉的消息存入 trimmed。
    """
    history = state.get("history", [])
    trimmed = state.get("trimmed", [])

    if len(history) <= MAX_HISTORY:
        graph_logger.debug(f"无需裁剪: history_len={len(history)} <= {MAX_HISTORY}")
        return {}

    trimmed_count = len(history) - MAX_HISTORY
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
    token_budget = state.get("token_budget", DEFAULT_TOKEN_BUDGET)

    # 构建预估消息列表（system + summary + history + context）
    est_messages: list[Any] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if state.get("summary"):
        est_messages.append({"role": "user", "content": f"【历史摘要】{state['summary']}"})
    est_messages.extend(state.get("history", []))
    if state.get("retrieved_context"):
        est_messages.append({"role": "user", "content": f"参考资料：{state['retrieved_context']}"})

    total_tokens = _estimate_tokens(est_messages)
    result = total_tokens >= token_budget

    if result:
        graph_logger.info(f"触发 summarize: tokens={total_tokens}, budget={token_budget}")
        return "summarize"
    else:
        graph_logger.debug(f"跳过 summarize: tokens={total_tokens}, budget={token_budget}")
        return "trim"
