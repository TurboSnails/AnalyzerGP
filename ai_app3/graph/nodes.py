"""
Agentic RAG 节点定义 — 从 ai_app2 的线性 pipeline 升级为迭代式 Self-RAG。

节点列表：
  intent_node      : 意图分析
  decompose_node   : 查询分解
  retrieve_node    : 多路子查询检索（复用 ai_app1）+ KG 扩展
  evaluate_node    : 检索质量评估
  rewrite_node     : 查询改写（迭代优化）
  expand_kg_node   : 知识图谱扩展（补充实体关系信息）
  build_messages   : 组装 LLM messages
  llm_node         : LLM 生成 + Tool calling
  direct_response  : 闲聊/无需检索时的直接回复
  self_check_node  : 回答自检（Self-RAG）
  save_reply_node  : 保存回复到 history
  summarize_node   : 历史摘要压缩
  trim_node        : 裁剪历史
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from ai_app3.core.config import SYSTEM_PROMPT, DEFAULT_TOKEN_BUDGET, MAX_HISTORY, MAX_STEPS
from ai_app3.core.logger import graph_logger, retrieve_logger, eval_logger
from ai_app3.service.query_engine import intent_analysis, decompose_query, rewrite_query, merge_contexts
from ai_app3.service.evaluator import evaluate_retrieval, decide_next_step
from ai_app3.service.context_compressor import compress_context, extract_key_facts, build_prompt_context
from ai_app3.service.knowledge_graph import expand_by_entities, fetch_docs_by_ids
from ai_app3.service.tools import TOOLS, TOOL_MAP
from ai_app3.service.retriever import query_context as query_db


def _estimate_tokens(messages: list[dict | Any]) -> int:
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


def _add_trace(state: dict, step: str, detail: dict) -> list[dict]:
    trace = list(state.get("trace", []))
    trace.append({"step": step, **detail})
    return trace


# ═════════════════════════════════════════════════════════════════════════════
# Node: intent_node
# ═════════════════════════════════════════════════════════════════════════════
def intent_node(state: dict) -> dict:
    query = state.get("user_message", "")
    result = intent_analysis(query)
    return {
        "intent": result,
        "trace": _add_trace(state, "intent", result),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: decompose_node
# ═════════════════════════════════════════════════════════════════════════════
def decompose_node(state: dict) -> dict:
    query = state.get("user_message", "")
    subs = decompose_query(query)
    return {
        "sub_queries": subs,
        "trace": _add_trace(state, "decompose", {"sub_queries": subs}),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: retrieve_node  —— 并行多路子查询检索 + KG 扩展
# ═════════════════════════════════════════════════════════════════════════════
async def retrieve_node(state: dict) -> dict:
    subs = state.get("sub_queries", [])
    if not subs:
        subs = [{"sub_query": state.get("user_message", ""), "confidence": 1.0}]

    valid_queries = [s.get("sub_query", "") for s in subs if s.get("sub_query")]

    # 子查询并发检索（asyncio.gather）
    raw_results = await asyncio.gather(
        *[query_db(q) for q in valid_queries],
        return_exceptions=True,
    )

    contexts: list[str | None] = []
    for q, ctx in zip(valid_queries, raw_results):
        if isinstance(ctx, Exception):
            retrieve_logger.warning(f"子查询检索异常: {q[:30]!r}: {ctx}")
            contexts.append(None)
        else:
            if ctx:
                retrieve_logger.info(f"子查询检索: {q[:30]!r}, len={len(ctx)}")
            contexts.append(ctx)

    merged = merge_contexts(contexts)

    # 知识图谱扩展（若合并结果较空）
    kg_ctx = None
    if not merged or len(merged) < 200:
        kg_ids = expand_by_entities(state.get("user_message", ""), top_k=5)
        if kg_ids:
            kg_ctx = fetch_docs_by_ids(kg_ids)
            if kg_ctx:
                retrieve_logger.info(f"KG 扩展补充: {len(kg_ctx)} 字符")

    return {
        "retrieved_context": merged,
        "kg_context": kg_ctx,
        "trace": _add_trace(state, "retrieve", {
            "sub_queries": len(subs),
            "merged_len": len(merged) if merged else 0,
            "kg_len": len(kg_ctx) if kg_ctx else 0,
        }),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: evaluate_node  —— 检索质量评估
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_node(state: dict) -> dict:
    query = state.get("user_message", "")
    ctx = state.get("retrieved_context") or state.get("kg_context")
    result = evaluate_retrieval(query, ctx)
    return {
        "confidence": float(result.get("confidence", 0)),
        "evaluation_result": result,
        "trace": _add_trace(state, "evaluate", result),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: rewrite_node  —— 查询改写迭代
# ═════════════════════════════════════════════════════════════════════════════
def rewrite_node(state: dict) -> dict:
    iteration = state.get("retrieval_iterations", 0)
    query = state.get("user_message", "")
    feedback = f"前次检索置信度不足，请换用更通用的 Android 技术术语重新表达。"
    rewritten = rewrite_query(query, feedback)

    # 更新子查询为改写后的查询
    new_subs = [{"sub_query": rewritten, "confidence": 0.8, "reason": f"改写迭代 {iteration + 1}"}]

    return {
        "user_message": rewritten,  # 注意：这里改写的是内部检索用查询，不影响原始 user_message
        "sub_queries": new_subs,
        "retrieval_iterations": iteration + 1,
        "trace": _add_trace(state, "rewrite", {"rewritten": rewritten, "iteration": iteration + 1}),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: expand_kg_node  —— 知识图谱扩展
# ═════════════════════════════════════════════════════════════════════════════
def expand_kg_node(state: dict) -> dict:
    query = state.get("user_message", "")
    kg_ids = expand_by_entities(query, top_k=8)
    kg_ctx = fetch_docs_by_ids(kg_ids)

    # 合并到现有上下文
    existing = state.get("retrieved_context") or ""
    if kg_ctx:
        if existing:
            merged = existing + "\n\n【知识图谱补充】\n" + kg_ctx
        else:
            merged = kg_ctx
    else:
        merged = existing or None

    return {
        "retrieved_context": merged,
        "kg_context": kg_ctx,
        "trace": _add_trace(state, "expand_kg", {"kg_docs": len(kg_ids), "kg_len": len(kg_ctx) if kg_ctx else 0}),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: build_messages
# ═════════════════════════════════════════════════════════════════════════════
def build_messages(state: dict) -> dict:
    messages: list[Any] = []
    messages.append(SystemMessage(content=SYSTEM_PROMPT))

    summary = state.get("summary", "")
    if summary:
        messages.append(HumanMessage(content=f"【历史摘要】{summary}"))

    history = state.get("history", [])
    for h in history:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # 参考资料（压缩后）
    ctx = state.get("retrieved_context")
    kg = state.get("kg_context")
    combined = ctx or ""
    if kg and combined:
        combined += "\n\n【知识图谱补充】\n" + kg
    elif kg:
        combined = kg

    if combined:
        compressed = compress_context(state.get("user_message", ""), combined)
        facts = extract_key_facts(compressed)
        prompt_ctx = build_prompt_context(state.get("user_message", ""), compressed, facts)
        messages.append(HumanMessage(content=prompt_ctx))

    user_msg = state.get("user_message", "")
    if user_msg:
        if not history or history[-1].get("content") != user_msg:
            messages.append(HumanMessage(content=user_msg))

    graph_logger.debug(f"build_messages: {len(messages)} 条消息")
    return {"messages": messages}


# ═════════════════════════════════════════════════════════════════════════════
# Node: llm_node  —— LLM + Tool calling
# ═════════════════════════════════════════════════════════════════════════════
async def llm_node(state: dict, llm: Any) -> dict:
    messages = list(state.get("messages", []))
    run_id = id(messages)
    graph_logger.info(f"[{run_id}] llm_node 启动: messages={len(messages)}")

    for step in range(MAX_STEPS):
        graph_logger.debug(f"[{run_id}] Step {step + 1}/{MAX_STEPS}")
        try:
            response = await llm.ainvoke(messages)
        except Exception as e:
            graph_logger.error(f"[{run_id}] LLM 调用异常: {e}")
            return {"reply": f"Error: {e}"}

        content = response.content or ""
        tool_calls = getattr(response, "tool_calls", None) or []

        if not tool_calls:
            graph_logger.info(f"[{run_id}] llm_node 结束: step={step + 1}")
            return {"reply": content}

        messages.append(response)
        for tc in tool_calls:
            func_name = tc.get("name") if isinstance(tc, dict) else tc.name
            func_args = tc.get("args") if isinstance(tc, dict) else tc.args
            tool = TOOL_MAP.get(func_name)
            if tool is None:
                result = f"Unknown tool: {func_name}"
                graph_logger.warning(f"[{run_id}] 未知工具: {func_name}")
            else:
                try:
                    result = await tool.ainvoke(func_args) if hasattr(tool, "ainvoke") else tool.invoke(func_args)
                except Exception as e:
                    result = f"Error: {e}"
                    graph_logger.error(f"[{run_id}] 工具异常: {func_name}, error={e}")

            tool_call_id = tc.get("id") if isinstance(tc, dict) else tc.id
            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call_id))

    graph_logger.warning(f"[{run_id}] llm_node 达到最大步数")
    return {"reply": "Agent stopped: max steps reached"}


# ═════════════════════════════════════════════════════════════════════════════
# Node: direct_response  —— 闲聊/无需检索时的快速回复
# ═════════════════════════════════════════════════════════════════════════════
async def direct_response(state: dict, llm: Any) -> dict:
    query = state.get("user_message", "")
    messages = [
        SystemMessage(content="你是 Android 开发助手，也是一个友好的对话伙伴。"),
        HumanMessage(content=query),
    ]
    try:
        resp = await llm.ainvoke(messages)
        return {"reply": resp.content or "", "trace": _add_trace(state, "direct", {"reason": "闲聊/无需检索"})}
    except Exception as e:
        return {"reply": f"Error: {e}"}


# ═════════════════════════════════════════════════════════════════════════════
# Node: self_check_node  —— Self-RAG 回答自检
# ═════════════════════════════════════════════════════════════════════════════
def self_check_node(state: dict) -> dict:
    """
    简单启发式自检，评估回答是否基于检索上下文。
    返回 trace，不阻断流程（保留扩展点供后续强化）。
    """
    reply = state.get("reply", "")
    ctx = state.get("retrieved_context") or ""
    # 若回答包含 "Error" 或为空，标记问题
    issues = []
    if not reply or reply.strip() == "":
        issues.append("回答为空")
    if reply.startswith("Error:"):
        issues.append("生成异常")
    if ctx and len(reply) < 20:
        issues.append("回答过短，可能未充分利用上下文")

    return {
        "trace": _add_trace(state, "self_check", {"issues": issues, "passed": len(issues) == 0}),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: save_reply_node
# ═════════════════════════════════════════════════════════════════════════════
def save_reply_node(state: dict) -> dict:
    reply = state.get("reply", "")
    history = list(state.get("history", []))
    if reply:
        history.append({"role": "assistant", "content": reply})
        graph_logger.debug(f"save_reply: assistant 消息入栈, history={len(history)}")
    return {"history": history}


# ═════════════════════════════════════════════════════════════════════════════
# Node: summarize_node
# ═════════════════════════════════════════════════════════════════════════════
async def summarize_node(state: dict, llm: Any) -> dict:
    history = state.get("history", [])
    if not history:
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
# Node: trim_node
# ═════════════════════════════════════════════════════════════════════════════
def trim_node(state: dict) -> dict:
    history = state.get("history", [])
    trimmed = state.get("trimmed", [])
    if len(history) <= MAX_HISTORY:
        return {}
    trimmed_count = len(history) - MAX_HISTORY
    new_trimmed = history[:trimmed_count]
    new_history = history[trimmed_count:]
    graph_logger.info(f"裁剪历史: 裁掉 {trimmed_count} 条, 剩余 {len(new_history)} 条")
    return {
        "history": new_history,
        "trimmed": trimmed + new_trimmed,
    }
