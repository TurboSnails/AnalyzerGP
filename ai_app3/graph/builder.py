"""
Agentic RAG StateGraph 构建器。

流程（第三代 RAG）：

  START ──→ intent ──┬──→ decompose ──→ retrieve ──→ evaluate ──┬──→ generate
                     │                                           │
                     │                                           ├──→ rewrite ──→ retrieve (循环)
                     │                                           │
                     │                                           └──→ expand_kg ──→ evaluate (循环)
                     │
                     └──→ direct_response ───────────────────────────────┐
                                                                       │
   generate: build_messages ──→ llm ──→ self_check ──→ save_reply ──────┤
                                                                       │
   save_reply ──→ should_summarize? ──┬──→ summarize ──→ trim ──→ END
                                     └──→ trim ──→ END
"""
from __future__ import annotations

import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from ai_app3.core.config import DEFAULT_TOKEN_BUDGET, SYSTEM_PROMPT
from ai_app3.core.llm_provider import get_chat_llm
from ai_app3.core.logger import graph_logger
from ai_app3.graph.state import RagState
from ai_app3.graph.nodes import (
    intent_node,
    decompose_node,
    retrieve_node,
    evaluate_node,
    rewrite_node,
    expand_kg_node,
    build_messages,
    llm_node,
    direct_response,
    self_check_node,
    save_reply_node,
    summarize_node,
    trim_node,
)
from ai_app3.graph.conditional_edges import after_intent, after_evaluate

# ── LLM 实例（主 LLM，用于对话/生成/摘要）───────────────────────────────
_base_llm = get_chat_llm(temperature=0.3)

# 为了让 builder.py 不依赖 TOOLS 的循环导入，我们在 tools.py 里已定义。
# 但 _llm 实际在节点内部通过参数传入，这里先不 bind（因为 llm_node 内部使用 TOOL_MAP 手工执行）
# 注意：ai_app2 使用 bind_tools；ai_app3 继续使用 bind_tools 让 LLM 知道工具定义
from ai_app3.service.tools import TOOLS
_llm = _base_llm.bind_tools(TOOLS)


# ── 包装异步节点 ──────────────────────────────────────────────────────────
async def _llm_wrapper(state: RagState):
    return await llm_node(state, _llm)


async def _summarize_wrapper(state: RagState):
    return await summarize_node(state, _llm)


async def _direct_wrapper(state: RagState):
    return await direct_response(state, _llm)


# ── 条件边: should_summarize ──────────────────────────────────────────────
def should_summarize(state: dict) -> str:
    token_budget = state.get("token_budget", DEFAULT_TOKEN_BUDGET)
    est_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if state.get("summary"):
        est_messages.append({"role": "user", "content": f"【历史摘要】{state['summary']}"})
    est_messages.extend(state.get("history", []))
    if state.get("retrieved_context"):
        est_messages.append({"role": "user", "content": f"参考资料：{state['retrieved_context']}"})

    total = 0
    for m in est_messages:
        text = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "") or ""
        cn = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        total += int(cn * 1.5 + (len(text) - cn) * 0.5)

    result = total >= token_budget
    if result:
        graph_logger.info(f"触发 summarize: tokens={total}, budget={token_budget}")
        return "summarize"
    graph_logger.debug(f"跳过 summarize: tokens={total}, budget={token_budget}")
    return "trim"


# ── 构建 StateGraph ───────────────────────────────────────────────────────
builder = StateGraph(RagState)

builder.add_node("intent", intent_node)
builder.add_node("decompose", decompose_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("expand_kg", expand_kg_node)
builder.add_node("build_messages", build_messages)
builder.add_node("llm", _llm_wrapper)
builder.add_node("self_check", self_check_node)
builder.add_node("save_reply", save_reply_node)
builder.add_node("summarize", _summarize_wrapper)
builder.add_node("trim", trim_node)
builder.add_node("direct_response", _direct_wrapper)

builder.set_entry_point("intent")

# 意图分支：技术问答 → decompose；闲聊 → direct_response
builder.add_conditional_edges(
    "intent",
    after_intent,
    {"decompose": "decompose", "direct_response": "direct_response"},
)

# 检索主链路
builder.add_edge("decompose", "retrieve")
builder.add_edge("retrieve", "evaluate")

# 评估后分支：充分 → 生成；不足 → 改写 或 知识图谱扩展
builder.add_conditional_edges(
    "evaluate",
    after_evaluate,
    {"generate": "build_messages", "rewrite": "rewrite", "expand_kg": "expand_kg"},
)

# 改写与 KG 扩展的循环
builder.add_edge("rewrite", "retrieve")
builder.add_edge("expand_kg", "evaluate")

# 生成链路
builder.add_edge("build_messages", "llm")
builder.add_edge("llm", "self_check")
builder.add_edge("self_check", "save_reply")

# 直接回复链路
builder.add_edge("direct_response", "save_reply")

# 后续：摘要/裁剪
builder.add_conditional_edges(
    "save_reply",
    should_summarize,
    {"summarize": "summarize", "trim": "trim"},
)
builder.add_edge("summarize", "trim")
builder.add_edge("trim", END)

# 编译图
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
graph_logger.info(
    "Agentic RAG Graph 编译完成: nodes=["
    "intent,decompose,retrieve,evaluate,rewrite,expand_kg,"
    "build_messages,llm,self_check,save_reply,summarize,trim,direct_response"
    "]"
)
