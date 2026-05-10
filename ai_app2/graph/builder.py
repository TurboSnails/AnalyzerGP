"""
LangGraph 状态图构建器。

流程：
    retrieve → build_messages → llm → save_reply ──→ should_summarize ──→ summarize ──→ trim ──→ END
                                                     │
                                                     └────→ trim ──→ END

状态管理使用 MemorySaver（内存 checkpointer），后续可替换为 Redis/Postgres。
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from ai_app2.core.config import OPENAI_API_KEY
from ai_app2.core.logger import graph_logger
from ai_app2.graph.state import RagState
from ai_app2.graph.nodes import (
    retrieve_node,
    build_messages_node,
    llm_node,
    save_reply_node,
    summarize_node,
    trim_node,
    should_summarize,
)
from ai_app2.service.tools import TOOLS


if not OPENAI_API_KEY:
    graph_logger.warning("OPENAI_API_KEY 未设置，LLM 调用将会失败")

# ── LLM 实例（复用 MiniMax 兼容端点，绑定工具）───────────────────────────────
_llm = ChatOpenAI(
    model="MiniMax-M2.7",
    base_url="https://api.minimaxi.com/v1",
    api_key=OPENAI_API_KEY or "",
    temperature=0.3,
).bind_tools(TOOLS)

# ── 包装异步节点（LangGraph 支持 async 节点）─────────────────────────────────
async def _llm_node_wrapper(state: RagState):
    return await llm_node(state, _llm)

async def _summarize_node_wrapper(state: RagState):
    return await summarize_node(state, _llm)

# ── 构建 StateGraph ─────────────────────────────────────────────────────────
builder = StateGraph(RagState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("build_messages", build_messages_node)
builder.add_node("llm", _llm_node_wrapper)
builder.add_node("save_reply", save_reply_node)
builder.add_node("summarize", _summarize_node_wrapper)
builder.add_node("trim", trim_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "build_messages")
builder.add_edge("build_messages", "llm")
builder.add_edge("llm", "save_reply")
builder.add_conditional_edges(
    "save_reply",
    should_summarize,
    {
        "summarize": "summarize",
        "trim": "trim",
    },
)
builder.add_edge("summarize", "trim")
builder.add_edge("trim", END)

# 编译图：使用内存 checkpointer 保存会话状态
graph = builder.compile(checkpointer=MemorySaver())
graph_logger.info("LangGraph 编译完成: nodes=['retrieve','build_messages','llm','save_reply','summarize','trim']")
