"""
ai_app4 LangGraph StateGraph 构建器。

流程：
  START ──→ classify ──┬──→ escalate ──→ handoff ──→ END
                       │
                       ├──→ chitchat ──→ generate ──→ save ──→ END
                       │
                       └──→ retrieve ──→ evaluate ──┬──→ generate ──→ save ──→ END
                                                      │
                                                      └──→ rewrite ──→ retrieve (循环)

节点说明：
  classify : 意图分类 + 情感分析 + NER（调用 PyTorch 模型）
  retrieve : LlamaIndex QueryEngine 检索（或 fallback 到 HybridRetriever）
  evaluate : 检索质量评估（复用 ai_app3 逻辑）
  generate : LLM 生成（注入客服话术风格）
  escalate: 转人工判断
  handoff  : 坐席交接
  save     : 保存回复到会话历史
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from ai_app4.graph.state import CS4State
from ai_app4.graph.nodes import (
    classify_node,
    retrieve_node,
    evaluate_node,
    rewrite_node,
    generate_node,
    escalate_node,
    handoff_node,
    save_reply_node,
)
from ai_app4.graph.conditional_edges import after_classify, after_evaluate

# ── 构建状态图 ──────────────────────────────────────────────────────────────

builder = StateGraph(CS4State)

# 注册节点
builder.add_node("classify", classify_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("rewrite", rewrite_node)
builder.add_node("generate", generate_node)
builder.add_node("escalate", escalate_node)
builder.add_node("handoff", handoff_node)
builder.add_node("save_reply", save_reply_node)

# 注册边
builder.set_entry_point("classify")
builder.add_conditional_edges("classify", after_classify)
builder.add_edge("escalate", "handoff")
builder.add_edge("handoff", END)
builder.add_edge("retrieve", "evaluate")
builder.add_conditional_edges("evaluate", after_evaluate)
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", "save_reply")
builder.add_edge("save_reply", END)

# 编译（带 MemorySaver 实现对话状态持久化）
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
