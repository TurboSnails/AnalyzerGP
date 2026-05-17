"""
ai_app4 Wealth AI Agent LangGraph StateGraph 构建器。

流程（Wealth AI v4.0 多步推演）：

  START ──→ analyze_and_route ──→ parallel_retrieval ──→ evaluate_and_rerank
                                                                │
                                        [Conditional Edge: after_evaluate]
                                                                │
                    ┌────────────────────┴────────────────────┐
                    │ (top_ce < threshold & loop < max)       │ (top_ce >= threshold 或 loop >= max)
                    ▼                                         ▼
          query_reflection ─────→ parallel_retrieval    strategy_reasoning
                    ↑                                         │
                    │            [Conditional Edge: after_strategy]
                    │                                         │
                    │              ┌──────────┴──────────┐
                    │         (needs_tool)          (纯文本)
                    │              ▼                      ▼
                    │    execute_math_tool        generate_final
                    │              │                      │
                    │              ▼                      │
                    │    merge_and_generate ──────────────┘
                    │              │
                    │              ▼
                    │            END
                    └───────────────────────────────────────┘

节点说明：
  analyze_and_route    : 本地 Qwen 提取特征、中英文 Query 拆解、生成子查询
  parallel_retrieval   : 复用 HybridRetriever 并发多域召回
  evaluate_and_rerank  : CrossEncoder 精排，提取真实 top_ce
  query_reflection     : 反思未命中原因，金融术语化改写 Query
  strategy_reasoning   : 主模型（MiniMax）多步推演，识别是否需要计算工具
  execute_math_tool    : 从 tool_call 提取参数，调用 Python 硬计算
  merge_and_generate   : 合并计算结果与 LLM 话术，生成严谨金融报告
  generate_final       : 纯文本最终回复（无需工具时）
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from ai_app4.graph.state import WealthState
from ai_app4.graph.nodes import (
    analyze_and_route_node,
    parallel_retrieval_node,
    evaluate_and_rerank_node,
    query_reflection_node,
    strategy_reasoning_node,
    execute_math_tool_node,
    merge_and_generate_node,
    generate_final_node,
)
from ai_app4.graph.conditional_edges import after_evaluate, after_strategy

# ── 构建 StateGraph ───────────────────────────────────────────────────────

builder = StateGraph(WealthState)

# 注册节点
builder.add_node("analyze_and_route", analyze_and_route_node)
builder.add_node("parallel_retrieval", parallel_retrieval_node)
builder.add_node("evaluate_and_rerank", evaluate_and_rerank_node)
builder.add_node("query_reflection", query_reflection_node)
builder.add_node("strategy_reasoning", strategy_reasoning_node)
builder.add_node("execute_math_tool", execute_math_tool_node)
builder.add_node("merge_and_generate", merge_and_generate_node)
builder.add_node("generate_final", generate_final_node)

# 注册边
builder.set_entry_point("analyze_and_route")
builder.add_edge("analyze_and_route", "parallel_retrieval")
builder.add_edge("parallel_retrieval", "evaluate_and_rerank")

# 评估后条件边：低置信度且未达循环上限 → 反思；否则 → 策略推演
builder.add_conditional_edges(
    "evaluate_and_rerank",
    after_evaluate,
    {"reflection": "query_reflection", "strategy": "strategy_reasoning"},
)

# 反思后返回重新检索
builder.add_edge("query_reflection", "parallel_retrieval")

# 策略推演后条件边：需要工具 → 执行工具；否则 → 直接生成
builder.add_conditional_edges(
    "strategy_reasoning",
    after_strategy,
    {"tool": "execute_math_tool", "final": "generate_final"},
)

# 工具执行后合并生成
builder.add_edge("execute_math_tool", "merge_and_generate")

# 终止边
builder.add_edge("merge_and_generate", END)
builder.add_edge("generate_final", END)

# 编译（带 MemorySaver 实现对话状态持久化）
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
