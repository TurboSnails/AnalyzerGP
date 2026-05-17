"""
ai_app4 Wealth AI Agent 条件边路由逻辑。
"""
from __future__ import annotations

from ai_app4.graph.state import WealthState


def after_evaluate(state: WealthState) -> str:
    """
    检索评估后的条件路由。

    判断逻辑：
      - top_ce < reflection_threshold 且 loop < max_loop → "reflection"（进入自旋锁反思）
      - 否则 → "strategy"（进入策略推演）

    Returns:
        "reflection" | "strategy"
    """
    top_ce = state.get("top_ce", 0.0)
    confidence = state.get("confidence", 0.0)
    iterations = state.get("retrieval_iterations", 0)

    # 从上下文中获取配置（fallback 到默认值）
    # 节点已将 threshold 和 max_loop 写入 evaluation_result
    eval_result = state.get("evaluation_result") or {}
    threshold = eval_result.get("reflection_threshold", 0.35)
    max_loop = eval_result.get("max_loop_count", 2)

    # 优先使用 top_ce，若未设置则 fallback 到 confidence
    effective_score = top_ce if top_ce > 0 else confidence

    if effective_score < threshold and iterations < max_loop:
        return "reflection"
    return "strategy"


def after_strategy(state: WealthState) -> str:
    """
    策略推演后的条件路由。

    判断逻辑：
      - needs_tool == True → "tool"（执行数学计算工具）
      - 否则 → "final"（直接生成纯文本回复）

    Returns:
        "tool" | "final"
    """
    needs_tool = state.get("needs_tool", False)
    if needs_tool:
        return "tool"
    return "final"
