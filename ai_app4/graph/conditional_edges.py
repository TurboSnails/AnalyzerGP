"""
ai_app4 条件边路由逻辑。
"""
from __future__ import annotations

from ai_app4.graph.state import CS4State


def after_classify(state: CS4State) -> str:
    """
    意图分类后的条件路由。

    Returns:
        "escalate" | "retrieve" | "generate"
    """
    intent = state.get("intent", "")
    sentiment = state.get("sentiment", "neutral")
    escalation_triggered = state.get("escalation_triggered", False)

    if escalation_triggered or intent in ("escalation_request", "complaint") or sentiment == "negative":
        return "escalate"
    if intent == "chitchat":
        return "generate"
    return "retrieve"


def after_evaluate(state: CS4State) -> str:
    """
    检索评估后的条件路由。

    Returns:
        "generate" | "rewrite"
    """
    confidence = state.get("confidence", 0.0)
    iterations = state.get("retrieval_iterations", 0)
    max_iterations = 3

    if confidence >= 0.6 or iterations >= max_iterations:
        return "generate"
    return "rewrite"
