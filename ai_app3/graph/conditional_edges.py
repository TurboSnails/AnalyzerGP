"""
条件边函数 — Agentic RAG 流程中的决策逻辑。
"""
from __future__ import annotations

from ai_app3.core.config import MAX_REWRITE_ITERATIONS, RETRIEVAL_CONFIDENCE_THRESHOLD
from ai_app3.core.logger import graph_logger


def after_intent(state: dict) -> str:
    """
    意图分析后决策：
    - 闲聊 / 无需检索 → direct_response
    - 技术问答 / 多步推理 → decompose
    """
    intent = state.get("intent") or {}
    needs = intent.get("needs_retrieval", True)
    intent_type = intent.get("intent", "technical")

    if not needs or intent_type == "casual":
        graph_logger.info(f"意图判定为闲聊/无需检索，跳过检索: intent={intent_type}")
        return "direct_response"
    return "decompose"


def after_evaluate(state: dict) -> str:
    """
    评估后决策：
    - 上下文充分 → generate
    - 不足但可改写 → rewrite_or_expand
    - 已达最大轮数 → generate（降级）
    """
    eval_result = state.get("evaluation_result") or {}
    confidence = float(eval_result.get("confidence", 0.0))
    iteration = state.get("retrieval_iterations", 0)

    if confidence >= RETRIEVAL_CONFIDENCE_THRESHOLD:
        graph_logger.info(f"检索质量达标 (conf={confidence:.2f})，进入生成")
        return "generate"

    if iteration >= MAX_REWRITE_ITERATIONS:
        graph_logger.warning(f"检索轮数已达上限 ({iteration})，降级生成")
        return "generate"

    # 若缺失实体关系类信息，优先知识图谱扩展
    gaps = eval_result.get("gaps", [])
    if any(k in g for g in gaps for k in ("关系", "关联", "依赖", "调用", "继承")):
        graph_logger.info("缺失实体关系信息，优先 KG 扩展")
        return "expand_kg"

    graph_logger.info(f"检索质量不足 (conf={confidence:.2f}, iter={iteration})，尝试改写")
    return "rewrite"


def after_self_check(state: dict) -> str:
    """
    自检后决策：
    - 当前版本默认通过，保留扩展点
    """
    return "pass"
