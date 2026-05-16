"""
Agentic Evaluator — 检索结果质量评估与决策。

能力：
1. evaluate_retrieval : 评估检索上下文对回答查询的充分性
2. decide_next_step   : 根据评估结果决定下一步（generate / rewrite / expand_kg）
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ai_app3.core.config import RETRIEVAL_CONFIDENCE_THRESHOLD
from ai_app3.core.llm_provider import get_chat_llm
from ai_app3.core.logger import eval_logger

_llm = get_chat_llm(temperature=0.1)


def evaluate_retrieval(query: str, context: str | None) -> dict:
    """
    评估检索上下文是否足以回答查询。
    返回 { "sufficient": bool, "confidence": float, "gaps": [str], "reason": str }
    """
    if not context or not context.strip():
        eval_logger.info("评估结果: 上下文为空，sufficient=False")
        return {"sufficient": False, "confidence": 0.0, "gaps": ["未检索到任何文档"], "reason": "上下文为空"}

    prompt = [
        SystemMessage(content="你是一个检索质量评估专家。只输出合法 JSON，不要附加解释。"),
        HumanMessage(
            content=f"请评估以下检索上下文是否足以回答用户查询。\n\n"
                    f'输出格式：{{"sufficient": true/false, "confidence": 0.0~1.0, '
                    f'"gaps": ["缺少的信息1", "..."], "reason": "简短评估理由"}}\n\n'
                    f"用户查询：{query}\n\n"
                    f"检索上下文（前2000字）：\n{context[:2000]}\n"
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        raw = resp.content or "{}"
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
    except Exception as e:
        eval_logger.warning(f"evaluate_retrieval 解析失败: {e}，默认 sufficient=False")
        result = {"sufficient": False, "confidence": 0.0, "gaps": ["评估解析失败"], "reason": str(e)}

    confidence = float(result.get("confidence", 0))
    sufficient = bool(result.get("sufficient", False)) and confidence >= RETRIEVAL_CONFIDENCE_THRESHOLD
    result["sufficient"] = sufficient

    eval_logger.info(
        f"评估结果: sufficient={sufficient}, confidence={confidence:.2f}, "
        f"gaps={len(result.get('gaps', []))}"
    )
    return result


def decide_next_step(
    query: str,
    eval_result: dict,
    iteration: int,
    max_iterations: int = 2,
) -> str:
    """
    根据评估结果与当前迭代次数，决定下一步动作。
    返回 "generate" | "rewrite" | "expand_kg"
    """
    if eval_result.get("sufficient"):
        eval_logger.info("决策: generate（上下文充分）")
        return "generate"

    if iteration >= max_iterations:
        eval_logger.warning(f"决策: generate（已达最大改写轮数 {max_iterations}）")
        return "generate"

    gaps = eval_result.get("gaps", [])
    # 若缺失信息涉及实体关系，尝试知识图谱扩展
    if any(k in g for g in gaps for k in ("关系", "关联", "依赖", "调用", "继承")):
        eval_logger.info("决策: expand_kg（缺失实体关系信息）")
        return "expand_kg"

    eval_logger.info("决策: rewrite（上下文不足，尝试查询改写）")
    return "rewrite"
