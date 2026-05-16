"""
Agentic Query Engine — 查询理解、分解与改写。

能力：
1. intent_analysis    : 识别用户意图（技术问答 / 闲聊 / 澄清 / 多步推理）
2. decompose_query    : 将复杂问题拆分为子查询列表
3. rewrite_query      : 基于反馈历史改写查询，提升召回
4. merge_contexts     : 多路子查询上下文合并与去重
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from ai_app3.core.config import (
    MAX_SUB_QUERIES,
    SUB_QUERY_MIN_CONFIDENCE,
)
from ai_app3.core.llm_provider import get_rewriter_llm
from ai_app3.core.logger import retrieve_logger


_llm = get_rewriter_llm(temperature=0.2)


def intent_analysis(query: str) -> dict:
    """
    分析查询意图，返回结构化结果：
      { "intent": "technical|casual|clarify|multi_step", "needs_retrieval": bool, "reason": str }
    """
    prompt = [
        SystemMessage(content="你是一个意图分类专家，只输出合法 JSON，不要附加解释。"),
        HumanMessage(
            content=f"请分析以下用户查询的意图，输出 JSON 格式：\n\n"
                    f'{{"intent": "technical|casual|clarify|multi_step", '
                    f'"needs_retrieval": true/false, "reason": "简短理由"}}\n\n'
                    f"查询：{query}"
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        raw = resp.content or "{}"
        # 去除可能的 markdown 代码块
        raw = re.sub(r"```json|```", "", raw).strip()
        result = json.loads(raw)
    except Exception as e:
        retrieve_logger.warning(f"intent_analysis 解析失败: {e}，默认 technical")
        result = {"intent": "technical", "needs_retrieval": True, "reason": "默认技术问答"}

    retrieve_logger.info(f"意图分析: intent={result.get('intent')}, needs_retrieval={result.get('needs_retrieval')}")
    return result


def decompose_query(query: str) -> list[dict]:
    """
    将复杂/多步问题拆分为子查询列表。
    返回 [{"sub_query": str, "confidence": float, "reason": str}, ...]
    """
    prompt = [
        SystemMessage(content="你是一个查询分解专家。将复杂问题拆分为最多3个独立的子查询，只输出合法 JSON 数组。"),
        HumanMessage(
            content=f"请将以下问题拆分为子查询，输出 JSON 数组：\n\n"
                    f'[{{"sub_query": "...", "confidence": 0.9, "reason": "..."}}]\n\n'
                    f"问题：{query}\n"
                    f"要求：\n1. 每个子查询独立可回答\n2. confidence 在 0~1 之间\n3. 若问题简单无需拆分，返回包含原问题的单元素数组"
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        raw = resp.content or "[]"
        raw = re.sub(r"```json|```", "", raw).strip()
        subs = json.loads(raw)
        if not isinstance(subs, list):
            subs = [subs] if isinstance(subs, dict) else []
    except Exception as e:
        retrieve_logger.warning(f"decompose_query 解析失败: {e}，回退单查询")
        subs = [{"sub_query": query, "confidence": 1.0, "reason": "回退原查询"}]

    # 过滤低置信度并限制数量
    subs = [s for s in subs if s.get("confidence", 0) >= SUB_QUERY_MIN_CONFIDENCE]
    subs = subs[:MAX_SUB_QUERIES]

    retrieve_logger.info(f"查询分解: {len(subs)} 个子查询")
    for s in subs:
        retrieve_logger.debug(f"  - {s.get('sub_query')!r} (conf={s.get('confidence')})")
    return subs


def rewrite_query(original_query: str, feedback: str | None = None) -> str:
    """
    基于历史反馈（如前一轮检索结果为空或质量低）改写查询。
    feedback 示例: "前次检索未找到有效结果，请换用更通用的技术术语"
    """
    if not feedback:
        return original_query

    prompt = [
        SystemMessage(content="你是一个查询改写专家，将用户原始查询改写为更利于向量检索的形式，只输出改写后的查询文本，不要解释。"),
        HumanMessage(
            content=f"原始查询：{original_query}\n"
                    f"反馈：{feedback}\n"
                    f"请输出改写后的查询（保持语言一致）："
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        rewritten = (resp.content or original_query).strip()
        if rewritten and rewritten != original_query:
            retrieve_logger.info(f"查询改写: {original_query!r} → {rewritten!r}")
            return rewritten
    except Exception as e:
        retrieve_logger.warning(f"rewrite_query 异常: {e}")
    return original_query


def merge_contexts(contexts: list[str | None]) -> str | None:
    """
    合并多路子查询的检索上下文，去重并排序。
    简单实现：按文本去重后拼接（高级实现可引入语义去重）。
    """
    valid = [c for c in contexts if c]
    if not valid:
        return None

    seen: set[str] = set()
    unique: list[str] = []
    for ctx in valid:
        # 以段落为单位去重（兼顾效率与效果）
        for para in ctx.split("\n\n"):
            para = para.strip()
            if not para or para in seen:
                continue
            seen.add(para)
            unique.append(para)

    retrieve_logger.info(f"上下文合并: {len(valid)} 路子查询 → {len(unique)} 个唯一段落")
    return "\n\n".join(unique)
