"""
工具定义：为 Agentic RAG 提供可调用工具。

当前工具集：
- multiply          : 示例数学工具（保留以验证 tool calling 链路）
- search_docs       : 文档检索工具（供 LLM 主动调用，实现 self-RAG）
- evaluate_answer   : 回答自检工具（让 LLM 评估自身回答质量）
"""
from __future__ import annotations

from langchain_core.tools import tool

from ai_app3.service.retriever import query_context as query_db
from ai_app3.service.knowledge_graph import expand_by_entities, fetch_docs_by_ids
from ai_app3.core.logger import graph_logger


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积"""
    return a * b


@tool
async def search_docs(query: str, top_k: int = 5) -> str:
    """
    在 Android 开发文档知识库中检索与查询相关的技术资料。
    返回检索到的文档片段，可直接作为回答依据。
    """
    graph_logger.info(f"[tool] search_docs: query={query!r}, top_k={top_k}")
    context = await query_db(query)
    if not context:
        # 尝试知识图谱补充
        kg_ids = expand_by_entities(query, top_k=top_k)
        if kg_ids:
            context = fetch_docs_by_ids(kg_ids)
    return context or "未找到与查询相关的文档资料。"


@tool
def evaluate_answer(query: str, answer: str, context: str) -> str:
    """
    评估回答是否基于给定上下文，是否准确回答了查询。
    返回 JSON 字符串: {"faithful": bool, "relevant": bool, "score": 0~1, "suggestion": str}
    """
    # 简单启发式评估（避免循环调用 LLM，降低延迟）
    faithful = bool(context and context.strip() in answer or len(answer) < 500)
    relevant = bool(query and any(w in answer for w in query.split() if len(w) > 2))
    score = 0.7 if (faithful and relevant) else 0.4
    suggestion = "回答基于检索结果，可信度高。" if score >= 0.7 else "建议核对检索结果准确性。"
    import json
    return json.dumps({"faithful": faithful, "relevant": relevant, "score": score, "suggestion": suggestion}, ensure_ascii=False)


TOOLS = [multiply, search_docs, evaluate_answer]
TOOL_MAP = {t.name: t for t in TOOLS}
