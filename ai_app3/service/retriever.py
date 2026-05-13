"""
Retriever adapter — 将 rag_framework 的 HybridRetriever 封装为简单的 query_context() 接口，
供 ai_app3 的 Agentic RAG 节点和工具调用。

retrieve() 现为 async，调用方需 await。
"""
from __future__ import annotations

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings

from ai_app3.core.logger import retrieve_logger

_container: RAGContainer | None = None


def _get_container() -> RAGContainer:
    global _container
    if _container is None:
        _container = RAGContainer.from_settings(get_settings())
    return _container


async def query_context(query: str, top_k: int = 5, history: list[dict] | None = None) -> str | None:
    """
    执行混合检索，返回格式化的上下文文本（async）。

    内部通过 rag_framework.HybridRetriever 执行
    Rewrite→Classify→Dense+HyDE+BM25+RRF+Rerank+Lost-in-Middle。
    """
    if not query or not query.strip():
        return None
    try:
        container = _get_container()
        routes = container.build_routes(query, history or [])
        result = await container.retriever.retrieve(routes, top_k=top_k)
        if not result.docs:
            retrieve_logger.info(f"检索无结果: query={query[:30]!r}")
            return None
        parts = [doc.text for doc in result.docs if doc.text]
        if not parts:
            return None
        context = "\n\n".join(parts)
        retrieve_logger.info(
            f"检索完成: query={query[:30]!r}, docs={len(result.docs)}, "
            f"latency={result.latency_ms:.0f}ms"
        )
        return context
    except Exception as e:
        retrieve_logger.error(f"检索异常: {e}")
        return None
