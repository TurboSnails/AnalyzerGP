"""
检索包装模块：复用 rag_framework 的 HybridRetriever（async）。

将 RetrievalResult 格式化为字符串上下文，保持与旧版 query_db 接口概念一致。
retrieve() 现为 async，调用方（retrieve_node）需 await。
"""
from __future__ import annotations

from ai_app2.core.container import get_app_container


async def query_db(query: str, history: list[dict] | None = None) -> str | None:
    """
    执行混合检索并返回格式化上下文字符串（async）。

    底层调用 rag_framework 的 build_routes + HybridRetriever.retrieve，包含完整管道：
    Rewrite → Classify → Dense → HyDE → BM25 → RRF → Rerank → Lost-in-Middle。
    """
    container = get_app_container()
    history = history or []

    routes = container.build_routes(query, history)
    result = await container.retriever.retrieve(routes)

    if not result.docs:
        return None

    top_ce = result.metadata.get("top_ce", 0.0)
    if top_ce < container.settings.low_confidence_threshold:
        return None

    contexts = [f"【来源: {d.id}】\n{d.text}" for d in result.docs]
    return "\n\n".join(contexts)
