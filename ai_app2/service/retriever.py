"""
检索包装模块：复用 rag_framework 的 HybridRetriever。

将 RetrievalResult 格式化为字符串上下文，保持与旧版 query_db 接口概念一致。
"""
from __future__ import annotations

from ai_app2.core.container import get_app_container


def query_db(query: str, history: list[dict] | None = None) -> str | None:
    """
    执行混合检索并返回格式化上下文字符串。

    底层调用 rag_framework 的 build_routes + HybridRetriever.retrieve，包含完整管道：
    Rewrite → Classify → Dense → HyDE → BM25 → RRF → Rerank → Lost-in-Middle。

    Args:
        query: 用户原始查询。
        history: 会话历史（用于 rewrite/classify）。

    Returns:
        格式化后的参考资料字符串；无结果或低置信度时返回 None。
    """
    container = get_app_container()
    history = history or []

    # 1. 分级 rewrite + classify → 多路 QueryRoute
    routes = container.build_routes(query, history)

    # 2. 多路检索
    result = container.retriever.retrieve(routes)

    if not result.docs:
        return None

    top_ce = result.metadata.get("top_ce", 0.0)
    threshold = container.settings.low_confidence_threshold

    if top_ce < threshold:
        return None

    contexts = [f"【来源: {d.id}】\n{d.text}" for d in result.docs]
    return "\n\n".join(contexts)
