import logging
import time

logger = logging.getLogger("retrieval")


def retrieve(query: str, history: list) -> dict:
    """
    检索主入口：rewrite → 多路召回 → RRF → rerank。
    返回 {context, top_ce, n_chunks}。
    """
    from ai_app1.retrieval.query_rewriter import rewrite_queries, RewriteQuery
    from ai_app1.retrieval.vector_store import query_db

    t0 = time.perf_counter()
    try:
        queries = rewrite_queries(query, history)
    except Exception as e:
        logger.warning("rewrite 失败，降级用原 query: %s", e)
        queries = [RewriteQuery(text=query, type="original", weight=1.0,
                                routes=["dense", "hyde", "bm25"])]
    t_rewrite_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    meta = query_db(queries, return_meta=True) or {"context": None, "top_ce": 0.0, "n_chunks": 0}
    t_retrieve_ms = (time.perf_counter() - t1) * 1000

    logger.info(
        "[rewrite→召回] rewrite=%.0fms (%d 条), 召回=%.0fms, 总=%.0fms, top_ce=%.3f, n_chunks=%d",
        t_rewrite_ms, len(queries), t_retrieve_ms, t_rewrite_ms + t_retrieve_ms,
        meta.get("top_ce", 0.0), meta.get("n_chunks", 0),
    )
    return meta
