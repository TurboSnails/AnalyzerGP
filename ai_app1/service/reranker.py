"""
精排模块（Phase 3）

rerank_chunks:
    基于多维度线性组合对候选 parent 片段重排：
    RRF 分（0.45）+ 词项覆盖率（0.30）+ 向量排名倒数（0.15）+ BM25 排名倒数（0.10）

reorder_lost_in_middle:
    按"Lost in the Middle"理论重排上下文顺序：
    最相关 → 首位；次相关 → 末位；其余居中。
    LLM 对首尾注意力最强，核心内容不会被埋在中间。
"""
import re
import logging

logger = logging.getLogger("reranker")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[一-鿿]{1,2}|[a-zA-Z0-9]+", text.lower()))


def _term_overlap(query_tokens: set[str], doc: str) -> float:
    """query 词中有多少比例出现在 doc 里"""
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc)
    return len(query_tokens & doc_tokens) / len(query_tokens)


def rerank_chunks(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    对候选 parent chunks 进行精排。

    每个 candidate 须含字段:
        id         : str
        text       : str
        rrf_score  : float
        vector_rank: int  (越小越好, 999 = 未命中)
        bm25_rank  : int  (越小越好, 999 = 未命中)

    Returns:
        top_k 个候选，按 final_score 降序，每个 candidate 追加 final_score 字段
    """
    q_tokens = _tokenize(query)

    for c in candidates:
        term_score = _term_overlap(q_tokens, c["text"])
        vector_inv = 1.0 / (1 + c.get("vector_rank", 999))
        bm25_inv = 1.0 / (1 + c.get("bm25_rank", 999))
        c["final_score"] = (
            0.45 * c.get("rrf_score", 0.0)
            + 0.30 * term_score
            + 0.15 * vector_inv
            + 0.10 * bm25_inv
        )

    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]
    top_score = ranked[0]["final_score"] if ranked else 0.0
    logger.debug(
        f"Rerank: {len(candidates)} → {len(ranked)} 个, "
        f"top_score={top_score:.3f}, query={query[:20]!r}"
    )
    return ranked


def reorder_lost_in_middle(chunks: list[dict]) -> list[dict]:
    """
    Lost-in-the-Middle 上下文重排。

    输入按相关度降序（index 0 最相关）：
        [rank1, rank2, rank3, rank4, rank5]
    输出：
        [rank1, rank3, rank4, rank5, rank2]
        ↑ 首位                       ↑ 末位
    最相关放首位，次相关放末位，确保 LLM 不遗漏核心内容。
    """
    if len(chunks) <= 2:
        return chunks

    most_relevant = chunks[0]
    second_relevant = chunks[1]
    middle = chunks[2:]

    result = [most_relevant] + middle + [second_relevant]
    logger.debug(
        f"Lost-in-Middle 重排: {[c['id'] for c in chunks]} "
        f"→ {[c['id'] for c in result]}"
    )
    return result
