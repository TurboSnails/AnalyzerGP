"""
混合检索管道（Phase 2 + Phase 3）

检索流程：
  路A  Dense  : 向量检索 android_child  → 回溯 android_parent
  路B  HyDE   : 向量检索 android_hyde   → 回溯 android_parent
  路C  BM25   : 稀疏全文检索 android_parent
  融合  RRF   : Reciprocal Rank Fusion 合并三路结果
  精排  Rerank: 多维度线性评分，取 Top RERANK_TOP_K
  重排  L-i-M : Lost-in-Middle 上下文重排

降级策略：
  若 v2 collection 不存在（Phase 1 未运行），自动回退旧版 android_docs 单路检索。
"""
from __future__ import annotations

import logging
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH

logger = logging.getLogger("vector_store")

# ─── 超参数 ───────────────────────────────────────────────────────────────────
MAX_DISTANCE = 1.2   # 旧版单路检索阈值
RRF_K = 60           # RRF 常数（越大越平滑排名差异）
DENSE_TOP_K = 10     # 向量检索 child 返回数
HYDE_TOP_K = 5       # HyDE 问题匹配数
BM25_TOP_K = 10      # BM25 返回数
RERANK_TOP_K = 5     # 最终喂给 LLM 的片段数

_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_collection(name: str):
    try:
        return _get_client().get_collection(name)
    except Exception:
        return None


# ─── RRF 融合 ─────────────────────────────────────────────────────────────────

def _rrf_merge(ranked_lists: list[list[str]]) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion：不依赖原始分值，仅按排名融合多路结果。
    score(d) = Σ 1/(rank + RRF_K)
    """
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + RRF_K)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─── 各检索路 ──────────────────────────────────────────────────────────────────

def _query_dense(query: str, col_child) -> list[str]:
    """路A：向量检索 child → 去重 parent_id，按 distance 升序"""
    result = col_child.query(query_texts=[query], n_results=DENSE_TOP_K)
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    seen: dict[str, float] = {}
    for meta, dist in zip(metas, distances):
        pid = meta["parent_id"]
        if pid not in seen:
            seen[pid] = dist

    ranked = sorted(seen.items(), key=lambda x: x[1])
    pids = [p for p, _ in ranked]
    top_dist = f"{ranked[0][1]:.3f}" if ranked else "N/A"
    logger.debug(f"Dense 检索: {len(pids)} 个 parent_id, top_dist={top_dist}")
    return pids


def _query_hyde(query: str, col_hyde) -> list[str]:
    """路B：向量检索 HyDE 假设问题 → 去重 parent_id，按 distance 升序"""
    result = col_hyde.query(query_texts=[query], n_results=HYDE_TOP_K)
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    seen: dict[str, float] = {}
    for meta, dist in zip(metas, distances):
        pid = meta["parent_id"]
        if pid not in seen:
            seen[pid] = dist

    ranked = sorted(seen.items(), key=lambda x: x[1])
    pids = [p for p, _ in ranked]
    logger.debug(f"HyDE 检索: {len(pids)} 个 parent_id")
    return pids


def _fetch_parents(parent_ids: list[str], col_parent) -> dict[str, str]:
    """批量拉取 parent 文档，返回 {id: text}"""
    if not parent_ids:
        return {}
    result = col_parent.get(ids=parent_ids)
    return dict(zip(result["ids"], result["documents"]))


# ─── 主检索入口 ────────────────────────────────────────────────────────────────

def query_db(query: str) -> str | None:
    """
    混合检索主入口，与旧版保持相同的对外接口（返回拼接文本或 None）。

    内部执行：多路召回 → RRF 融合 → Rerank → Lost-in-Middle 重排
    """
    from ai_app1.service import bm25_store
    from ai_app1.service.reranker import rerank_chunks, reorder_lost_in_middle

    col_parent = _get_collection("android_parent")
    col_child = _get_collection("android_child")
    col_hyde = _get_collection("android_hyde")
    v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

    if not v2_ready:
        logger.warning("v2 collections 未就绪，回退旧版单路检索（请运行 init_vector_db_v2.py）")
        return _legacy_query(query)

    # ── 多路召回 ──────────────────────────────────────────────────────────────
    dense_pids = _query_dense(query, col_child)
    hyde_pids = _query_hyde(query, col_hyde)
    bm25_results = bm25_store.search(query, top_k=BM25_TOP_K)
    bm25_pids = [r[0] for r in bm25_results]

    logger.info(
        f"多路召回: dense={len(dense_pids)}, hyde={len(hyde_pids)}, bm25={len(bm25_pids)}"
    )

    # ── RRF 融合 ──────────────────────────────────────────────────────────────
    rrf_results = _rrf_merge([dense_pids, hyde_pids, bm25_pids])
    rrf_score_map = {pid: score for pid, score in rrf_results}
    top20_ids = [pid for pid, _ in rrf_results[:20]]
    logger.info(
        f"RRF 融合: {len(rrf_results)} 候选, "
        f"top={rrf_results[0][0] if rrf_results else 'N/A'}"
    )

    # ── 拉取 parent 文本 ──────────────────────────────────────────────────────
    parent_texts = _fetch_parents(top20_ids, col_parent)

    # ── 构建候选结构 ──────────────────────────────────────────────────────────
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for v_rank, pid in enumerate(dense_pids):
        if pid in parent_texts and pid not in seen_ids:
            seen_ids.add(pid)
            candidates.append({
                "id": pid,
                "text": parent_texts[pid],
                "rrf_score": rrf_score_map.get(pid, 0.0),
                "vector_rank": v_rank,
                "bm25_rank": bm25_pids.index(pid) if pid in bm25_pids else 999,
            })

    for pid in top20_ids:
        if pid in parent_texts and pid not in seen_ids:
            seen_ids.add(pid)
            candidates.append({
                "id": pid,
                "text": parent_texts[pid],
                "rrf_score": rrf_score_map.get(pid, 0.0),
                "vector_rank": dense_pids.index(pid) if pid in dense_pids else 999,
                "bm25_rank": bm25_pids.index(pid) if pid in bm25_pids else 999,
            })

    if not candidates:
        logger.warning(f"所有路径均无结果: query={query[:30]!r}")
        return None

    # ── Rerank ────────────────────────────────────────────────────────────────
    reranked = rerank_chunks(query, candidates, top_k=RERANK_TOP_K)
    for r in reranked:
        logger.info(
            f"  [{r['id']}] final={r['final_score']:.3f} "
            f"rrf={r['rrf_score']:.4f} v_rank={r['vector_rank']} "
            f"b_rank={r['bm25_rank']} | {r['text'][:50]!r}"
        )

    # ── Lost-in-Middle 重排 ───────────────────────────────────────────────────
    ordered = reorder_lost_in_middle(reranked)

    result_text = "\n\n".join(c["text"] for c in ordered)
    logger.info(
        f"query_db 完成: {len(ordered)} 个片段, total_len={len(result_text)}"
    )
    return result_text


# ─── 旧版单路降级 ─────────────────────────────────────────────────────────────

def _legacy_query(query: str) -> str | None:
    """旧版 android_docs collection 单路向量检索（降级用）"""
    col = _get_client().get_or_create_collection("android_docs")
    results = col.query(query_texts=[query], n_results=5)
    docs = results["documents"][0]
    distances = results["distances"][0]

    valid = [doc for doc, dist in zip(docs, distances) if dist <= MAX_DISTANCE]
    logger.debug(f"旧版检索: {len(valid)}/{len(docs)} 有效结果")
    return "\n".join(valid) if valid else None
