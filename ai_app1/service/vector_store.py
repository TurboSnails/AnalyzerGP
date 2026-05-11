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
import time
from concurrent.futures import ThreadPoolExecutor, Future
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH
from ai_app1.service.embedding import get_embedding_service

logger = logging.getLogger("vector_store")

# ─── 超参数 ───────────────────────────────────────────────────────────────────
MAX_DISTANCE = 1.2       # 旧版单路检索阈值
MAX_CHILD_DISTANCE = 1.3 # child 层面向量距离阈值
RRF_K = 60               # RRF 常数（越大越平滑排名差异）
DENSE_QUERY_K = 25       # child 查询量（去重/聚合后保留 DENSE_TOP_K）
DENSE_TOP_K = 10         # 向量检索最终 parent 返回数
HYDE_QUERY_K = 15        # HyDE 查询量
HYDE_TOP_K = 5           # HyDE 最终 parent 返回数
BM25_TOP_K = 10          # BM25 返回数
RERANK_TOP_K = 5         # 最终喂给 LLM 的片段数

_client: chromadb.PersistentClient | None = None


_embed_svc = None


def _get_embed():
    global _embed_svc
    if _embed_svc is None:
        _embed_svc = get_embedding_service()
    return _embed_svc


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

def _aggregate_parent_hits(metas, distances, max_dist: float, top_k: int):
    """按 parent_id 聚合 child/hyde 命中结果，返回排序后的 parent_id 列表。

    策略：
      1. 过滤 distance > max_dist 的噪声
      2. 同一 parent 下收集所有命中 child 的 distance
      3. parent 级得分 = min(distance) - 0.05 * (hit_count - 1)
         （命中次数越多，轻微提升排序）
      4. 按得分升序返回 top_k 个 parent_id
    """
    parent_hits: dict[str, list[float]] = {}
    for meta, dist in zip(metas, distances):
        if dist > max_dist:
            continue
        pid = meta["parent_id"]
        parent_hits.setdefault(pid, []).append(dist)

    scored = []
    for pid, dists in parent_hits.items():
        min_dist = min(dists)
        hit_count = len(dists)
        score = min_dist - 0.05 * (hit_count - 1)
        scored.append((pid, score, min_dist, hit_count))

    scored.sort(key=lambda x: x[1])
    pids = [s[0] for s in scored[:top_k]]
    return pids, scored[:top_k] if scored else []


def _query_dense(query: str, col_child) -> list[str]:
    """路A：向量检索 child → 聚合 parent_id，按多命中加权 distance 升序"""
    q_emb = _get_embed().encode([query])
    result = col_child.query(query_embeddings=q_emb, n_results=DENSE_QUERY_K)
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    pids, top_scored = _aggregate_parent_hits(
        metas, distances, MAX_CHILD_DISTANCE, DENSE_TOP_K
    )

    if top_scored:
        top = top_scored[0]
        logger.debug(
            f"Dense 检索: {len(pids)} 个 parent, "
            f"top_dist={top[2]:.3f} (hits={top[3]})"
        )
    else:
        logger.debug("Dense 检索: 无有效结果")
    return pids


def _query_hyde(query: str, col_hyde) -> list[str]:
    """路B：向量检索 HyDE 假设问题 → 聚合 parent_id，按多命中加权 distance 升序"""
    q_emb = _get_embed().encode([query])
    result = col_hyde.query(query_embeddings=q_emb, n_results=HYDE_QUERY_K)
    metas = result["metadatas"][0]
    distances = result["distances"][0]

    pids, top_scored = _aggregate_parent_hits(
        metas, distances, MAX_CHILD_DISTANCE, HYDE_TOP_K
    )

    if top_scored:
        top = top_scored[0]
        logger.debug(
            f"HyDE 检索: {len(pids)} 个 parent, "
            f"top_dist={top[2]:.3f} (hits={top[3]})"
        )
    else:
        logger.debug("HyDE 检索: 无有效结果")
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

    # ── 多路召回（并发）────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as pool:
        f_dense: Future = pool.submit(_query_dense, query, col_child)
        f_hyde:  Future = pool.submit(_query_hyde,  query, col_hyde)
        f_bm25:  Future = pool.submit(bm25_store.search, query, BM25_TOP_K)
        dense_pids   = f_dense.result()
        hyde_pids    = f_hyde.result()
        bm25_results = f_bm25.result()
    bm25_pids = [r[0] for r in bm25_results]

    logger.info(
        f"多路召回: dense={len(dense_pids)}, hyde={len(hyde_pids)}, bm25={len(bm25_pids)}"
        f" | 耗时={1000*(time.perf_counter()-t0):.0f}ms"
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
    # 强制断言：Rerank 后不允许存在重复的 parent_id（避免上下文污染）
    _rerank_ids = [r["id"] for r in reranked]
    assert len(_rerank_ids) == len(set(_rerank_ids)),         f"Rerank 输出存在重复 parent_id: {_rerank_ids}"
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
    q_emb = _get_embed().encode([query])
    results = col.query(query_embeddings=q_emb, n_results=5)
    docs = results["documents"][0]
    distances = results["distances"][0]

    valid = [doc for doc, dist in zip(docs, distances) if dist <= MAX_DISTANCE]
    logger.debug(f"旧版检索: {len(valid)}/{len(docs)} 有效结果")
    return "\n".join(valid) if valid else None
