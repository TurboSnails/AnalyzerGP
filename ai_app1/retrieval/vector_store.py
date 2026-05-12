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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH
from ai_app1.retrieval.embedding import get_embedding_service
from ai_app1.retrieval.query_rewriter import RewriteQuery

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
RERANK_TOP_K = 3         # 最终喂给 LLM 的片段数（3 个 vs 5 个：本地省 ~150ms 重排，prompt 短 800 字 → MiniMax TTFT 再省 500ms+）

# ─── 低置信度兜底阈值 ─────────────────────────────────────────────────────────
# CrossEncoder ce_score 经 sigmoid 归一化到 0~1；query 与 doc 强相关时通常 > 0.5
# 阈值含义：top1 ce_score < 此值 → 视为「知识库无相关内容」，触发拒答路径
# 经验值：0.30 偏宽松（允许弱相关也喂给 LLM）；0.50 偏严格（避免幻觉但可能漏召）
LOW_CONFIDENCE_CE_THRESHOLD = 0.30

_client: chromadb.PersistentClient | None = None
_embed_svc = None
_embed_lock = threading.Lock()
_client_lock = threading.Lock()


def _get_embed():
    global _embed_svc
    if _embed_svc is None:
        with _embed_lock:
            if _embed_svc is None:
                _embed_svc = get_embedding_service()
    return _embed_svc


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def _get_collection(name: str):
    try:
        return _get_client().get_collection(name)
    except Exception:
        return None


# ─── RRF 融合 ─────────────────────────────────────────────────────────────────

def _rrf_merge(ranked_lists: "list[tuple[list[str], float]]") -> list[tuple[str, float]]:
    """
    Weighted Reciprocal Rank Fusion：按权重融合多路结果。
    score(d) = Σ weight_i / (rank_i + RRF_K)

    weight 由 RewriteQuery.weight 传入：原始 query 权重最高（1.0），
    扩写变体按 type 递减（semantic=0.9, keyword=0.85, api=0.75~0.80），
    使原始问题对最终排名的贡献始终最大。
    """
    scores: dict[str, float] = {}
    for lst, weight in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rank + RRF_K)
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

def query_db(
    queries: "list[RewriteQuery] | list[str] | str",
    return_meta: bool = False,
) -> "str | None | dict":
    """
    多查询混合检索主入口（兼容单条 str 调用）。

    queries[0] 为原始用户问题，用于 CrossEncoder Rerank；
    其余为改写扩展 query，用于扩大三路召回的覆盖范围。

    内部执行：N×3路并发召回 → RRF融合 → Rerank → Lost-in-Middle重排

    Args:
        return_meta: 默认 False 保持向后兼容，返回 context 字符串。
                     True 时返回 {"context", "top_ce", "n_chunks"} 字典，
                     供 session 层做低置信度兜底判断。
    """
    # ── 输入归一化：str / list[str] → list[RewriteQuery] ─────────────────────
    if isinstance(queries, str):
        queries = [RewriteQuery(text=queries, type="original", weight=1.0,
                                routes=["dense", "hyde", "bm25"])]
    elif queries and isinstance(queries[0], str):
        # 向后兼容 list[str]：第一条视为 original，其余视为 semantic
        rq_list: list[RewriteQuery] = []
        for i, q in enumerate(queries):
            q = q.strip()
            if not q:
                continue
            if i == 0:
                rq_list.append(RewriteQuery(text=q, type="original", weight=1.0,
                                            routes=["dense", "hyde", "bm25"]))
            else:
                rq_list.append(RewriteQuery(text=q, type="semantic", weight=0.9,
                                            routes=["dense", "hyde"]))
        queries = rq_list

    queries = [rq for rq in queries if rq.text.strip()]
    if not queries:
        return {"context": None, "top_ce": 0.0, "n_chunks": 0} if return_meta else None

    original_query = queries[0].text

    from ai_app1.retrieval import bm25_store
    from ai_app1.retrieval.reranker import rerank_chunks, reorder_lost_in_middle

    col_parent = _get_collection("android_parent")
    col_child  = _get_collection("android_child")
    col_hyde   = _get_collection("android_hyde")
    v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

    if not v2_ready:
        logger.warning("v2 collections 未就绪，回退旧版单路检索（请运行 init_vector_db_v2.py）")
        return _legacy_query(original_query)

    # ── 按路由元数据选择性提交任务（Retrieval Orchestration） ─────────────────
    # 每条 RewriteQuery 只送往 routes 指定的路径，避免低质量 query 污染所有路径
    t0 = time.perf_counter()
    weighted_lists: list[tuple[list[str], float]] = []   # (pid_list, weight) for Weighted RRF
    pid_best_dense_rank: dict[str, int] = {}
    pid_best_bm25_rank:  dict[str, int] = {}

    n_tasks = sum(len(rq.routes) for rq in queries)
    with ThreadPoolExecutor(max_workers=max(n_tasks, 1)) as pool:
        futures: list[tuple[str, float, Future]] = []
        for rq in queries:
            if "dense" in rq.routes:
                futures.append(("dense", rq.weight, pool.submit(_query_dense, rq.text, col_child)))
            if "hyde" in rq.routes:
                futures.append(("hyde",  rq.weight, pool.submit(_query_hyde,  rq.text, col_hyde)))
            if "bm25" in rq.routes:
                futures.append(("bm25",  rq.weight, pool.submit(bm25_store.search, rq.text, BM25_TOP_K)))

        for kind, weight, f in futures:
            result = f.result()
            if kind == "bm25":
                pids = [r[0] for r in result]
                for rank, pid in enumerate(pids):
                    pid_best_bm25_rank[pid] = min(pid_best_bm25_rank.get(pid, 999), rank)
            else:
                pids = result
                if kind == "dense":
                    for rank, pid in enumerate(pids):
                        pid_best_dense_rank[pid] = min(pid_best_dense_rank.get(pid, 999), rank)
            weighted_lists.append((pids, weight))

    route_summary = " | ".join(f"{rq.type}({'+'.join(rq.routes)})" for rq in queries)
    logger.info(
        f"多路召回: {n_tasks} 路 [{route_summary}]"
        f" | 耗时={1000*(time.perf_counter()-t0):.0f}ms"
    )

    # ── Weighted RRF 融合（按 query 权重加权） ──────────────────────────────────
    rrf_results = _rrf_merge(weighted_lists)
    rrf_score_map = {pid: score for pid, score in rrf_results}
    top20_ids = [pid for pid, _ in rrf_results[:20]]
    logger.info(
        f"RRF 融合: {len(rrf_results)} 候选, "
        f"top={rrf_results[0][0] if rrf_results else 'N/A'}"
    )

    # ── 拉取 parent 文本 ──────────────────────────────────────────────────────
    parent_texts = _fetch_parents(top20_ids, col_parent)

    # ── 构建候选结构（按 RRF 排名顺序，天然去重） ──────────────────────────────
    seen_ids: set[str] = set()
    candidates: list[dict] = []

    for pid in top20_ids:
        if pid in parent_texts and pid not in seen_ids:
            seen_ids.add(pid)
            candidates.append({
                "id":          pid,
                "text":        parent_texts[pid],
                "rrf_score":   rrf_score_map[pid],
                "vector_rank": pid_best_dense_rank.get(pid, 999),
                "bm25_rank":   pid_best_bm25_rank.get(pid, 999),
            })

    if not candidates:
        logger.warning(f"所有路径均无结果: queries={[q[:20] for q in queries]}")
        return {"context": None, "top_ce": 0.0, "n_chunks": 0} if return_meta else None

    reranked = rerank_chunks(original_query, candidates, top_k=RERANK_TOP_K)
    _rerank_ids = [r["id"] for r in reranked]
    assert len(_rerank_ids) == len(set(_rerank_ids)),         f"Rerank 输出存在重复 parent_id: {_rerank_ids}"
    for r in reranked:
        logger.info(
            f"  [{r['id']}] final={r['final_score']:.3f} "
            f"ce={r.get('ce_score', 0.0):.3f} "
            f"rrf={r['rrf_score']:.4f} v_rank={r['vector_rank']} "
            f"b_rank={r['bm25_rank']} | {r['text'][:50]!r}"
        )

    top_ce = float(reranked[0].get("ce_score", 0.0)) if reranked else 0.0

    ordered = reorder_lost_in_middle(reranked)
    result_text = "\n\n".join(c["text"] for c in ordered)
    logger.info(
        f"query_db 完成: {len(ordered)} 个片段, total_len={len(result_text)}, "
        f"top_ce={top_ce:.3f}"
    )

    if return_meta:
        return {"context": result_text, "top_ce": top_ce, "n_chunks": len(ordered)}
    return result_text


# ═══════════════════════════════════════════════════════════════════════════════
#  结构化检索接口 —— 评测与消融实验专用
# ═══════════════════════════════════════════════════════════════════════════════

from dataclasses import dataclass, field


@dataclass
class RetrievalConfig:
    """
    检索管道配置开关，用于消融实验（Ablation Study）。

    所有字段默认 True，表示完整 pipeline；评测时通过置 False 关闭某模块，
    观察该模块对 Recall / MRR / Latency 的真实贡献。
    """
    # 召回路径开关
    enable_dense: bool = True          # 路A：向量检索 child → parent
    enable_hyde: bool = True           # 路B：HyDE 假设问题检索
    enable_bm25: bool = True           # 路C：BM25 稀疏全文检索

    # 融合与精排开关
    enable_rerank: bool = True         # CrossEncoder 语义精排
    enable_lost_in_middle: bool = True # Lost-in-Middle 上下文重排

    # Query 扩写开关（由调用方控制是否传入 rewrite_queries 结果）
    enable_rewrite: bool = True

    # 超参数覆写（实验时无需改源码）
    dense_top_k: int = DENSE_TOP_K
    hyde_top_k: int = HYDE_TOP_K
    bm25_top_k: int = BM25_TOP_K
    rerank_top_k: int = RERANK_TOP_K
    rrf_k: int = RRF_K

    def summary(self) -> str:
        """返回人类可读的配置摘要，用于实验报告。"""
        paths = []
        if self.enable_dense:
            paths.append("dense")
        if self.enable_hyde:
            paths.append("hyde")
        if self.enable_bm25:
            paths.append("bm25")
        stages = []
        if self.enable_rerank:
            stages.append("rerank")
        if self.enable_lost_in_middle:
            stages.append("LiM")
        return f"paths=[{'+'.join(paths)}] stages=[{'+'.join(stages)}] rewrite={'on' if self.enable_rewrite else 'off'}"


@dataclass
class ChunkInfo:
    """单个召回 chunk 的完整元数据，用于排序质量分析。"""
    id: str
    text: str
    # 各阶段分数
    rrf_score: float = 0.0
    ce_score: float = 0.0
    final_score: float = 0.0
    # 各路原始排名（越小越好，999 表示未命中该路）
    vector_rank: int = 999
    bm25_rank: int = 999
    # 最终排序位置（0-based，按 Lost-in-Middle 后的顺序）
    final_position: int = 0


@dataclass
class RetrievalResult:
    """
    结构化检索结果，包含每个 chunk 的详细评分与耗时分解。

    Attributes:
        chunks         : 按最终顺序排列的 ChunkInfo 列表
        ordered_text   : 拼接后的文本（与 query_db 返回值一致）
        config         : 本次检索使用的配置
        latency_ms     : 总耗时（毫秒）
        latency_breakdown: 各阶段耗时（召回 / RRF / Rerank / LiM）
        query_count    : 实际送入检索的 query 数量（含 rewrite）
    """
    chunks: list[ChunkInfo] = field(default_factory=list)
    ordered_text: str = ""
    config: RetrievalConfig | None = None
    latency_ms: float = 0.0
    latency_breakdown: dict[str, float] = field(default_factory=dict)
    query_count: int = 0


def query_db_structured(
    queries: "list[RewriteQuery] | list[str] | str",
    config: RetrievalConfig | None = None,
) -> RetrievalResult:
    """
    结构化检索主入口，支持配置开关与完整评分输出。

    与 query_db 的区别：
      - 返回 RetrievalResult（含每个 chunk 的 rrf_score / ce_score / final_score / 各路排名）
      - 支持通过 RetrievalConfig 开关任意模块，做消融实验
      - 附带 latency_breakdown，用于 TTFT 工程分析

    示例：
        # 消融实验：关闭 HyDE，观察对 MRR 的影响
        cfg = RetrievalConfig(enable_hyde=False)
        result = query_db_structured("Handler 内存泄漏", config=cfg)
        for c in result.chunks:
            print(c.id, c.final_position, c.final_score)
    """
    cfg = config or RetrievalConfig()
    result = RetrievalResult(config=cfg)
    t_total_0 = time.perf_counter()

    # ── 输入归一化 ────────────────────────────────────────────────────────────
    if isinstance(queries, str):
        queries = [RewriteQuery(text=queries, type="original", weight=1.0,
                                routes=["dense", "hyde", "bm25"])]
    elif queries and isinstance(queries[0], str):
        rq_list: list[RewriteQuery] = []
        for i, q in enumerate(queries):
            q = q.strip()
            if not q:
                continue
            if i == 0:
                rq_list.append(RewriteQuery(text=q, type="original", weight=1.0,
                                            routes=["dense", "hyde", "bm25"]))
            else:
                rq_list.append(RewriteQuery(text=q, type="semantic", weight=0.9,
                                            routes=["dense", "hyde"]))
        queries = rq_list

    queries = [rq for rq in queries if rq.text.strip()]
    if not queries:
        result.latency_ms = (time.perf_counter() - t_total_0) * 1000
        return result

    original_query = queries[0].text
    result.query_count = len(queries)

    from ai_app1.retrieval import bm25_store
    from ai_app1.retrieval.reranker import rerank_chunks, reorder_lost_in_middle

    col_parent = _get_collection("android_parent")
    col_child  = _get_collection("android_child")
    col_hyde   = _get_collection("android_hyde")
    v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

    if not v2_ready:
        logger.warning("v2 collections 未就绪，回退旧版单路检索（请运行 init_vector_db_v2.py）")
        # 旧版结构化降级：仅返回文本与粗略耗时
        t0 = time.perf_counter()
        text = _legacy_query(original_query)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.ordered_text = text or ""
        if text:
            result.chunks.append(ChunkInfo(id="legacy", text=text, final_position=0))
        return result

    # ── 按配置选择性提交召回任务 ──────────────────────────────────────────────
    t_retrieve_0 = time.perf_counter()
    weighted_lists: list[tuple[list[str], float]] = []
    pid_best_dense_rank: dict[str, int] = {}
    pid_best_bm25_rank: dict[str, int] = {}

    n_tasks = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures: list[tuple[str, float, Future]] = []
        for rq in queries:
            if cfg.enable_dense and "dense" in rq.routes:
                futures.append(("dense", rq.weight, pool.submit(_query_dense, rq.text, col_child)))
                n_tasks += 1
            if cfg.enable_hyde and "hyde" in rq.routes:
                futures.append(("hyde", rq.weight, pool.submit(_query_hyde, rq.text, col_hyde)))
                n_tasks += 1
            if cfg.enable_bm25 and "bm25" in rq.routes:
                futures.append(("bm25", rq.weight, pool.submit(bm25_store.search, rq.text, cfg.bm25_top_k)))
                n_tasks += 1

        for kind, weight, f in futures:
            res = f.result()
            if kind == "bm25":
                pids = [r[0] for r in res]
                for rank, pid in enumerate(pids):
                    pid_best_bm25_rank[pid] = min(pid_best_bm25_rank.get(pid, 999), rank)
            else:
                pids = res
                if kind == "dense":
                    for rank, pid in enumerate(pids):
                        pid_best_dense_rank[pid] = min(pid_best_dense_rank.get(pid, 999), rank)
            weighted_lists.append((pids, weight))

    latency_retrieve = (time.perf_counter() - t_retrieve_0) * 1000

    # ── Weighted RRF 融合 ─────────────────────────────────────────────────────
    t_rrf_0 = time.perf_counter()
    rrf_results = _rrf_merge(weighted_lists)
    rrf_score_map = {pid: score for pid, score in rrf_results}
    top20_ids = [pid for pid, _ in rrf_results[:20]]
    latency_rrf = (time.perf_counter() - t_rrf_0) * 1000

    # ── 拉取 parent 文本 ──────────────────────────────────────────────────────
    parent_texts = _fetch_parents(top20_ids, col_parent)

    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for pid in top20_ids:
        if pid in parent_texts and pid not in seen_ids:
            seen_ids.add(pid)
            candidates.append({
                "id": pid,
                "text": parent_texts[pid],
                "rrf_score": rrf_score_map[pid],
                "vector_rank": pid_best_dense_rank.get(pid, 999),
                "bm25_rank": pid_best_bm25_rank.get(pid, 999),
            })

    if not candidates:
        result.latency_ms = (time.perf_counter() - t_total_0) * 1000
        result.latency_breakdown = {"retrieve": latency_retrieve, "rrf": latency_rrf}
        return result

    # ── Rerank（可按配置关闭）──────────────────────────────────────────────────
    t_rerank_0 = time.perf_counter()
    if cfg.enable_rerank:
        reranked = rerank_chunks(original_query, candidates, top_k=cfg.rerank_top_k)
    else:
        # 关闭 rerank：按 RRF 分数降序截断
        reranked = sorted(candidates, key=lambda x: x["rrf_score"], reverse=True)[:cfg.rerank_top_k]
        for c in reranked:
            c["ce_score"] = 0.0
            c["final_score"] = c["rrf_score"]
    latency_rerank = (time.perf_counter() - t_rerank_0) * 1000

    # ── Lost-in-Middle 重排（可按配置关闭）─────────────────────────────────────
    t_lim_0 = time.perf_counter()
    if cfg.enable_lost_in_middle:
        ordered = reorder_lost_in_middle(reranked)
    else:
        ordered = reranked
    latency_lim = (time.perf_counter() - t_lim_0) * 1000

    # ── 组装结构化输出 ─────────────────────────────────────────────────────────
    result.chunks = [
        ChunkInfo(
            id=c["id"],
            text=c["text"],
            rrf_score=c.get("rrf_score", 0.0),
            ce_score=c.get("ce_score", 0.0),
            final_score=c.get("final_score", 0.0),
            vector_rank=c.get("vector_rank", 999),
            bm25_rank=c.get("bm25_rank", 999),
            final_position=pos,
        )
        for pos, c in enumerate(ordered)
    ]
    result.ordered_text = "\n\n".join(c["text"] for c in ordered)
    result.latency_ms = (time.perf_counter() - t_total_0) * 1000
    result.latency_breakdown = {
        "retrieve": round(latency_retrieve, 2),
        "rrf": round(latency_rrf, 2),
        "rerank": round(latency_rerank, 2),
        "lost_in_middle": round(latency_lim, 2),
    }

    logger.info(
        f"query_db_structured 完成: {len(result.chunks)} 个片段, "
        f"total={result.latency_ms:.0f}ms, breakdown={result.latency_breakdown}, "
        f"config=({cfg.summary()})"
    )
    return result


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
