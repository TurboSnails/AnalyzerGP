"""
Hybrid Retriever — 混合检索 + RRF 融合 + Rerank + Lost-in-Middle

编排 Dense、HyDE、BM25 多路异步并发召回，融合后精排。

并发模型：
  - 每路检索（Dense/HyDE/BM25）通过 asyncio.to_thread 卸载到线程池
  - asyncio.gather 并发执行所有路，asyncio.wait_for 控制单路超时
  - Rerank（CrossEncoder GPU 推理）同样 to_thread + timeout
  - fetch_parents（ChromaDB 批量 get）to_thread + timeout
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag_framework.core.config import RAGSettings
from rag_framework.core.factories import register_retriever
from rag_framework.core.logger import retrieval_logger
from rag_framework.domain.base import DomainPlugin, QueryRoute
from rag_framework.retrieval.base import Retriever, RetrievalResult, RetrievedDoc, VectorStore
from rag_framework.rerank.base import Reranker, RankedDoc

if TYPE_CHECKING:
    from rag_framework.eval.failure_analysis import FailureCollector

if TYPE_CHECKING:
    from rag_framework.embedding.base import Embedder
    from rag_framework.retrieval.sparse import BM25Store


@dataclass
class HybridConfig:
    """混合检索配置。"""
    rrf_k: int = 60
    dense_top_k: int = 10
    hyde_top_k: int = 5
    bm25_top_k: int = 10
    rerank_top_k: int = 3
    max_child_distance: float = 1.3
    low_confidence_threshold: float = 0.30
    enable_dense: bool = True
    enable_hyde: bool = True
    enable_bm25: bool = True
    enable_rerank: bool = True
    enable_lost_in_middle: bool = True
    branch_timeout: float = 10.0    # 单路检索超时（秒）
    rerank_timeout: float = 15.0    # Rerank 超时（秒）


class HybridRetriever(Retriever):
    """
    混合检索器。

    多路异步并发召回 → Weighted RRF 融合 → CrossEncoder 精排 → Lost-in-Middle 重排。
    """

    def __init__(
        self,
        settings: RAGSettings,
        embedder: "Embedder",
        vector_store: VectorStore,
        sparse_store: "BM25Store",
        reranker: Reranker,
        domain: DomainPlugin,
        domain_filter: str = "",
    ) -> None:
        self._cfg = HybridConfig(
            rrf_k=settings.rrf_k,
            dense_top_k=settings.dense_top_k,
            hyde_top_k=settings.hyde_top_k,
            bm25_top_k=settings.bm25_top_k,
            rerank_top_k=settings.rerank_top_k,
            max_child_distance=settings.max_child_distance,
            low_confidence_threshold=settings.low_confidence_threshold,
            branch_timeout=settings.retrieval_branch_timeout,
            rerank_timeout=settings.retrieval_rerank_timeout,
        )
        self._embedder = embedder
        self._dense = vector_store
        self._sparse = sparse_store
        self._reranker = reranker
        self._domain = domain
        self._domain_filter = domain_filter

    async def retrieve(
        self,
        query: str | QueryRoute | list[QueryRoute],
        top_k: int = 10,
    ) -> RetrievalResult:
        # Lazy imports to break circular dependency with rag_framework.eval package
        from rag_framework.eval.latency_breakdown import PhaseTimer
        from rag_framework.eval.retrieval_trace import (
            BranchTrace,
            RerankTrace,
            RetrievalTrace,
            record_trace,
        )

        t0 = time.perf_counter()
        timer = PhaseTimer()
        trace = RetrievalTrace()

        # 输入归一化
        if isinstance(query, str):
            queries = [QueryRoute(text=query)]
        elif isinstance(query, QueryRoute):
            queries = [query]
        else:
            queries = [q for q in query if q.text.strip()]

        if not queries:
            return RetrievalResult(docs=[], latency_ms=0.0)

        original_query = queries[0].text
        trace.original_query = original_query
        trace.query = original_query

        # 从多路路由中提取改写查询（第一个非 original 类型的 route）
        for q in queries[1:]:
            if q.type != "original" and q.text != original_query:
                trace.rewritten_query = q.text
                trace.rewrite_type = q.type
                break

        collections = self._domain.get_collection_names()

        # 检查 v2 collections
        col_parent = self._dense.get_collection(collections.parent)
        col_child  = self._dense.get_collection(collections.child)
        col_hyde   = self._dense.get_collection(collections.hyde)
        v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

        if not v2_ready:
            retrieval_logger.warning("v2 collections 未就绪，回退旧版单路检索")
            return await self._legacy_retrieve(original_query)

        # ── 多路异步并发召回 ────────────────────────────────────────────────────
        t1 = time.perf_counter()
        weighted_lists, pid_best_dense, pid_best_bm25, branch_traces = await self._async_multi_route_fetch(
            queries, collections, trace
        )
        t_fetch = time.perf_counter() - t1
        timer.record("fetch", t_fetch * 1000)
        trace.branches = branch_traces

        # ── Weighted RRF 融合 ───────────────────────────────────────────────────
        t_rrf_start = time.perf_counter()
        rrf_results  = self._rrf_merge(weighted_lists)
        rrf_score_map = {pid: score for pid, score in rrf_results}
        top20_ids    = [pid for pid, _ in rrf_results[:20]]
        t_rrf = (time.perf_counter() - t_rrf_start) * 1000
        timer.record("rrf", t_rrf)
        trace.rrf_latency_ms = t_rrf
        trace.rrf_input_count = len(weighted_lists)
        trace.rrf_output_count = len(rrf_results)

        # ── 拉取 parent 文本（to_thread，避免阻塞 event loop）────────────────────
        t_fetch_parent_start = time.perf_counter()
        try:
            parent_texts = await asyncio.wait_for(
                asyncio.to_thread(self._dense.fetch_parents, top20_ids, collections.parent),
                timeout=self._cfg.branch_timeout,
            )
        except asyncio.TimeoutError:
            retrieval_logger.warning("fetch_parents 超时，返回空结果")
            total_ms = (time.perf_counter() - t0) * 1000
            trace.final_latency_ms = total_ms
            record_trace(trace)
            return RetrievalResult(docs=[], latency_ms=total_ms)
        timer.record("fetch_parents", (time.perf_counter() - t_fetch_parent_start) * 1000)

        # ── 构建候选列表 ─────────────────────────────────────────────────────────
        seen: set[str] = set()
        candidates: list[RankedDoc] = []
        for pid in top20_ids:
            if pid in parent_texts and pid not in seen:
                seen.add(pid)
                candidates.append(RankedDoc(
                    id=pid,
                    text=parent_texts[pid],
                    rrf_score=rrf_score_map[pid],
                    vector_rank=pid_best_dense.get(pid, 999),
                    bm25_rank=pid_best_bm25.get(pid, 999),
                ))

        if not candidates:
            total_ms = (time.perf_counter() - t0) * 1000
            trace.final_latency_ms = total_ms
            record_trace(trace)
            return RetrievalResult(docs=[], latency_ms=total_ms)

        # ── Rerank（CrossEncoder，to_thread + timeout）───────────────────────────
        t_rerank_start = time.perf_counter()
        rerank_trace = RerankTrace()
        if self._cfg.enable_rerank:
            try:
                reranked = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._reranker.rerank,
                        original_query,
                        candidates,
                        self._cfg.rerank_top_k,
                    ),
                    timeout=self._cfg.rerank_timeout,
                )
                rerank_trace.status = "success"
            except asyncio.TimeoutError:
                retrieval_logger.warning(f"Rerank 超时（>{self._cfg.rerank_timeout}s），降级 RRF 排序")
                rerank_trace.status = "timeout"
                rerank_trace.error = "timeout"
                for c in candidates:
                    c.score = c.rrf_score
                    c.ce_score = 0.0
                reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[
                    : self._cfg.rerank_top_k
                ]
        else:
            for c in candidates:
                c.score = c.rrf_score
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[
                : self._cfg.rerank_top_k
            ]
            rerank_trace.status = "skipped"
        t_rerank = (time.perf_counter() - t_rerank_start) * 1000
        timer.record("rerank", t_rerank)
        rerank_trace.latency_ms = t_rerank
        rerank_trace.input_count = len(candidates)
        rerank_trace.output_count = len(reranked)
        top_ce = float(reranked[0].ce_score) if reranked else 0.0
        rerank_trace.top_ce_score = top_ce
        trace.rerank = rerank_trace

        # ── Lost-in-Middle 重排 ──────────────────────────────────────────────────
        ordered = self._lost_in_middle(reranked) if self._cfg.enable_lost_in_middle else reranked
        trace.lost_in_middle = self._cfg.enable_lost_in_middle

        docs = [
            RetrievedDoc(
                id=c.id,
                text=c.text,
                score=c.score,
                source="hybrid",
                metadata={
                    "ce_score": c.ce_score,
                    "rrf_score": c.rrf_score,
                    "vector_rank": c.vector_rank,
                    "bm25_rank": c.bm25_rank,
                },
            )
            for c in ordered
        ]

        total_ms = (time.perf_counter() - t0) * 1000
        trace.final_latency_ms = total_ms
        trace.final_chunk_count = len(docs)
        trace.final_top_ids = [d.id for d in docs]
        trace.top_ce_score = top_ce

        retrieval_logger.info(
            f"HybridRetriever: {len(docs)} 个片段, top_ce={top_ce:.3f}, "
            f"fetch={t_fetch*1000:.0f}ms, total={total_ms:.0f}ms"
        )
        retrieval_logger.info(trace.print_trace())
        record_trace(trace)

        return RetrievalResult(
            docs=docs,
            query=original_query,
            latency_ms=total_ms,
            metadata={"top_ce": top_ce, "n_chunks": len(docs), "trace": trace.to_dict()},
        )

    # ─── 异步并发多路召回 ────────────────────────────────────────────────────────

    async def _async_multi_route_fetch(
        self,
        queries: list[QueryRoute],
        collections,
        trace: RetrievalTrace | None = None,
    ) -> tuple[list[tuple[list[str], float]], dict[str, int], dict[str, int], list[BranchTrace]]:
        """
        用 asyncio.gather 并发执行所有检索分支，每路独立 timeout。
        各分支在线程池中运行（to_thread），不阻塞事件循环。
        返回分支 trace 列表用于 observability。
        """
        # Lazy import to break circular dependency with rag_framework.eval package
        from rag_framework.eval.retrieval_trace import BranchTrace
        task_meta: list[tuple[str, float]] = []   # (kind, weight)
        coros: list = []

        where = None
        if self._domain_filter:
            where = {"domain": {"$eq": self._domain_filter}}

        for q in queries:
            if "dense" in q.routes and self._cfg.enable_dense:
                coros.append(asyncio.wait_for(
                    asyncio.to_thread(
                        self._query_dense, q.text, collections.child, where
                    ),
                    timeout=self._cfg.branch_timeout,
                ))
                task_meta.append(("dense", q.weight))

            if "hyde" in q.routes and self._cfg.enable_hyde:
                coros.append(asyncio.wait_for(
                    asyncio.to_thread(
                        self._query_dense, q.text, collections.hyde, where
                    ),
                    timeout=self._cfg.branch_timeout,
                ))
                task_meta.append(("hyde", q.weight))

            if "bm25" in q.routes and self._cfg.enable_bm25:
                coros.append(asyncio.wait_for(
                    asyncio.to_thread(
                        self._sparse.search,
                        q.text,
                        self._cfg.bm25_top_k,
                        self._domain_filter,
                    ),
                    timeout=self._cfg.branch_timeout,
                ))
                task_meta.append(("bm25", q.weight))

        # 并发执行，单路超时不中断其他路
        t_start = time.perf_counter()
        results = await asyncio.gather(*coros, return_exceptions=True)
        total_fetch_ms = (time.perf_counter() - t_start) * 1000

        weighted_lists: list[tuple[list[str], float]] = []
        pid_best_dense: dict[str, int] = {}
        pid_best_bm25: dict[str, int] = {}
        branch_traces: list[BranchTrace] = []

        for (kind, weight), result in zip(task_meta, results):
            bt = BranchTrace(kind=kind, query_text=queries[0].text, weight=weight)
            if isinstance(result, (Exception, asyncio.TimeoutError)):
                retrieval_logger.warning(f"{kind} 分支失败/超时: {result!r}")
                bt.status = "timeout" if isinstance(result, asyncio.TimeoutError) else "error"
                bt.error = str(result)
                branch_traces.append(bt)
                continue

            if kind == "bm25":
                pids = [r[0] for r in result]
                for rank, pid in enumerate(pids):
                    pid_best_bm25[pid] = min(pid_best_bm25.get(pid, 999), rank)
            else:
                pids = result
                for rank, pid in enumerate(pids):
                    pid_best_dense[pid] = min(pid_best_dense.get(pid, 999), rank)

            bt.status = "success"
            bt.result_count = len(pids)
            bt.top_ids = pids[:5]
            # 均摊总耗时（近似），更精确需逐个计时
            bt.latency_ms = total_fetch_ms / len(task_meta) if task_meta else 0.0
            branch_traces.append(bt)
            weighted_lists.append((pids, weight))

        return weighted_lists, pid_best_dense, pid_best_bm25, branch_traces

    # ─── 同步子步骤（在线程中运行）────────────────────────────────────────────────

    def _query_dense(
        self, query: str, collection_name: str, where: dict | None = None
    ) -> list[str]:
        """向量检索 → parent_id 聚合（同步，由 to_thread 调用）。"""
        try:
            ids, distances, metas = self._dense.query(
                query,
                collection_name,
                n_results=25,
                max_distance=self._cfg.max_child_distance,
                where=where,
            )
        except Exception:
            return []

        parent_hits: dict[str, list[float]] = {}
        for meta, dist in zip(metas, distances):
            pid = meta.get("parent_id", "")
            if pid:
                parent_hits.setdefault(pid, []).append(dist)

        scored = [
            (pid, min(dists) - 0.05 * (len(dists) - 1))
            for pid, dists in parent_hits.items()
        ]
        scored.sort(key=lambda x: x[1])
        return [s[0] for s in scored[: self._cfg.dense_top_k]]

    # ─── 纯函数工具 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_merge(ranked_lists: list[tuple[list[str], float]]) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for lst, weight in ranked_lists:
            for rank, doc_id in enumerate(lst):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rank + 60)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _lost_in_middle(chunks: list[RankedDoc]) -> list[RankedDoc]:
        """最相关首位，次相关末位，其余居中。"""
        if len(chunks) <= 2:
            return chunks
        return [chunks[0]] + chunks[2:] + [chunks[1]]

    # ─── 降级路径 ────────────────────────────────────────────────────────────────

    async def _legacy_retrieve(self, query: str) -> RetrievalResult:
        """旧版单路检索回退（无 child/hyde collection 时）。"""
        collections = self._domain.get_collection_names()
        try:
            hit_ids, distances, _ = await asyncio.wait_for(
                asyncio.to_thread(
                    self._dense.query,
                    query, collections.parent, 5, 1.2,
                ),
                timeout=self._cfg.branch_timeout,
            )
            if not hit_ids:
                return RetrievalResult(docs=[], query=query)
            texts = await asyncio.wait_for(
                asyncio.to_thread(self._dense.fetch_parents, hit_ids, collections.parent),
                timeout=self._cfg.branch_timeout,
            )
            docs = [
                RetrievedDoc(id=i, text=texts.get(i, ""), score=1.0 - d)
                for i, d in zip(hit_ids, distances)
                if i in texts
            ]
            return RetrievalResult(docs=docs, query=query)
        except Exception:
            return RetrievalResult(docs=[], query=query)


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_hybrid_retriever(
    settings: RAGSettings,
    embedder: Embedder,
    vector_store: VectorStore,
    reranker: Reranker,
    domain: DomainPlugin,
    sparse_store: Any = None,
    domain_filter: str = "",
) -> HybridRetriever:
    """创建 HybridRetriever，sparse_store 可选，默认自动创建 BM25。"""
    if sparse_store is None:
        from rag_framework.retrieval.sparse import BM25Store
        sparse_store = BM25Store(
            index_dir=settings.bm25_index_dir,
            chroma_path=settings.chroma_db_path,
            collection_name=domain.get_collection_names().parent,
        )
    return HybridRetriever(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
        sparse_store=sparse_store,
        reranker=reranker,
        domain=domain,
        domain_filter=domain_filter,
    )


register_retriever("hybrid", _create_hybrid_retriever)
