"""
Hybrid Retriever — 混合检索 + RRF 融合 + Rerank + Lost-in-Middle

编排 Dense、HyDE、BM25 多路召回，融合后精排。
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_framework.core.config import RAGSettings
from rag_framework.core.logger import retrieval_logger
from rag_framework.domain.base import DomainPlugin, QueryRoute
from rag_framework.retrieval.base import Retriever, RetrievalResult, RetrievedDoc

from rag_framework.rerank.base import Reranker, RankedDoc

if TYPE_CHECKING:
    from rag_framework.embedding.base import Embedder
    from rag_framework.retrieval.dense import DenseStore
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


class HybridRetriever(Retriever):
    """
    混合检索器。

    多路并发召回 → Weighted RRF 融合 → CrossEncoder 精排 → Lost-in-Middle 重排。
    """

    def __init__(
        self,
        settings: RAGSettings,
        embedder: Embedder,
        dense_store: DenseStore,
        sparse_store: BM25Store,
        reranker: Reranker,
        domain: DomainPlugin,
    ) -> None:
        self._cfg = HybridConfig(
            rrf_k=settings.rrf_k,
            dense_top_k=settings.dense_top_k,
            hyde_top_k=settings.hyde_top_k,
            bm25_top_k=settings.bm25_top_k,
            rerank_top_k=settings.rerank_top_k,
            max_child_distance=settings.max_child_distance,
            low_confidence_threshold=settings.low_confidence_threshold,
        )
        self._embedder = embedder
        self._dense = dense_store
        self._sparse = sparse_store
        self._reranker = reranker
        self._domain = domain

    def retrieve(
        self,
        query: str | QueryRoute | list[QueryRoute],
        top_k: int = 10,
    ) -> RetrievalResult:
        t0 = time.perf_counter()

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
        collections = self._domain.get_collection_names()

        # 检查 v2 collections
        col_parent = self._dense.get_collection(collections.parent)
        col_child = self._dense.get_collection(collections.child)
        col_hyde = self._dense.get_collection(collections.hyde)
        v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

        if not v2_ready:
            retrieval_logger.warning("v2 collections 未就绪，回退旧版单路检索")
            return self._legacy_retrieve(original_query)

        # 多路并发召回
        t1 = time.perf_counter()
        weighted_lists, pid_best_dense, pid_best_bm25 = self._multi_route_fetch(
            queries, collections
        )
        t_fetch = time.perf_counter() - t1

        # Weighted RRF 融合
        rrf_results = self._rrf_merge(weighted_lists)
        rrf_score_map = {pid: score for pid, score in rrf_results}
        top20_ids = [pid for pid, _ in rrf_results[:20]]

        # 拉取 parent 文本
        parent_texts = self._dense.fetch_parents(top20_ids, collections.parent)

        # 构建候选
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
            return RetrievalResult(docs=[], latency_ms=(time.perf_counter() - t0) * 1000)

        # 精排
        if self._cfg.enable_rerank:
            reranked = self._reranker.rerank(original_query, candidates, top_k=self._cfg.rerank_top_k)
        else:
            for c in candidates:
                c.score = c.rrf_score
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:self._cfg.rerank_top_k]

        top_ce = float(reranked[0].ce_score) if reranked else 0.0

        # Lost-in-Middle 重排
        ordered = self._lost_in_middle(reranked) if self._cfg.enable_lost_in_middle else reranked

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
        retrieval_logger.info(
            f"HybridRetriever: {len(docs)} 个片段, top_ce={top_ce:.3f}, "
            f"fetch={t_fetch*1000:.0f}ms, total={total_ms:.0f}ms"
        )

        return RetrievalResult(
            docs=docs,
            query=original_query,
            latency_ms=total_ms,
            metadata={"top_ce": top_ce, "n_chunks": len(docs)},
        )

    # ─── 内部方法 ───────────────────────────────────────────────────────────────

    def _multi_route_fetch(
        self,
        queries: list[QueryRoute],
        collections,
    ) -> tuple[list[tuple[list[str], float]], dict[str, int], dict[str, int]]:
        weighted_lists: list[tuple[list[str], float]] = []
        pid_best_dense: dict[str, int] = {}
        pid_best_bm25: dict[str, int] = {}

        n_tasks = sum(len(q.routes) for q in queries)
        with ThreadPoolExecutor(max_workers=max(n_tasks, 1)) as pool:
            futures = []
            for q in queries:
                if "dense" in q.routes and self._cfg.enable_dense:
                    futures.append((
                        "dense", q.weight,
                        pool.submit(self._query_dense, q.text, collections.child)
                    ))
                if "hyde" in q.routes and self._cfg.enable_hyde:
                    futures.append((
                        "hyde", q.weight,
                        pool.submit(self._query_dense, q.text, collections.hyde)
                    ))
                if "bm25" in q.routes and self._cfg.enable_bm25:
                    futures.append((
                        "bm25", q.weight,
                        pool.submit(self._sparse.search, q.text, self._cfg.bm25_top_k)
                    ))

            for kind, weight, f in futures:
                result = f.result()
                if kind == "bm25":
                    pids = [r[0] for r in result]
                    for rank, pid in enumerate(pids):
                        pid_best_bm25[pid] = min(pid_best_bm25.get(pid, 999), rank)
                else:
                    pids = result
                    for rank, pid in enumerate(pids):
                        pid_best_dense[pid] = min(pid_best_dense.get(pid, 999), rank)
                weighted_lists.append((pids, weight))

        return weighted_lists, pid_best_dense, pid_best_bm25

    def _query_dense(self, query: str, collection_name: str) -> list[str]:
        """向量检索 → parent_id 聚合。"""
        try:
            ids, distances, metas = self._dense.query(
                query, collection_name,
                n_results=25, max_distance=self._cfg.max_child_distance,
            )
        except Exception:
            return []

        parent_hits: dict[str, list[float]] = {}
        for meta, dist in zip(metas, distances):
            pid = meta.get("parent_id", "")
            if pid:
                parent_hits.setdefault(pid, []).append(dist)

        scored = []
        for pid, dists in parent_hits.items():
            score = min(dists) - 0.05 * (len(dists) - 1)
            scored.append((pid, score))

        scored.sort(key=lambda x: x[1])
        return [s[0] for s in scored[:self._cfg.dense_top_k]]

    @staticmethod
    def _rrf_merge(ranked_lists: list[tuple[list[str], float]]) -> list[tuple[str, float]]:
        scores: dict[str, float] = {}
        for lst, weight in ranked_lists:
            for rank, doc_id in enumerate(lst):
                scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rank + 60)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    @staticmethod
    def _lost_in_middle(chunks: list[RankedDoc]) -> list[RankedDoc]:
        """Lost-in-the-Middle 重排：最相关首位，次相关末位。"""
        if len(chunks) <= 2:
            return chunks
        return [chunks[0]] + chunks[2:] + [chunks[1]]

    def _legacy_retrieve(self, query: str) -> RetrievalResult:
        """旧版单路检索回退。"""
        # 简化实现：直接查 parent collection
        collections = self._domain.get_collection_names()
        try:
            ids, distances, _ = self._dense.query(
                query, collections.parent, n_results=5, max_distance=1.2
            )
            if not ids:
                return RetrievalResult(docs=[], query=query)
            texts = self._dense.fetch_parents(ids, collections.parent)
            docs = [
                RetrievedDoc(id=i, text=texts.get(i, ""), score=1.0 - d)
                for i, d in zip(ids, distances)
                if i in texts
            ]
            return RetrievalResult(docs=docs, query=query)
        except Exception:
            return RetrievalResult(docs=[], query=query)
