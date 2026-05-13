"""
Rule-Based Fallback Reranker

无需加载模型，基于 RRF 分数和词元重叠实现轻量精排。
用于 CrossEncoder 不可用时的降级，或低延迟场景。
"""
from __future__ import annotations

import re

from rag_framework.core.logger import get_logger
from rag_framework.rerank.base import RankedDoc, Reranker

_logger = get_logger("rag.rerank.fallback")

_TOKEN_RE = re.compile(r"[一-鿿]{1,2}|[a-zA-Z0-9]+")


class FallbackReranker(Reranker):
    """
    规则降级排序器。

    score = rrf_weight × (rrf_score / max_rrf) + overlap_weight × token_overlap
    """

    def __init__(
        self,
        rrf_weight: float = 0.80,
        overlap_weight: float = 0.20,
    ) -> None:
        self._rrf_w = rrf_weight
        self._ovl_w = overlap_weight

    def rerank(
        self,
        query: str,
        candidates: list[RankedDoc],
        top_k: int = 5,
    ) -> list[RankedDoc]:
        if not candidates:
            return []

        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        max_rrf = max(c.rrf_score for c in candidates) or 1.0

        for c in candidates:
            doc_tokens = set(_TOKEN_RE.findall(c.text.lower()))
            overlap = len(q_tokens & doc_tokens) / len(q_tokens) if q_tokens else 0.0
            c.score = self._rrf_w * (c.rrf_score / max_rrf) + self._ovl_w * overlap
            c.ce_score = 0.0

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        _logger.info(
            f"FallbackReranker: {len(candidates)} → {len(ranked)}, "
            f"top_score={ranked[0].score:.3f}"
        )
        return ranked
