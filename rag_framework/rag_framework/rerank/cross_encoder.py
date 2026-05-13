"""
CrossEncoder Reranker 实现

基于 sentence-transformers CrossEncoder，sigmoid 归一化。
"""
from __future__ import annotations

import math
import re
import threading
from typing import cast

from sentence_transformers import CrossEncoder

from rag_framework.core.exceptions import ModelLoadError, ModelNotFoundError
from rag_framework.core.logger import reranker_logger
from rag_framework.rerank.base import Reranker, RankedDoc


class CrossEncoderReranker(Reranker):
    """基于 CrossEncoder 的精排器。"""

    def __init__(
        self,
        model_path: str,
        max_length: int = 512,
        batch_size: int = 32,
    ) -> None:
        self._path = model_path
        self._max_length = max_length
        self._batch_size = batch_size
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        reranker_logger.info(f"正在加载 CrossEncoder reranker: {self._path}")
        try:
            self._model = CrossEncoder(
                self._path,
                max_length=self._max_length,
                token=False,
            )
        except Exception as e:
            raise ModelLoadError(f"加载 reranker 失败: {e}") from e
        reranker_logger.info("CrossEncoder reranker 加载完成")

    def rerank(
        self,
        query: str,
        candidates: list[RankedDoc],
        top_k: int = 5,
    ) -> list[RankedDoc]:
        if not candidates:
            return []

        pairs = [[query, c.text] for c in candidates]

        try:
            self._ensure_model()
            with self._lock:
                scores = self._model.predict(  # type: ignore[union-attr]
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
        except Exception as exc:
            reranker_logger.warning(f"CrossEncoder 预测失败，降级到规则排序: {exc}")
            return self._fallback_rerank(query, candidates, top_k)

        # sigmoid 归一化到 0~1
        ce_probs = [1.0 / (1.0 + math.exp(-s)) for s in scores]
        max_rrf = max(c.rrf_score for c in candidates) or 1.0

        for idx, c in enumerate(candidates):
            c.ce_score = ce_probs[idx]
            rrf_norm = c.rrf_score / max_rrf
            c.score = 0.75 * ce_probs[idx] + 0.25 * rrf_norm

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        top_score = ranked[0].score if ranked else 0.0
        reranker_logger.info(
            f"CrossEncoder 重排: {len(candidates)} → {len(ranked)} 个, "
            f"top_final={top_score:.3f}, top_ce={ranked[0].ce_score:.3f}, "
            f"query={query[:20]!r}"
        )
        return ranked

    @staticmethod
    def _fallback_rerank(
        query: str, candidates: list[RankedDoc], top_k: int
    ) -> list[RankedDoc]:
        """规则降级排序。"""
        tokens = set(re.findall(r"[一-鿿]{1,2}|[a-zA-Z0-9]+", query.lower()))
        max_rrf = max(c.rrf_score for c in candidates) or 1.0

        for c in candidates:
            doc_tokens = set(re.findall(r"[一-鿿]{1,2}|[a-zA-Z0-9]+", c.text.lower()))
            overlap = len(tokens & doc_tokens) / len(tokens) if tokens else 0.0
            c.score = 0.80 * (c.rrf_score / max_rrf) + 0.20 * overlap
            c.ce_score = 0.0

        ranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]
        reranker_logger.warning(f"Fallback 规则排序: {len(candidates)} → {len(ranked)} 个")
        return ranked
