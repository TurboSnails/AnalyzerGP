"""
精排模块（Phase 3）— CrossEncoder 语义重排

rerank_chunks:
    1. 使用 BGE CrossEncoder 对 (query, doc) pair 进行语义相关性打分
    2. 与 RRF 分数做加权融合（CrossEncoder 为主，RRF 为辅）
    3. 取 Top-K

reorder_lost_in_middle:
    按"Lost in the Middle"理论重排上下文顺序：
    最相关 → 首位；次相关 → 末位；其余居中。
    LLM 对首尾注意力最强，核心内容不会被埋在中间。
"""
from __future__ import annotations

import logging
import math
import re
import threading
from typing import cast

from sentence_transformers import CrossEncoder

from ai_app1.core.config import RERANKER_MODEL

logger = logging.getLogger("reranker")


# ─── CrossEncoder Reranker 服务 ───────────────────────────────────────────────

class BgeRerankerService:
    """封装 CrossEncoder，对外暴露 predict 语义打分。"""

    def __init__(self, model_name_or_path: str | None = None) -> None:
        self._path = model_name_or_path or RERANKER_MODEL
        self._model: CrossEncoder | None = None
        # HF fast tokenizer (Rust RefCell) 不允许并发调用；用锁序列化 predict
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info(f"正在加载 CrossEncoder reranker: {self._path}")
        # token=False 避免 .env / IDE 缓存中过期 HF token 干扰公开模型加载
        self._model = CrossEncoder(self._path, max_length=512, token=False)
        logger.info("CrossEncoder reranker 加载完成")

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        """
        对 (query, doc) pair 列表预测相关性分数。

        Args:
            pairs: [[query, doc], ...]
            batch_size: 预测批次大小

        Returns:
            每个 pair 的相关性分数（原始 logits），值越大越相关
        """
        self._ensure_model()
        if not pairs:
            return []
        with self._lock:
            scores = self._model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # 统一返回 list[float]
            if hasattr(scores, "tolist"):
                return cast(list[float], scores.tolist())
            return cast(list[float], list(scores))


_reranker_service: BgeRerankerService | None = None


def _get_reranker_service(model_path: str | None = None) -> BgeRerankerService:
    """全局 Reranker 服务（惰性加载模型）。"""
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = BgeRerankerService(model_name_or_path=model_path)
    return _reranker_service


def reset_reranker_service() -> None:
    """释放模型引用（测试或热替换模型时）。"""
    global _reranker_service
    if _reranker_service is not None:
        _reranker_service._model = None
        _reranker_service = None
    logger.info("BgeRerankerService 已重置")


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[一-鿿]{1,2}|[a-zA-Z0-9]+", text.lower()))


def _term_overlap(query_tokens: set[str], doc: str) -> float:
    """query 词中有多少比例出现在 doc 里"""
    if not query_tokens:
        return 0.0
    doc_tokens = _tokenize(doc)
    return len(query_tokens & doc_tokens) / len(query_tokens)


def _fallback_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """CrossEncoder 不可用时的降级排序（保留旧版 RRF + term_overlap 逻辑）"""
    q_tokens = _tokenize(query)
    max_rrf = max(c.get("rrf_score", 0.0) for c in candidates) or 1.0

    for c in candidates:
        normalized_rrf = c.get("rrf_score", 0.0) / max_rrf
        term_score = _term_overlap(q_tokens, c["text"])
        c["final_score"] = 0.80 * normalized_rrf + 0.20 * term_score
        c["ce_score"] = 0.0

    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]
    logger.warning(f"Fallback 规则排序: {len(candidates)} → {len(ranked)} 个")
    return ranked


# ─── 主精排函数 ───────────────────────────────────────────────────────────────

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
        top_k 个候选，按 final_score 降序，每个 candidate 追加字段:
            ce_score   : float  CrossEncoder 语义分数（sigmoid 归一化到 0~1）
            final_score: float  加权融合分
    """
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]

    try:
        ce_scores = _get_reranker_service().predict(pairs, batch_size=32)
    except Exception as exc:
        logger.warning(f"CrossEncoder 预测失败，降级到规则排序: {exc}")
        return _fallback_rerank(query, candidates, top_k)

    # 对 CrossEncoder logits 做 sigmoid 映射到 0~1，便于与 RRF 加权融合
    ce_probs = [1.0 / (1.0 + math.exp(-s)) for s in ce_scores]

    max_rrf = max(c.get("rrf_score", 0.0) for c in candidates) or 1.0

    for idx, c in enumerate(candidates):
        ce_norm = ce_probs[idx]
        rrf_norm = c.get("rrf_score", 0.0) / max_rrf

        # CrossEncoder 语义分为主（0.75），RRF 召回分为辅（0.25）
        c["ce_score"] = ce_probs[idx]
        c["final_score"] = 0.75 * ce_norm + 0.25 * rrf_norm

    ranked = sorted(candidates, key=lambda x: x["final_score"], reverse=True)[:top_k]
    top_score = ranked[0]["final_score"] if ranked else 0.0
    logger.info(
        f"CrossEncoder 重排: {len(candidates)} → {len(ranked)} 个, "
        f"top_final={top_score:.3f}, top_ce={ranked[0]['ce_score']:.3f}, "
        f"query={query[:20]!r}"
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
