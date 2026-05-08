"""
BM25 稀疏检索模块：基于 android_parent collection 构建倒排索引。

懒加载：首次调用 search() 时自动从 ChromaDB 加载数据并构建索引，
后续调用复用内存缓存，避免启动开销。
调用 reload() 可在 re-indexing 后强制重建。
"""
from __future__ import annotations

import re
import logging
import chromadb
from rank_bm25 import BM25Plus  # BM25Okapi IDF 在 df>=N/2 时归零，Plus 加偏置项避免
from ai_app1.core.config import CHROMA_DB_PATH

logger = logging.getLogger("bm25_store")

_bm25: BM25Plus | None = None
_doc_ids: list[str] = []
_doc_texts: list[str] = []


def _tokenize(text: str) -> list[str]:
    """中英文混合分词：中文双字滑窗 + 英文/数字按词"""
    return re.findall(r"[一-鿿]{1,2}|[a-zA-Z0-9]+", text.lower())


def _ensure_loaded() -> bool:
    global _bm25, _doc_ids, _doc_texts
    if _bm25 is not None:
        return True

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        col = client.get_collection("android_parent")
    except Exception as e:
        logger.warning(f"android_parent 未就绪，BM25 不可用: {e}")
        return False

    result = col.get()
    _doc_ids = result["ids"]
    _doc_texts = result["documents"]
    tokenized = [_tokenize(d) for d in _doc_texts]
    _bm25 = BM25Plus(tokenized)
    logger.info(f"BM25 索引就绪: {len(_doc_texts)} 个 parent chunks")
    return True


def search(query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
    """
    BM25 全文检索。

    Returns:
        [(parent_id, text, bm25_score), ...] 按分值降序，过滤零分结果
    """
    if not _ensure_loaded():
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = [(
        _doc_ids[idx],
        _doc_texts[idx],
        float(scores[idx]),
    ) for idx in top_indices if scores[idx] > 0]

    top_score = results[0][2] if results else 0.0
    logger.debug(f"BM25 检索: query={query[:30]!r}, 命中={len(results)}, top_score={top_score:.3f}")
    return results


def reload():
    """re-indexing 后调用，强制重建 BM25 索引"""
    global _bm25, _doc_ids, _doc_texts
    _bm25 = None
    _doc_ids = []
    _doc_texts = []
    logger.info("BM25 索引已清空，下次 search() 时重建")
    _ensure_loaded()
