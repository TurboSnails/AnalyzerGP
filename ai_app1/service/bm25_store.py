"""
商业级 BM25 稀疏检索：Tantivy (Rust 引擎) + jieba 中文分词

架构优势（对比原 rank-bm25 内存方案）：
- 磁盘持久化索引 (mmap)，重启无需重建，首次查询毫秒级
- 增量写入，无需全量重载即可追加文档
- 内存占用与查询量相关，而非文档总量
- 支持千万级文档，BM25 评分由 Rust 层计算

索引位置：CHROMA_DB_PATH/../tantivy_bm25/
接口：与原 bm25_store.py 完全兼容（search / reload）
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import pathlib
import warnings

# jieba SyntaxWarning 来自其内部 re 模式，不影响功能
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba
    jieba.initialize()  # 预热，避免首次 search 时 0.3s 延迟

import tantivy
import chromadb
from ai_app1.core.config import CHROMA_DB_PATH

logger = logging.getLogger("bm25_store")

_INDEX_DIR = str(pathlib.Path(CHROMA_DB_PATH).parent / "tantivy_bm25")
_HEAP_SIZE = 200 * 1024 * 1024   # 200 MB writer heap，构建期临时用
_BATCH_SIZE = 10_000              # ChromaDB 分批拉取，避免 OOM

_index: tantivy.Index | None = None
_searcher: tantivy.Searcher | None = None
_lock = threading.Lock()


# ── Schema ────────────────────────────────────────────────────────────────────

def _make_schema() -> tantivy.Schema:
    sb = tantivy.SchemaBuilder()
    sb.add_text_field("doc_id", stored=True, tokenizer_name="raw")
    sb.add_text_field("body", stored=True, tokenizer_name="whitespace")
    sb.add_text_field("raw_text", stored=True, tokenizer_name="raw")
    return sb.build()


# ── 分词 ──────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> str:
    """jieba 精确模式分词，返回空格连接的 token 串，供 whitespace 分词器匹配"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tokens = jieba.cut(text, cut_all=False)
    # 过滤掉 ' " ( ) 等 tantivy 查询语法特殊字符，只保留纯字词 token
    import re
    return " ".join(t for t in tokens if t.strip() and re.match(r"^\w+$", t))


# ── 索引构建 ──────────────────────────────────────────────────────────────────

def _build_from_chroma(idx: tantivy.Index) -> bool:
    """从 ChromaDB android_parent 分批拉取并写入 Tantivy 索引"""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        col = client.get_collection("android_parent")
    except Exception as e:
        logger.warning("android_parent 未就绪，BM25 不可用: %s", e)
        return False

    total = col.count()
    logger.info("开始构建 Tantivy BM25 索引，共 %d 个文档", total)

    writer = idx.writer(heap_size=_HEAP_SIZE)
    indexed = 0
    offset = 0

    while offset < total:
        batch = col.get(limit=_BATCH_SIZE, offset=offset, include=["documents"])
        for doc_id, text in zip(batch["ids"], batch["documents"]):
            writer.add_document(tantivy.Document(
                doc_id=doc_id,
                body=_tokenize(text),
                raw_text=text,
            ))
        indexed += len(batch["ids"])
        offset += _BATCH_SIZE
        logger.info("  索引进度 %d / %d", indexed, total)

    writer.commit()
    writer.wait_merging_threads()
    logger.info("Tantivy BM25 索引构建完成：%d 个文档", indexed)
    return True


# ── 懒加载 ────────────────────────────────────────────────────────────────────

def _ensure_loaded() -> bool:
    global _index, _searcher

    if _searcher is not None:
        return True

    with _lock:
        if _searcher is not None:  # double-check after lock
            return True

        os.makedirs(_INDEX_DIR, exist_ok=True)
        schema = _make_schema()
        idx = tantivy.Index(schema=schema, path=_INDEX_DIR)

        # 空索引则从 ChromaDB 构建
        if idx.searcher().num_docs == 0:
            if not _build_from_chroma(idx):
                return False
            idx.reload()

        _index = idx
        _searcher = idx.searcher()
        logger.info("Tantivy BM25 索引就绪：%d 个文档", _searcher.num_docs)
        return True


# ── 公开接口 ──────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
    """
    BM25 全文检索（接口与原 bm25_store.py 兼容）

    Returns:
        [(parent_id, text, bm25_score), ...] 按分值降序，过滤零分结果
    """
    if not _ensure_loaded():
        return []

    tokenized = _tokenize(query)
    q = _index.parse_query(tokenized, ["body"])
    hits = _searcher.search(q, limit=top_k)

    results = [
        (doc["doc_id"][0], doc["raw_text"][0], float(score))
        for score, addr in hits.hits
        if (doc := _searcher.doc(addr)) and score > 0
    ]

    logger.debug(
        "BM25 检索: query=%r, 命中=%d, top_score=%.3f",
        query[:30],
        len(results),
        results[0][2] if results else 0.0,
    )
    return results


def add_documents(docs: list[tuple[str, str]]) -> None:
    """
    增量写入文档（re-indexing 后的追加场景）

    Args:
        docs: [(doc_id, text), ...]
    """
    if not _ensure_loaded():
        return

    writer = _index.writer(heap_size=_HEAP_SIZE)
    for doc_id, text in docs:
        writer.add_document(tantivy.Document(
            doc_id=doc_id,
            body=_tokenize(text),
            raw_text=text,
        ))
    writer.commit()
    writer.wait_merging_threads()

    _index.reload()
    global _searcher
    _searcher = _index.searcher()
    logger.info("增量写入 %d 个文档，当前总量 %d", len(docs), _searcher.num_docs)


def reload() -> None:
    """删除磁盘索引并从 ChromaDB 全量重建（re-indexing 后调用）"""
    global _index, _searcher

    with _lock:
        _index = None
        _searcher = None

    if os.path.exists(_INDEX_DIR):
        shutil.rmtree(_INDEX_DIR)

    logger.info("Tantivy BM25 索引已清空，重新构建中...")
    _ensure_loaded()
