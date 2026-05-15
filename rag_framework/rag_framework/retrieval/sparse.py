"""
Sparse Store — Tantivy BM25 全文检索

商业级 BM25，磁盘持久化索引，支持增量写入。
"""
from __future__ import annotations

import logging
import os
import pathlib
import shutil
import threading
import warnings

import chromadb
import tantivy

from rag_framework.core.factories import register_vector_store

logger = logging.getLogger("bm25_store")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import jieba
    jieba.initialize()


class BM25Store:
    """Tantivy BM25 稀疏检索封装。"""

    def __init__(
        self,
        index_dir: str,
        chroma_path: str,
        heap_size: int = 200 * 1024 * 1024,
        batch_size: int = 10_000,
    ) -> None:
        self._index_dir = index_dir
        self._chroma_path = chroma_path
        self._heap_size = heap_size
        self._batch_size = batch_size
        self._index: tantivy.Index | None = None
        self._searcher: tantivy.Searcher | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _make_schema() -> tantivy.Schema:
        sb = tantivy.SchemaBuilder()
        sb.add_text_field("doc_id", stored=True, tokenizer_name="raw")
        sb.add_text_field("body", stored=True, tokenizer_name="whitespace")
        sb.add_text_field("raw_text", stored=True, tokenizer_name="raw")
        return sb.build()

    @staticmethod
    def _tokenize(text: str) -> str:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tokens = jieba.cut(text, cut_all=False)
        import re
        return " ".join(t for t in tokens if t.strip() and re.match(r"^\w+$", t))

    def _ensure_loaded(self) -> bool:
        if self._searcher is not None:
            return True

        with self._lock:
            if self._searcher is not None:
                return True

            os.makedirs(self._index_dir, exist_ok=True)
            schema = self._make_schema()
            idx = tantivy.Index(schema=schema, path=self._index_dir)

            if idx.searcher().num_docs == 0:
                if not self._build_from_chroma(idx):
                    return False
                idx.reload()

            self._index = idx
            self._searcher = idx.searcher()
            logger.info(f"Tantivy BM25 索引就绪：{self._searcher.num_docs} 个文档")
            return True

    def _build_from_chroma(self, idx: tantivy.Index) -> bool:
        """从 ChromaDB android_parent 分批拉取构建索引。"""
        client = chromadb.PersistentClient(path=self._chroma_path)
        try:
            col = client.get_collection("android_parent")
        except Exception as e:
            logger.warning(f"android_parent 未就绪，BM25 不可用: {e}")
            return False

        total = col.count()
        logger.info(f"开始构建 Tantivy BM25 索引，共 {total} 个文档")

        writer = idx.writer(heap_size=self._heap_size)
        indexed = 0
        offset = 0

        while offset < total:
            batch = col.get(limit=self._batch_size, offset=offset, include=["documents"])
            for doc_id, text in zip(batch["ids"], batch["documents"]):
                writer.add_document(tantivy.Document(
                    doc_id=doc_id,
                    body=self._tokenize(text),
                    raw_text=text,
                ))
            indexed += len(batch["ids"])
            offset += self._batch_size
            logger.info(f"  索引进度 {indexed} / {total}")

        writer.commit()
        writer.wait_merging_threads()
        logger.info(f"Tantivy BM25 索引构建完成：{indexed} 个文档")
        return True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        BM25 全文检索。

        Returns:
            [(parent_id, text, bm25_score), ...] 按分值降序
        """
        if not self._ensure_loaded():
            return []

        tokenized = self._tokenize(query)
        q = self._index.parse_query(tokenized, ["body"])
        hits = self._searcher.search(q, limit=top_k)

        results = [
            (doc["doc_id"][0], doc["raw_text"][0], float(score))
            for score, addr in hits.hits
            if (doc := self._searcher.doc(addr)) and score > 0
        ]
        logger.debug(
            f"BM25 检索: query={query[:30]!r}, 命中={len(results)}, "
            f"top_score={results[0][2] if results else 0.0:.3f}"
        )
        return results

    def add_documents(self, docs: list[tuple[str, str]]) -> None:
        """增量写入文档。"""
        if not self._ensure_loaded():
            return

        writer = self._index.writer(heap_size=self._heap_size)
        for doc_id, text in docs:
            writer.add_document(tantivy.Document(
                doc_id=doc_id,
                body=self._tokenize(text),
                raw_text=text,
            ))
        writer.commit()
        writer.wait_merging_threads()

        self._index.reload()
        self._searcher = self._index.searcher()
        logger.info(f"增量写入 {len(docs)} 个文档，当前总量 {self._searcher.num_docs}")

    def reload(self) -> None:
        """删除磁盘索引并从 ChromaDB 全量重建。"""
        with self._lock:
            self._index = None
            self._searcher = None

        if os.path.exists(self._index_dir):
            shutil.rmtree(self._index_dir)

        logger.info("Tantivy BM25 索引已清空，重新构建中...")
        self._ensure_loaded()


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_bm25_store(
    index_dir: str,
    chroma_path: str,
) -> BM25Store:
    return BM25Store(index_dir=index_dir, chroma_path=chroma_path)


register_vector_store("bm25", _create_bm25_store)
