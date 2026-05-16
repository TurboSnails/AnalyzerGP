"""
VectorIndexer — 索引构建编排器

将文档完整入库：分块 → 嵌入 → 写入 DenseStore（parent/hyde）+ BM25。

典型用法：
    indexer = VectorIndexer(domain, embedder, dense_store, sparse_store, llm)
    stats = indexer.index_files(["docs/android.txt", "docs/jetpack.txt"])
    print(stats)
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag_framework.core.logger import get_logger
from rag_framework.domain.base import DomainPlugin
from rag_framework.embedding.base import Embedder
from rag_framework.indexing.chunker import chunk_file, chunk_text
from rag_framework.llm.base import LLMClient
from rag_framework.retrieval.dense import DenseStore
from rag_framework.retrieval.sparse import BM25Store

_logger = get_logger("rag.indexing.indexer")


@dataclass
class IndexConfig:
    """索引构建参数。"""
    chunk_size: int = 512
    overlap: int = 64
    child_chunk_size: int = 128
    child_overlap: int = 25
    hyde_batch_size: int = 4
    enable_child: bool = True
    enable_hyde: bool = True
    enable_bm25: bool = True


@dataclass
class IndexStats:
    """构建结果统计。"""
    total_files: int = 0
    total_chunks: int = 0
    hyde_generated: int = 0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"IndexStats(files={self.total_files}, chunks={self.total_chunks}, "
            f"hyde={self.hyde_generated}, errors={len(self.errors)})"
        )


class VectorIndexer:
    """
    向量索引构建器。

    支持从文件列表或文本列表构建 parent / hyde 两层 DenseStore
    以及 BM25 稀疏索引。
    """

    def __init__(
        self,
        domain: DomainPlugin,
        embedder: Embedder,
        dense_store: DenseStore,
        sparse_store: BM25Store | None = None,
        llm: LLMClient | None = None,
        config: IndexConfig | None = None,
    ) -> None:
        self._domain = domain
        self._embedder = embedder
        self._dense = dense_store
        self._sparse = sparse_store
        self._llm = llm
        self._cfg = config or IndexConfig()

    # ─── 公开 API ──────────────────────────────────────────────────────────────

    def index_files(
        self,
        file_paths: list[str | Path],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """从文件列表构建索引。"""
        stats = IndexStats(total_files=len(file_paths))
        chunks, sources = [], []

        for path in file_paths:
            path = Path(path)
            try:
                fc = chunk_file(str(path), self._cfg.chunk_size, self._cfg.overlap)
                chunks.extend(fc)
                sources.extend([path.name] * len(fc))
                _logger.info(f"分块: {path.name} → {len(fc)} 个")
            except Exception as e:
                msg = f"{path.name}: {e}"
                stats.errors.append(msg)
                _logger.warning(f"读取失败 {msg}")

        return self._do_index(chunks, sources, stats, on_progress)

    def index_texts(
        self,
        texts: list[str],
        sources: list[str] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IndexStats:
        """从文本列表构建索引（适用于已在内存中的内容）。"""
        stats = IndexStats(total_files=len(texts))
        chunks, chunk_sources = [], []

        for i, text in enumerate(texts):
            src = sources[i] if sources else f"text_{i}"
            fc = chunk_text(text, self._cfg.chunk_size, self._cfg.overlap)
            chunks.extend(fc)
            chunk_sources.extend([src] * len(fc))

        return self._do_index(chunks, chunk_sources, stats, on_progress)

    # ─── 内部 ──────────────────────────────────────────────────────────────────

    def _do_index(
        self,
        chunks: list[str],
        sources: list[str],
        stats: IndexStats,
        on_progress: Callable[[int, int], None] | None,
    ) -> IndexStats:
        stats.total_chunks = len(chunks)
        if not chunks:
            _logger.warning("没有可索引的 chunk")
            return stats

        names = self._domain.get_collection_names()
        ids = [str(uuid.uuid4()) for _ in chunks]
        domain_name = getattr(self._domain, "name", "")
        metadatas = [{"source": s, "domain": domain_name} for s in sources]

        # Dense (parent collection)
        try:
            col = self._dense.get_or_create_collection(names.parent)
            embeddings = self._embedder.encode(chunks)
            col.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
            _logger.info(f"DenseStore[parent] 写入 {len(chunks)} 条 → {names.parent}")
        except Exception as e:
            stats.errors.append(f"dense.parent: {e}")
            _logger.error(f"DenseStore[parent] 写入失败: {e}")

        # Child chunks (fine-grained, linked to parent via parent_id metadata)
        if self._cfg.enable_child:
            child_texts, child_ids, child_metas = [], [], []
            for parent_id, parent_text in zip(ids, chunks):
                sub_chunks = chunk_text(
                    parent_text, self._cfg.child_chunk_size, self._cfg.child_overlap
                )
                for c_idx, sub in enumerate(sub_chunks):
                    child_ids.append(f"{parent_id}_c{c_idx}")
                    child_texts.append(sub)
                    child_metas.append({"parent_id": parent_id, "domain": domain_name})
            if child_texts:
                try:
                    col = self._dense.get_or_create_collection(names.child)
                    child_emb = self._embedder.encode(child_texts)
                    col.add(
                        ids=child_ids,
                        documents=child_texts,
                        metadatas=child_metas,
                        embeddings=child_emb,
                    )
                    _logger.info(
                        f"DenseStore[child] 写入 {len(child_texts)} 条 → {names.child}"
                    )
                except Exception as e:
                    stats.errors.append(f"dense.child: {e}")
                    _logger.error(f"DenseStore[child] 写入失败: {e}")

        # BM25
        if self._cfg.enable_bm25 and self._sparse is not None:
            try:
                self._sparse.add_documents(
                    list(zip(ids, chunks)), domain=domain_name
                )
                _logger.info(f"BM25 写入 {len(chunks)} 条")
            except Exception as e:
                stats.errors.append(f"bm25: {e}")
                _logger.error(f"BM25 写入失败: {e}")

        # HyDE collection
        if self._cfg.enable_hyde and self._llm is not None:
            self._build_hyde(chunks, ids, sources, names.hyde, stats)

        if on_progress:
            on_progress(stats.total_chunks, stats.total_chunks)

        _logger.info(str(stats))
        return stats

    def _build_hyde(
        self,
        chunks: list[str],
        ids: list[str],
        sources: list[str],
        hyde_collection: str,
        stats: IndexStats,
    ) -> None:
        from rag_framework.indexing.hyde import generate_hyde_questions

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                questions = loop.run_until_complete(
                    generate_hyde_questions(
                        chunks,
                        self._domain,
                        self._llm,  # type: ignore[arg-type]
                        self._cfg.hyde_batch_size,
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            stats.errors.append(f"hyde.generate: {e}")
            _logger.error(f"HyDE 生成失败: {e}")
            return

        valid = [(q, i, s) for q, i, s in zip(questions, ids, sources) if q]
        stats.hyde_generated = len(valid)
        if not valid:
            return

        hyde_texts, hyde_ids, hyde_srcs = zip(*valid)
        try:
            col = self._dense.get_or_create_collection(hyde_collection)
            embeddings = self._embedder.encode(list(hyde_texts))
            hyde_domain = getattr(self._domain, "name", "")
            col.add(
                ids=list(hyde_ids),
                documents=list(hyde_texts),
                metadatas=[{"source": s, "domain": hyde_domain} for s in hyde_srcs],
                embeddings=embeddings,
            )
            _logger.info(f"DenseStore[hyde] 写入 {len(valid)} 条 → {hyde_collection}")
        except Exception as e:
            stats.errors.append(f"hyde.dense: {e}")
            _logger.error(f"DenseStore[hyde] 写入失败: {e}")
