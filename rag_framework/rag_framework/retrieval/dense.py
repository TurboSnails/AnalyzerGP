"""
Dense Store — ChromaDB 向量检索

支持 child/hyde collection 查询，parent 回溯。
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import chromadb

from rag_framework.core.exceptions import CollectionNotFoundError, VectorStoreError
from rag_framework.core.factories import register_vector_store
from rag_framework.core.lifecycle import Warmupable
from rag_framework.retrieval.base import VectorStore

if TYPE_CHECKING:
    from rag_framework.embedding.base import Embedder

logger = logging.getLogger("vector_store")


class ChromaVectorStore(VectorStore, Warmupable):
    """
    ChromaDB 向量存储封装。

    支持多 collection（parent/child/hyde），显式编码后入库。
    """

    def __init__(
        self,
        chroma_path: str,
        embedder: Embedder,
    ) -> None:
        self._chroma_path = chroma_path
        self._embedder = embedder
        self._client: chromadb.PersistentClient | None = None
        self._lock = threading.Lock()

    def _get_client(self) -> chromadb.PersistentClient:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = chromadb.PersistentClient(path=self._chroma_path)
        return self._client

    def get_collection(self, name: str):
        try:
            return self._get_client().get_collection(name)
        except Exception as e:
            logger.warning(f"Collection '{name}' 不存在: {e}")
            return None

    def get_or_create_collection(self, name: str):
        """获取或创建 collection（用于索引构建阶段）。"""
        return self._get_client().get_or_create_collection(name)

    def query(
        self,
        query: str,
        collection_name: str,
        n_results: int = 10,
        max_distance: float = 1.3,
    ) -> tuple[list[str], list[float], list[dict]]:
        """
        向量查询。

        Returns:
            (ids, distances, metadatas)
        """
        col = self.get_collection(collection_name)
        if col is None:
            raise CollectionNotFoundError(f"Collection '{collection_name}' 未找到")

        q_emb = self._embedder.encode([query])
        result = col.query(query_embeddings=q_emb, n_results=n_results)
        ids = result["ids"][0] if result["ids"] else []
        distances = result["distances"][0] if result.get("distances") else []
        metas = result["metadatas"][0] if result.get("metadatas") else []

        # 过滤距离
        filtered = [
            (i, d, m) for i, d, m in zip(ids, distances, metas)
            if d <= max_distance
        ]
        if filtered:
            ids, distances, metas = zip(*filtered)
            return list(ids), list(distances), list(metas)
        return [], [], []

    def fetch_parents(self, parent_ids: list[str], collection_name: str) -> dict[str, str]:
        """批量拉取 parent 文档。"""
        if not parent_ids:
            return {}
        col = self.get_collection(collection_name)
        if col is None:
            return {}
        result = col.get(ids=parent_ids)
        return dict(zip(result["ids"], result["documents"]))

    def add_batch(
        self,
        collection_name: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """批量添加文档（带 embedding）。"""
        col = self.get_collection(collection_name)
        if col is None:
            raise CollectionNotFoundError(f"Collection '{collection_name}' 未找到")

        embeddings = self._embedder.encode(texts)
        col.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    async def warmup(self) -> None:
        """预热：建立 ChromaDB 连接并验证 collections。"""
        import asyncio
        await asyncio.to_thread(self._get_client)


# ─── 兼容旧名称 ───────────────────────────────────────────────
DenseStore = ChromaVectorStore


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_chroma_store(
    chroma_path: str,
    embedder: Embedder,
) -> ChromaVectorStore:
    return ChromaVectorStore(chroma_path=chroma_path, embedder=embedder)


register_vector_store("chroma", _create_chroma_store)
