"""
Retriever & VectorStore 抽象基类

支持单路/多路检索，返回带元数据的候选文档。
retrieve() 为 async 接口，CPU/IO 密集型子步骤通过 asyncio.to_thread 卸载。

VectorStore 抽象 ChromaDB/Milvus/Pinecone 等向量存储的通用操作。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_framework.domain.base import QueryRoute


@dataclass
class RetrievedDoc:
    """检索到的文档。"""
    id: str
    text: str
    score: float = 0.0
    source: str = ""          # dense | hyde | bm25 | fusion
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果。"""
    docs: list[RetrievedDoc]
    query: str = ""
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class Retriever(ABC):
    """检索器抽象基类。"""

    @abstractmethod
    async def retrieve(
        self,
        query: str | QueryRoute | list[QueryRoute],
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        执行检索（async）。

        Args:
            query: 原始查询或 QueryRoute（支持多路扩写）
            top_k: 返回文档数

        Returns:
            RetrievalResult
        """
        ...


class VectorStore(ABC):
    """
    向量存储抽象基类。

    统一 ChromaDB、Milvus、Pinecone 等向量数据库的通用操作接口。
    支持多 collection 查询与批量写入。
    """

    @abstractmethod
    def get_collection(self, name: str) -> Any | None:
        """
        获取 collection 句柄或元数据。

        Returns:
            collection 对象，或 None 表示不存在。
        """
        ...

    @abstractmethod
    def query(
        self,
        query: str,
        collection_name: str,
        n_results: int = 10,
        **filters: Any,
    ) -> tuple[list[str], list[float], list[dict]]:
        """
        向量相似度查询。

        Returns:
            (ids, distances, metadatas)
        """
        ...

    @abstractmethod
    def fetch_parents(self, parent_ids: list[str], collection_name: str) -> dict[str, str]:
        """
        批量拉取 parent 文档全文。

        Returns:
            {parent_id: text, ...}
        """
        ...

    @abstractmethod
    def add_batch(
        self,
        collection_name: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """批量添加文档（含自动编码 embedding）。"""
        ...
