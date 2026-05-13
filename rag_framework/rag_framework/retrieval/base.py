"""
Retriever 抽象基类

支持单路/多路检索，返回带元数据的候选文档。
retrieve() 为 async 接口，CPU/IO 密集型子步骤通过 asyncio.to_thread 卸载。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
