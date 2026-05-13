"""
Reranker 抽象基类

对候选文档按与查询的相关性进行重排序。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RankedDoc:
    """精排后的文档。"""
    id: str
    text: str
    score: float = 0.0
    ce_score: float = 0.0
    rrf_score: float = 0.0
    vector_rank: int = 999
    bm25_rank: int = 999


class Reranker(ABC):
    """精排器抽象基类。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RankedDoc],
        top_k: int = 5,
    ) -> list[RankedDoc]:
        """
        对候选文档进行精排。

        Args:
            query: 原始查询
            candidates: 候选文档列表（已含 RRF 分数等元数据）
            top_k: 返回前 K 个

        Returns:
            按相关性降序排列的文档列表
        """
        ...
