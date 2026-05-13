"""
Query Rewriter 抽象基类

支持多路查询扩写，每条输出带路由元数据。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from rag_framework.domain.base import QueryRoute


class QueryRewriter(ABC):
    """查询改写器抽象基类。"""

    @abstractmethod
    def rewrite(self, query: str, history: list[dict]) -> list[QueryRoute]:
        """
        将用户查询改写为多路检索 query。

        Returns:
            QueryRoute 列表，第一条为原始 query
        """
        ...
