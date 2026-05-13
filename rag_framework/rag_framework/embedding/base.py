"""
Embedding 抽象基类

支持文本 → 向量的编码，框架层不绑定具体模型。
"""
from abc import ABC, abstractmethod
from typing import Any


class Embedder(ABC):
    """文本编码器抽象基类。"""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """向量维度。"""
        ...

    @abstractmethod
    def encode(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """
        将文本列表编码为向量列表。

        Args:
            texts: 待编码文本列表
            batch_size: 分批编码大小，None 表示不分批

        Returns:
            向量列表，每个向量是 float 列表
        """
        ...

    @abstractmethod
    def _ensure_model(self) -> None:
        """懒加载模型（内部使用）。"""
        ...
