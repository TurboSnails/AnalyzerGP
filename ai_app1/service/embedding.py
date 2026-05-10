"""
Embedding 模块：封装本地 BGE-M3 模型，供 ChromaDB 使用。

替代 ChromaDB 默认的内置 embedding，使用 models/bge-m3 本地模型，
保证索引（init_vector_db）和查询（vector_store）使用完全一致的向量表示。
"""

from __future__ import annotations

import logging
import os
from typing import cast

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

from ai_app1.core.config import BGE_M3_PATH

logger = logging.getLogger("embedding")


class BgeM3EmbeddingFunction(EmbeddingFunction):
    """
    基于本地 BGE-M3 的 ChromaDB EmbeddingFunction。

    单例模型：首次实例化时加载 SentenceTransformer，后续复用，
    避免重复初始化带来的内存和启动开销。
    """

    _instance: BgeM3EmbeddingFunction | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls, *args, **kwargs) -> BgeM3EmbeddingFunction:  # type: ignore[no-untyped-def]
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_path: str | None = None) -> None:
        """
        初始化/复用 BGE-M3 模型。

        Args:
            model_path: 本地模型目录路径，默认读取 core.config.BGE_M3_PATH。
        """
        if self._model is not None:
            return

        path = model_path or BGE_M3_PATH
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"BGE-M3 模型目录不存在: {path}\n"
                "请先下载模型，例如:\n"
                "  uv run python ai_app1/test/download_bge_m3.py --preset bge-m3"
            )

        logger.info(f"正在加载 BGE-M3 模型: {path}")
        self._model = SentenceTransformer(path)
        logger.info("BGE-M3 模型加载完成")

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        """
        对输入文本列表进行向量化。

        Args:
            input: ChromaDB 传入的文本列表（Documents）。

        Returns:
            Embeddings: 二维浮点列表，每行对应一个文本的向量。
        """
        if self._model is None:
            raise RuntimeError("BgeM3EmbeddingFunction 未正确初始化")

        embeddings = self._model.encode(
            list(input),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return cast(Embeddings, embeddings.tolist())

    @classmethod
    def reset(cls) -> None:
        """强制释放模型实例（主要用于测试或热更新场景）。"""
        cls._instance = None
        cls._model = None
        logger.info("BgeM3EmbeddingFunction 已重置")


# ─── 便捷入口 ─────────────────────────────────────────────────────────────────

_embedding_fn: BgeM3EmbeddingFunction | None = None


def get_embedding_function() -> EmbeddingFunction:
    """获取全局 BGE-M3 EmbeddingFunction 实例（惰性初始化）。"""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = BgeM3EmbeddingFunction()
    return _embedding_fn
