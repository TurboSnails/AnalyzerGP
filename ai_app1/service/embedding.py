"""
Embedding：本地 BGE-M3，显式 encode 后交给 Chroma（add(embeddings=...) / query(query_embeddings=...)），
不把编码交给 Chroma 内置 embedding_function，便于缓存、换模型、插 pipeline。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import cast

from sentence_transformers import SentenceTransformer

from ai_app1.core.config import BGE_M3_PATH

logger = logging.getLogger("embedding")


class BgeM3EmbeddingService:
    """封装 SentenceTransformer，对外只暴露 encode。"""

    def __init__(self, model_path: str | None = None) -> None:
        self._path = model_path or BGE_M3_PATH
        self._model: SentenceTransformer | None = None
        # HF fast tokenizer (Rust RefCell) 不允许并发调用；用锁序列化 encode
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        path = self._path
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"BGE-M3 模型目录不存在: {path}\n"
                "请先下载模型，例如:\n"
                "  uv run python ai_app1/test/download_bge_m3.py --preset bge-m3"
            )
        logger.info(f"正在加载 BGE-M3 模型: {path}")
        self._model = SentenceTransformer(path)
        logger.info("BGE-M3 模型加载完成")

    def encode(self, texts: list[str], batch_size: int | None = 32) -> list[list[float]]:
        """
        文本 → 向量（已 L2 normalize，与原先 Chroma 内嵌编码一致）。

        batch_size：长列表分批 encode，降低峰值显存；None 表示不分批。
        """
        self._ensure_model()
        if not texts:
            return []
        with self._lock:
            if batch_size is None or len(texts) <= batch_size:
                embeddings = self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                return cast(list[list[float]], embeddings.tolist())

            out: list[list[float]] = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                embeddings = self._model.encode(
                    chunk,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                out.extend(embeddings.tolist())
            return out


_embedding_service: BgeM3EmbeddingService | None = None


def get_embedding_service(model_path: str | None = None) -> BgeM3EmbeddingService:
    """全局 Embedding 服务（惰性加载模型）。"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = BgeM3EmbeddingService(model_path=model_path)
    return _embedding_service


def reset_embedding_service() -> None:
    """释放模型引用（测试或热替换模型时）。"""
    global _embedding_service
    if _embedding_service is not None:
        _embedding_service._model = None
        _embedding_service = None
    logger.info("BgeM3EmbeddingService 已重置")
