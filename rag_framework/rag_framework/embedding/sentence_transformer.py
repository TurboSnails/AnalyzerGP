"""
SentenceTransformer Embedder 实现

支持 BGE-M3 等模型，线程安全（带锁）。
"""
from __future__ import annotations

import os
import threading
from typing import cast

import torch
from sentence_transformers import SentenceTransformer

from rag_framework.core.exceptions import ModelLoadError, ModelNotFoundError
from rag_framework.core.factories import register_embedder
from rag_framework.core.lifecycle import Warmupable
from rag_framework.core.logger import embed_logger
from rag_framework.embedding.base import Embedder


def _resolve_device(requested: str) -> str:
    """将 'auto' 解析为实际可用设备。"""
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class STEmbedder(Embedder, Warmupable):
    """基于 SentenceTransformer 的 Embedder。"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        normalize: bool = True,
    ) -> None:
        self._path = model_path
        self._device = device
        self._normalize = normalize
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not os.path.isdir(self._path):
            raise ModelNotFoundError(
                f"模型目录不存在: {self._path}\n"
                "请先下载模型，例如:\n"
                "  uv run python scripts/download_bge_m3.py --preset bge-m3"
            )
        embed_logger.info(f"正在加载 Embedding 模型: {self._path}")
        try:
            resolved_device = _resolve_device(self._device)
            self._model = SentenceTransformer(self._path, device=resolved_device)
        except Exception as e:
            raise ModelLoadError(f"加载模型失败: {e}") from e
        embed_logger.info("Embedding 模型加载完成")

    @property
    def embedding_dim(self) -> int:
        self._ensure_model()
        return self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = 32,
    ) -> list[list[float]]:
        self._ensure_model()
        if not texts:
            return []

        with self._lock:
            _batch = batch_size or len(texts)
            if len(texts) <= _batch:
                embeddings = self._model.encode(  # type: ignore[union-attr]
                    texts,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                return cast(list[list[float]], embeddings.tolist())

            out: list[list[float]] = []
            for i in range(0, len(texts), _batch):
                chunk = texts[i : i + _batch]
                embeddings = self._model.encode(  # type: ignore[union-attr]
                    chunk,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                out.extend(embeddings.tolist())
            return out

    async def warmup(self) -> None:
        """异步预热：将模型加载卸载到线程池。"""
        import asyncio
        await asyncio.to_thread(self._ensure_model)


# ─── 工厂函数与自注册 ──────────────────────────────────────────

def _create_st_embedder(
    model_path: str | None = None,
    device: str = "auto",
    normalize: bool = True,
) -> STEmbedder:
    """Factory：处理默认值后创建 STEmbedder 实例。"""
    # 延迟导入避免循环依赖；Phase 2 会将路径解析完全移出 config
    from rag_framework.core.config import _resolve_bge_m3_path
    path = model_path or _resolve_bge_m3_path()
    return STEmbedder(model_path=path, device=device, normalize=normalize)


register_embedder("sentence_transformer", _create_st_embedder)
register_embedder("st", _create_st_embedder)
