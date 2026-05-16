"""
PyTorch 轻量任务模型抽象基类。

设计原则：
  1. 惰性加载：模型权重在首次 predict() 或 warmup() 时加载
  2. 异步接口：predict() 为 async，内部通过 asyncio.to_thread 卸载同步推理
  3. 独立生命周期：不纳入工厂注册表（与 RAG 组件解耦），通过 torch_model_registry 管理
  4. 批处理支持：predict_batch() 默认实现为串行 predict()，子类可覆盖为真实批处理
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag_framework.core.lifecycle import Warmupable


@dataclass
class TaskPrediction:
    """任务模型推理结果。"""
    task_name: str = ""
    text: str = ""
    label: str = ""              # 主标签（如意图类别、情感极性）
    score: float = 0.0           # 置信度
    details: dict[str, Any] = field(default_factory=dict)  # 详细结果（如 NER 实体列表）
    model_name: str = ""
    latency_ms: float = 0.0


class TorchTaskModel(ABC, Warmupable):
    """PyTorch 轻量任务模型抽象基类。"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._batch_size = batch_size
        self._extra_kwargs = kwargs
        self._pipeline: Any | None = None
        self._model_loaded = False

    # ------------------------------------------------------------------
    # 抽象属性 / 方法
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def task_name(self) -> str:
        """任务标识，如 intent_classification / sentiment_analysis / ner。"""
        ...

    @abstractmethod
    def _load_model(self) -> None:
        """加载模型权重（内部使用，由子类实现）。"""
        ...

    @abstractmethod
    def _run_inference(self, text: str) -> TaskPrediction:
        """执行同步单条推理（内部使用，由子类实现）。"""
        ...

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def load(self) -> None:
        """显式加载模型（线程安全，幂等）。"""
        if self._model_loaded:
            return
        self._load_model()
        self._model_loaded = True

    async def predict(self, text: str) -> TaskPrediction:
        """异步推理单条文本。"""
        if not self._model_loaded:
            await asyncio.to_thread(self.load)
        return await asyncio.to_thread(self._run_inference, text)

    async def predict_batch(self, texts: list[str]) -> list[TaskPrediction]:
        """异步推理批量文本（默认串行，子类可覆盖为真实批处理）。"""
        results: list[TaskPrediction] = []
        for t in texts:
            results.append(await self.predict(t))
        return results

    def warmup(self) -> None:
        """预热模型，避免首请求延迟。"""
        self.load()

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备字符串，auto 时优先 CUDA > MPS > CPU。"""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"
