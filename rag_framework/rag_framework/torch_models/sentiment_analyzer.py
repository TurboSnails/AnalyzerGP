"""
情感分析器 — 基于 transformers pipeline 的轻量实现。

默认模型：distilbert-base-uncased-finetuned-sst-2-english（英文）
中文场景可替换为：uer/roberta-base-finetuned-jd-binary-chinese

输出标签映射：
  positive  -> 满意
  negative  -> 愤怒/不满
  neutral   -> 中性
"""
from __future__ import annotations

import time
from typing import Any

from rag_framework.core.exceptions import ModelLoadError
from rag_framework.core.logger import get_logger
from rag_framework.core.factories import register_torch_model
from rag_framework.torch_models.base import TorchTaskModel, TaskPrediction

_logger = get_logger("rag.torch.sentiment")

_DEFAULT_SENTIMENT_LABELS = ["positive", "negative", "neutral"]


class SentimentAnalyzer(TorchTaskModel):
    """情感分析模型。"""

    @property
    def task_name(self) -> str:
        return "sentiment_analysis"

    def _load_model(self) -> None:
        try:
            from transformers import pipeline
            resolved_device = self._resolve_device(self._device)
            # 默认使用 distilbert SST-2；中文场景覆盖 model_path
            model = self._model_path or "distilbert-base-uncased-finetuned-sst-2-english"
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                device=resolved_device,
            )
            _logger.info(f"情感分析模型加载完成: {model}, device={resolved_device}")
        except Exception as exc:
            raise ModelLoadError(f"情感分析模型加载失败: {exc}") from exc

    def _run_inference(self, text: str) -> TaskPrediction:
        if self._pipeline is None:
            raise RuntimeError("模型未加载")

        start = time.monotonic()
        result = self._pipeline(text)[0]  # [{'label': 'POSITIVE', 'score': 0.99}]
        latency_ms = (time.monotonic() - start) * 1000

        raw_label = result["label"].lower()
        # 统一映射
        if raw_label in ("positive", "pos", "满意"):
            label = "positive"
        elif raw_label in ("negative", "neg", "不满", "愤怒"):
            label = "negative"
        else:
            label = "neutral"

        return TaskPrediction(
            task_name=self.task_name,
            text=text,
            label=label,
            score=result["score"],
            details={"raw_label": result["label"]},
            model_name=self._model_path or "distilbert-base-uncased-finetuned-sst-2-english",
            latency_ms=latency_ms,
        )


# ── 自注册 ────────────────────────────────────────────────────────────────────
register_torch_model("sentiment_analyzer", SentimentAnalyzer)
