"""
意图分类器 — 基于 transformers pipeline 的轻量实现。

支持两种模式：
  1. 文本分类（text-classification）：单标签/多标签分类
  2. 零样本分类（zero-shot-classification）：无需训练，通过候选标签推理

默认候选标签（客服场景）：
  technical_question, complaint, escalation_request, chitchat, billing, general_inquiry
"""
from __future__ import annotations

import time
from typing import Any

from rag_framework.core.exceptions import ModelLoadError
from rag_framework.core.logger import get_logger
from rag_framework.core.factories import register_torch_model
from rag_framework.torch_models.base import TorchTaskModel, TaskPrediction

_logger = get_logger("rag.torch.intent")

# 默认客服意图候选标签
_DEFAULT_INTENT_LABELS = [
    "technical_question",   # 技术问题
    "complaint",            # 投诉
    "escalation_request",   # 要求转人工
    "chitchat",             # 闲聊
    "billing",              # 账单/支付
    "general_inquiry",      # 一般咨询
]


class IntentClassifier(TorchTaskModel):
    """意图分类模型。"""

    @property
    def task_name(self) -> str:
        return "intent_classification"

    def _load_model(self) -> None:
        try:
            from transformers import pipeline
            resolved_device = self._resolve_device(self._device)
            # 若 model_path 为空，使用默认零 shot 模型
            model = self._model_path or "facebook/bart-large-mnli"
            self._pipeline = pipeline(
                "zero-shot-classification",
                model=model,
                device=resolved_device,
            )
            _logger.info(f"意图分类模型加载完成: {model}, device={resolved_device}")
        except Exception as exc:
            raise ModelLoadError(f"意图分类模型加载失败: {exc}") from exc

    def _run_inference(self, text: str) -> TaskPrediction:
        if self._pipeline is None:
            raise RuntimeError("模型未加载")

        start = time.monotonic()
        labels = self._extra_kwargs.get("labels", _DEFAULT_INTENT_LABELS)
        result = self._pipeline(text, candidate_labels=labels, multi_label=False)
        latency_ms = (time.monotonic() - start) * 1000

        # result 格式: {'sequence': text, 'labels': [...], 'scores': [...]}
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        return TaskPrediction(
            task_name=self.task_name,
            text=text,
            label=top_label,
            score=top_score,
            details={
                "all_labels": result["labels"],
                "all_scores": result["scores"],
            },
            model_name=self._model_path or "facebook/bart-large-mnli",
            latency_ms=latency_ms,
        )


# ── 自注册 ────────────────────────────────────────────────────────────────────
register_torch_model("intent_classifier", IntentClassifier)
