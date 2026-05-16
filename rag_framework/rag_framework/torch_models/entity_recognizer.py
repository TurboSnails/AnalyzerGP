"""
命名实体识别器 — 基于 transformers pipeline 的轻量实现。

默认模型：dslim/bert-base-NER（英文 CoNLL-2003）
中文场景可替换为：shibing624/macbert4cner-base-chinese

客服场景关注实体：
  ORDER_ID, PRODUCT_NAME, PHONE_NUMBER, EMAIL, DATE, PRICE, PERSON
"""
from __future__ import annotations

import re
import time
from typing import Any

from rag_framework.core.exceptions import ModelLoadError
from rag_framework.core.logger import get_logger
from rag_framework.core.factories import register_torch_model
from rag_framework.torch_models.base import TorchTaskModel, TaskPrediction

_logger = get_logger("rag.torch.ner")

# 客服场景实体标签映射（从 BIO 格式简化）
_ENTITY_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "MISC": "MISC",
    "PRODUCT": "PRODUCT_NAME",
    "PHONE": "PHONE_NUMBER",
    "TEL": "PHONE_NUMBER",
    "EMAIL": "EMAIL",
    "DATE": "DATE",
    "MONEY": "PRICE",
    "PRICE": "PRICE",
    "ORDER": "ORDER_ID",
}


class EntityRecognizer(TorchTaskModel):
    """命名实体识别模型。"""

    @property
    def task_name(self) -> str:
        return "ner"

    def _load_model(self) -> None:
        try:
            from transformers import pipeline
            resolved_device = self._resolve_device(self._device)
            model = self._model_path or "dslim/bert-base-NER"
            self._pipeline = pipeline(
                "ner",
                model=model,
                device=resolved_device,
                aggregation_strategy="simple",
            )
            _logger.info(f"NER 模型加载完成: {model}, device={resolved_device}")
        except Exception as exc:
            raise ModelLoadError(f"NER 模型加载失败: {exc}") from exc

    def _run_inference(self, text: str) -> TaskPrediction:
        if self._pipeline is None:
            raise RuntimeError("模型未加载")

        start = time.monotonic()
        entities = self._pipeline(text)
        latency_ms = (time.monotonic() - start) * 1000

        # 合并并标准化实体
        normalized = []
        for ent in entities:
            label = ent.get("entity_group", ent.get("entity", "")).upper()
            mapped = _ENTITY_MAP.get(label, label)
            normalized.append({
                "text": ent.get("word", ""),
                "label": mapped,
                "score": ent.get("score", 0.0),
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
            })

        # 简单启发式：订单号正则补充（若 NER 未识别）
        order_pattern = self._extra_kwargs.get("order_pattern", r"[A-Z]{2,4}-?\d{6,12}")
        for m in re.finditer(order_pattern, text):
            normalized.append({
                "text": m.group(),
                "label": "ORDER_ID",
                "score": 1.0,
                "start": m.start(),
                "end": m.end(),
            })

        # 主标签取出现频次最高的实体类型
        if normalized:
            from collections import Counter
            top_label = Counter(e["label"] for e in normalized).most_common(1)[0][0]
        else:
            top_label = "none"

        return TaskPrediction(
            task_name=self.task_name,
            text=text,
            label=top_label,
            score=1.0 if normalized else 0.0,
            details={"entities": normalized},
            model_name=self._model_path or "dslim/bert-base-NER",
            latency_ms=latency_ms,
        )


# ── 自注册 ────────────────────────────────────────────────────────────────────
register_torch_model("entity_recognizer", EntityRecognizer)
