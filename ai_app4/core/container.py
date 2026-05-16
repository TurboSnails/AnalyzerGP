"""
ai_app4 扩展容器。

在 RAGContainer 基础上，暴露 LlamaIndex Retriever 和 PyTorch 模型实例，
供 ai_app4 的 LangGraph 节点直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag_framework.container import RAGContainer

from ai_app4.core.config import CS4Settings


@dataclass(frozen=True, slots=True)
class CS4Container(RAGContainer):
    """
    ai_app4 扩展依赖注入容器。

    新增：
      - get_torch_model(task_name) — 按任务名获取 PyTorch 模型
      - get_llamaindex_retriever() — 获取 LlamaIndex 检索器
      - is_escalation_needed(intent, sentiment) — 转人工判断
    """

    @classmethod
    def from_settings(cls, settings: CS4Settings | None = None) -> "CS4Container":
        """从 CS4Settings 构建扩展容器。"""
        settings = settings or CS4Settings()
        # 复用父类构建逻辑，得到基础容器实例
        base = RAGContainer.from_settings(settings)
        # 通过 __class__ 构造子类实例（绕过 frozen 限制）
        # 注意：base 是 RAGContainer，我们需要把它升级为 CS4Container
        # 方案：直接调用 object.__new__ + 手动设置字段
        instance = object.__new__(cls)
        object.__setattr__(instance, "settings", base.settings)
        object.__setattr__(instance, "embedder", base.embedder)
        object.__setattr__(instance, "vector_store", base.vector_store)
        object.__setattr__(instance, "retriever", base.retriever)
        object.__setattr__(instance, "reranker", base.reranker)
        object.__setattr__(instance, "llm", base.llm)
        object.__setattr__(instance, "rewriter_llm", base.rewriter_llm)
        object.__setattr__(instance, "session_store", base.session_store)
        object.__setattr__(instance, "domain", base.domain)
        object.__setattr__(instance, "rule_rewriter", base.rule_rewriter)
        object.__setattr__(instance, "llm_rewriter", base.llm_rewriter)
        object.__setattr__(instance, "llamaindex_retriever", base.llamaindex_retriever)
        object.__setattr__(instance, "torch_models", base.torch_models)
        return instance

    def get_torch_model(self, task_name: str) -> Any | None:
        """按任务名获取 PyTorch 模型实例。"""
        return self.torch_models.get(task_name)

    def get_llamaindex_retriever(self) -> Any | None:
        """获取 LlamaIndex 检索器实例（若已启用）。"""
        return self.llamaindex_retriever

    def is_escalation_needed(self, intent: str, sentiment: str) -> bool:
        """
        判断是否需要转人工。

        规则：
          - 意图在 escalation_intents 列表中 → 是
          - 情感为 negative → 是
          - 否则 → 否
        """
        settings = self.settings
        if not isinstance(settings, CS4Settings):
            return False
        if not settings.enable_escalation:
            return False
        if intent in settings.escalation_intents:
            return True
        if sentiment == "negative":
            return True
        return False
