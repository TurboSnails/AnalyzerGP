"""
ai_app4 扩展容器。

在 RAGContainer 基础上，暴露 LlamaIndex Retriever 和 PyTorch 模型实例，
供 ai_app4 的 LangGraph 节点直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_framework.container import RAGContainer

from ai_app4.core.config import CS4Settings


@dataclass(frozen=True, slots=True)
class CS4Container(RAGContainer):
    """
    ai_app4 扩展依赖注入容器。

    继承 RAGContainer 全部字段，仅新增方法（无新字段）。
    通过 super().from_settings() 复用父类构建逻辑，自动返回 CS4Container 实例。
    """

    @classmethod
    def from_settings(cls, settings: CS4Settings | None = None) -> "CS4Container":
        """从 CS4Settings 构建扩展容器。"""
        settings = settings or CS4Settings()
        # 父类 from_settings 使用 cls 参数构造实例，
        # super() 传递 CS4Container 作为 cls，因此返回的是 CS4Container 实例
        return super().from_settings(settings)  # type: ignore[return-value]

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
