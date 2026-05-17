"""
ai_app4 Wealth AI Agent 扩展容器。

在 RAGContainer 基础上，提供 Wealth AI 专属方法，
供 LangGraph 节点直接调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_framework.container import RAGContainer

from ai_app4.core.config import WealthSettings


@dataclass(frozen=True, slots=True)
class WealthContainer(RAGContainer):
    """
    Wealth AI Agent 扩展依赖注入容器。

    继承 RAGContainer 全部字段，仅新增方法（无新字段）。
    通过 super().from_settings() 复用父类构建逻辑，自动返回 WealthContainer 实例。
    """

    @classmethod
    def from_settings(cls, settings: WealthSettings | None = None) -> "WealthContainer":
        """从 WealthSettings 构建扩展容器。"""
        settings = settings or WealthSettings()
        # 显式传递 cls，确保 RAGContainer.from_settings 内部 return cls(...) 返回 WealthContainer
        return super(WealthContainer, cls).from_settings(settings)  # type: ignore[return-value]
