"""
ai_app2 配置模块。

复用 rag_framework.core.config.RAGSettings，提供 LangGraph 特有配置常量。
"""
from __future__ import annotations

from rag_framework.core.config import RAGSettings, get_settings, reload_settings

# LangGraph / Agent 特有常量
MAX_STEPS = 10  # Agent tool calling 最大步数

__all__ = [
    "RAGSettings",
    "get_settings",
    "reload_settings",
    "MAX_STEPS",
]
