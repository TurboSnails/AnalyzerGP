"""
ai_app2 日志模块。

复用 rag_framework.core.logger，提供 ai_app2 各子模块的 logger 快捷访问。
"""
from __future__ import annotations

from rag_framework.core.logger import get_logger

graph_logger = get_logger("graph")
retrieve_logger = get_logger("retrieve")
chat_logger = get_logger("chat")
