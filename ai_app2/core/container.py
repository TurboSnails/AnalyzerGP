"""
ai_app2 全局容器持有器。

在 main.py startup 中初始化并预热，供 Graph 节点和 API 层复用。
避免模块级直接实例化导致的启动延迟和循环导入。
"""
from __future__ import annotations

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings

_app_container: RAGContainer | None = None


def get_app_container() -> RAGContainer:
    """获取或创建 RAGContainer 单例。"""
    global _app_container
    if _app_container is None:
        _app_container = RAGContainer.from_settings(get_settings())
    return _app_container


def set_app_container(container: RAGContainer) -> None:
    """手动注入容器（用于测试）。"""
    global _app_container
    _app_container = container
