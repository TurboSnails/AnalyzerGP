"""
ai_app4 全局上下文。

用于在 Lifespan → LangGraph 节点之间传递 Container 和 Settings，
避免循环导入（nodes.py 不直接导入 main.py）。

注意：这不是长期方案，ai_app4 商业化后应替换为正式的依赖注入或 contextvars。
"""
from __future__ import annotations

from typing import Any

_container: Any = None
_settings: Any = None


def set_container(container: Any) -> None:
    """在 lifespan 中设置容器实例。"""
    global _container
    _container = container


def get_container() -> Any:
    """在 LangGraph 节点中获取容器实例。"""
    return _container


def set_settings(settings: Any) -> None:
    """在 lifespan 中设置配置实例。"""
    global _settings
    _settings = settings


def get_settings() -> Any:
    """在 LangGraph 节点中获取配置实例。"""
    return _settings
