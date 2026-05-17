"""
ai_app4 全局上下文。

用于在 Lifespan → LangGraph 节点之间传递 Container、Settings、
ThreeTrackRetriever、Cache 和 QuotaManager，
避免循环导入（nodes.py 不直接导入 main.py）。

注意：这不是长期方案，ai_app4 商业化后应替换为正式的依赖注入或 contextvars。
"""
from __future__ import annotations

from typing import Any

_container: Any = None
_settings: Any = None
_three_track_retriever: Any = None
_cache: Any = None
_quota_manager: Any = None


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


# ── ThreeTrackRetriever ──────────────────────────────────────────────────


def set_three_track_retriever(retriever: Any) -> None:
    """在 lifespan 中设置三轨融合检索器。"""
    global _three_track_retriever
    _three_track_retriever = retriever


def get_three_track_retriever() -> Any:
    """在 LangGraph 节点中获取三轨融合检索器。"""
    return _three_track_retriever


# ── Cache ────────────────────────────────────────────────────────────────


def set_cache(cache: Any) -> None:
    """在 lifespan 中设置缓存实例。"""
    global _cache
    _cache = cache


def get_cache() -> Any:
    """在 LangGraph 节点中获取缓存实例。"""
    return _cache


# ── QuotaManager ─────────────────────────────────────────────────────────


def set_quota_manager(quota: Any) -> None:
    """在 lifespan 中设置配额管理器。"""
    global _quota_manager
    _quota_manager = quota


def get_quota_manager() -> Any:
    """在 LangGraph 节点中获取配额管理器。"""
    return _quota_manager
