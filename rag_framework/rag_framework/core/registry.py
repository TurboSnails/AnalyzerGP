"""
插件注册中心

支持领域插件（DomainPlugin）的动态注册与发现。
"""
from __future__ import annotations

from typing import Type

from rag_framework.core.exceptions import DomainNotFoundError
from rag_framework.domain.base import DomainPlugin


class PluginRegistry:
    """领域插件注册表。"""

    def __init__(self) -> None:
        self._plugins: dict[str, Type[DomainPlugin]] = {}

    def register(self, plugin_cls: Type[DomainPlugin]) -> None:
        """注册一个领域插件类。"""
        # 实例化一次获取 name（无状态）
        instance = plugin_cls()
        self._plugins[instance.name] = plugin_cls

    def get(self, name: str) -> DomainPlugin:
        """
        获取指定领域的插件实例。

        Raises:
            DomainNotFoundError: 领域未注册
        """
        cls = self._plugins.get(name)
        if cls is None:
            raise DomainNotFoundError(
                f"领域 '{name}' 未注册。已注册: {list(self._plugins.keys())}"
            )
        return cls()

    def list_domains(self) -> list[str]:
        """列出所有已注册领域。"""
        return list(self._plugins.keys())


# 全局注册表实例
_global_registry = PluginRegistry()


def register_domain(plugin_cls: Type[DomainPlugin]) -> None:
    """快捷注册函数。"""
    _global_registry.register(plugin_cls)


def get_domain(name: str) -> DomainPlugin:
    """快捷获取函数。"""
    return _global_registry.get(name)


def list_domains() -> list[str]:
    """快捷列取函数。"""
    return _global_registry.list_domains()
