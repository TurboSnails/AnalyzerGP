"""
组件生命周期协议

定义 Warmupable（预热）和 Closable（关闭）两个运行时协议，
lifespan 通过 isinstance 检查统一调用，无需关心具体实现类。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Warmupable(Protocol):
    """支持异步预热的组件协议。"""

    async def warmup(self) -> None:
        """异步预热：加载模型、建立连接、预热缓存等。"""
        ...


@runtime_checkable
class Closable(Protocol):
    """支持异步关闭的组件协议。"""

    async def shutdown(self) -> None:
        """异步关闭：释放连接、清理资源、持久化状态等。"""
        ...
