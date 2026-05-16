"""
组件工厂注册表 — 所有可插拔组件的统一创建入口。

设计原则：
  1. 注册与创建分离：实现类在模块加载时自注册，容器仅通过名称创建
  2. 配置透传：factory 接收 kwargs，具体实现类自行解析需要的字段
  3. 延迟加载：factory 本身不 import 重型依赖，由具体实现类按需 import
  4. 每个组件类型独立注册表，互不干扰

使用示例：
    from rag_framework.core.factories import llm_registry
    client = llm_registry.create("openai", base_url=..., api_key=...)
"""
from __future__ import annotations

from typing import TypeVar, Generic, Callable, Any

from rag_framework.core.exceptions import ComponentNotFoundError

T = TypeVar("T")


class _Registry(Generic[T]):
    """泛型组件注册表。线程安全（只读创建，注册在 import 时完成）。"""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> None:
        """注册一个具名工厂函数。"""
        if name in self._entries:
            raise ValueError(f"{self._kind} backend '{name}' 已注册")
        self._entries[name] = factory

    def create(self, name: str, **kwargs: Any) -> T:
        """通过名称创建组件实例。"""
        factory = self._entries.get(name)
        if factory is None:
            available = ", ".join(sorted(self._entries.keys()))
            raise ComponentNotFoundError(
                f"{self._kind} backend '{name}' 未注册。可用: [{available}]"
            )
        return factory(**kwargs)

    def list(self) -> list[str]:
        """列出所有已注册的后端名称。"""
        return sorted(self._entries.keys())

    def is_registered(self, name: str) -> bool:
        """检查指定后端是否已注册。"""
        return name in self._entries


# ─── 全局注册表实例 ──────────────────────────────────────────
embedder_registry = _Registry[Any]("embedder")
vector_store_registry = _Registry[Any]("vector_store")
llm_registry = _Registry[Any]("llm")
reranker_registry = _Registry[Any]("reranker")
session_store_registry = _Registry[Any]("session_store")
rewriter_registry = _Registry[Any]("rewriter")
retriever_registry = _Registry[Any]("retriever")
llamaindex_retriever_registry = _Registry[Any]("llamaindex_retriever")
torch_model_registry = _Registry[Any]("torch_model")
persistence_registry = _Registry[Any]("persistence")


# ─── 便捷注册函数 ────────────────────────────────────────────
def register_embedder(name: str, factory: Callable[..., Any]) -> None:
    embedder_registry.register(name, factory)


def register_vector_store(name: str, factory: Callable[..., Any]) -> None:
    vector_store_registry.register(name, factory)


def register_llm(name: str, factory: Callable[..., Any]) -> None:
    llm_registry.register(name, factory)


def register_reranker(name: str, factory: Callable[..., Any]) -> None:
    reranker_registry.register(name, factory)


def register_session_store(name: str, factory: Callable[..., Any]) -> None:
    session_store_registry.register(name, factory)


def register_rewriter(name: str, factory: Callable[..., Any]) -> None:
    rewriter_registry.register(name, factory)


def register_retriever(name: str, factory: Callable[..., Any]) -> None:
    retriever_registry.register(name, factory)


def register_llamaindex_retriever(name: str, factory: Callable[..., Any]) -> None:
    llamaindex_retriever_registry.register(name, factory)


def register_torch_model(name: str, factory: Callable[..., Any]) -> None:
    torch_model_registry.register(name, factory)


def register_persistence(name: str, factory: Callable[..., Any]) -> None:
    persistence_registry.register(name, factory)
