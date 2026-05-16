"""
LlamaIndex 索引配置描述。

将 LlamaIndex 丰富的索引结构抽象为 DomainPlugin 可声明的配置对象，
供 IndexBuilder 在 lifespan 中按需构建或加载持久化索引。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LlamaIndexConfig:
    """
    LlamaIndex 索引配置描述。

    由 DomainPlugin 声明，用于指导索引构建和 QueryEngine 行为。
    """
    index_type: str = "vector"           # vector | summary | keyword | tree | kg
    doc_paths: list[str] = field(default_factory=list)
    persist_dir: str = ""
    response_mode: str = "no_text"       # no_text | compact | tree_summarize | refine
    similarity_top_k: int = 10
    enable_hybrid: bool = False
    node_parser: dict[str, Any] = field(default_factory=dict)  # 分块参数，如 chunk_size / overlap


@dataclass
class IndexDescription:
    """
    运行时索引描述。

    在容器初始化时从 DomainPlugin.llamaindex_config 转换而来，
    包含已解析的绝对路径和运行时参数。
    """
    domain_name: str = ""
    index_type: str = "vector"
    persist_dir: str = ""
    doc_paths: list[str] = field(default_factory=list)
    response_mode: str = "no_text"
    similarity_top_k: int = 10
    enable_hybrid: bool = False
    node_parser: dict[str, Any] = field(default_factory=dict)
