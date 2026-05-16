"""
LlamaIndex 适配层

提供 LlamaIndex QueryEngine 的 Retriever 接口适配，以及索引配置描述。
ai_app1/2/3 不触发此模块加载（惰性导入）。
"""
from rag_framework.llamaindex.base import LlamaIndexRetriever
from rag_framework.llamaindex.index_config import IndexDescription, LlamaIndexConfig

__all__ = ["LlamaIndexRetriever", "IndexDescription", "LlamaIndexConfig"]
