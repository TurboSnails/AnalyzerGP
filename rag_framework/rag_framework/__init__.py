"""
RAG Framework — 通用检索增强生成框架

提供插件化的领域支持：
  - 核心抽象：LLM、Embedding、Retriever、Reranker、Session
  - 基础设施：配置、日志、异常、注册中心
  - 开箱即用：OpenAI 兼容 LLM、BGE-M3 Embedding、CrossEncoder Rerank、
              ChromaDB 向量存储、Tantivy BM25、混合检索 Fusion

使用示例：
    from rag_framework.core.config import RAGSettings
    from rag_framework.container import RAGContainer

    settings = RAGSettings()
    container = RAGContainer.from_settings(settings)
    container.chat_stream("什么是 Handler 内存泄漏？")
"""

__version__ = "0.1.0"

__all__ = [
    "RAGSettings",
    "RAGContainer",
    "DomainPlugin",
    "Embedder",
    "Reranker",
    "LLMClient",
    "Retriever",
    "SessionStore",
]

from rag_framework.core.config import RAGSettings
from rag_framework.container import RAGContainer
from rag_framework.domain.base import DomainPlugin
from rag_framework.embedding.base import Embedder
from rag_framework.rerank.base import Reranker
from rag_framework.llm.base import LLMClient
from rag_framework.retrieval.base import Retriever
from rag_framework.session.base import SessionStore
