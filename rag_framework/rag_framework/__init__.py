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

# ── 触发所有实现类的自注册 ──────────────────────────────────────────
# 以下导入不直接使用符号，但会执行模块底部的 register_xxx() 调用，
# 将各后端实现注册到全局工厂注册表中。
from rag_framework.embedding import sentence_transformer  # noqa: F401
from rag_framework.retrieval import dense, sparse, fusion  # noqa: F401
from rag_framework.llm import local_client, openai_client  # noqa: F401
from rag_framework.rerank import cross_encoder  # noqa: F401
from rag_framework.session import memory_store  # noqa: F401
from rag_framework.retrieval.query_rewriter import (  # noqa: F401
    rule_rewriter,
    llm_rewriter,
    qwen_rewriter,
)
