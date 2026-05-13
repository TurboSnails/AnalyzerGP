"""
RAG 依赖注入容器

替代全局单例，统一管理所有组件的生命周期。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_framework.core.config import RAGSettings
from rag_framework.core.logger import setup_logging
from rag_framework.core.registry import get_domain
from rag_framework.domain.base import DomainPlugin
from rag_framework.embedding.base import Embedder
from rag_framework.embedding.sentence_transformer import STEmbedder
from rag_framework.llm.base import LLMClient
from rag_framework.llm.openai_client import OpenAILLMClient
from rag_framework.rerank.base import Reranker
from rag_framework.rerank.cross_encoder import CrossEncoderReranker
from rag_framework.retrieval.dense import DenseStore
from rag_framework.retrieval.sparse import BM25Store
from rag_framework.retrieval.fusion import HybridRetriever
from rag_framework.retrieval.query_rewriter import (
    RuleQueryRewriter,
    LLMQueryRewriter,
    QwenQueryRewriter,
)
from rag_framework.session.base import SessionStore
from rag_framework.session.memory_store import MemorySessionStore


@dataclass
class RAGContainer:
    """
    RAG 依赖注入容器。

    组合所有框架组件，提供统一访问入口。
    """
    settings: RAGSettings
    embedder: Embedder
    dense_store: DenseStore
    sparse_store: BM25Store
    retriever: HybridRetriever
    reranker: Reranker
    llm: LLMClient
    session_store: SessionStore
    domain: DomainPlugin
    rule_rewriter: RuleQueryRewriter | None = None
    llm_rewriter: LLMQueryRewriter | None = None

    @classmethod
    def from_settings(cls, settings: RAGSettings | None = None) -> "RAGContainer":
        """
        从配置构建完整容器。

        自动初始化日志、加载模型、建立数据库连接。
        """
        settings = settings or RAGSettings()
        setup_logging(level=settings.log_level)

        # Embedding
        embedder = STEmbedder(
            model_path=settings.embed_model_path,
            device=settings.embed_device,
            normalize=settings.embed_normalize,
        )

        # Dense Store
        dense_store = DenseStore(
            chroma_path=settings.chroma_db_path,
            embedder=embedder,
        )

        # Sparse Store
        sparse_store = BM25Store(
            index_dir=settings.bm25_index_dir,
            chroma_path=settings.chroma_db_path,
        )

        # Reranker
        reranker = CrossEncoderReranker(
            model_path=settings.reranker_model_path,
            max_length=settings.reranker_max_length,
            batch_size=settings.reranker_batch_size,
        )

        # LLM
        llm = OpenAILLMClient(
            base_url=settings.llm_base_url,
            api_key=settings.resolved_llm_api_key,
            model=settings.llm_model,
            backend=settings.llm_backend,
            max_tokens=settings.llm_max_tokens,
        )

        # Session
        session_store = MemorySessionStore(
            default_budget=settings.default_token_budget,
        )

        # Domain
        domain = get_domain(settings.active_domain)

        # Hybrid Retriever (依赖 domain 的 collection names)
        retriever = HybridRetriever(
            settings=settings,
            embedder=embedder,
            dense_store=dense_store,
            sparse_store=sparse_store,
            reranker=reranker,
            domain=domain,
        )

        # Rewriters
        rule_rewriter = RuleQueryRewriter(domain=domain)
        from pathlib import Path
        from rag_framework.core.logger import get_logger as _get_logger
        _clog = _get_logger("rag.container")
        backend = settings.rewriter_backend
        model_path = settings.rewriter_model
        if backend == "local" or (backend == "auto" and Path(model_path).is_dir()):
            _clog.info(f"LLM 改写器: backend=local, path={model_path!r}")
            llm_rewriter = QwenQueryRewriter(
                model_path=model_path,
                max_new_tokens=settings.rewriter_max_tokens,
            )
        else:
            _clog.info(f"LLM 改写器: backend=remote(minimax), model={llm.model!r}")
            llm_rewriter = LLMQueryRewriter(llm=llm, max_tokens=settings.rewriter_max_tokens)

        return cls(
            settings=settings,
            embedder=embedder,
            dense_store=dense_store,
            sparse_store=sparse_store,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            session_store=session_store,
            domain=domain,
            rule_rewriter=rule_rewriter,
            llm_rewriter=llm_rewriter,
        )

    # ─── 快捷方法 ───────────────────────────────────────────────────────────────

    async def chat_stream(
        self, query: str, user_id: str = "default_user"
    ):
        """
        端到端流式对话入口。

        1. 获取/创建 session
        2. 构建 messages（system + summary + history + retrieved context）
        3. 流式调用 LLM
        """
        from rag_framework.session.manager import SessionManager
        manager = SessionManager(
            store=self.session_store,
            llm=self.llm,
            retriever=self.retriever,
            domain=self.domain,
            settings=self.settings,
            rule_rewriter=self.rule_rewriter,
            llm_rewriter=self.llm_rewriter,
        )
        async for chunk in manager.chat_stream(query, user_id):
            yield chunk

    def build_routes(self, query: str, history: list[dict]) -> list:
        """
        根据 DomainPlugin 的分级规则生成多路检索路由。

        Returns:
            list[QueryRoute]，第一条为原始/改写后的主 query。
        """
        from rag_framework.domain.base import QueryRoute

        level = self.domain.rewrite_router_rules(query, history)

        if level == 2 and self.llm_rewriter is not None:
            return self.llm_rewriter.rewrite(query, history)

        if level == 1 and self.rule_rewriter is not None:
            return self.rule_rewriter.rewrite(query, history)

        # Level 0：只做查询分类，不扩写
        route = self.domain.classify_query(query, history)
        return [route]
