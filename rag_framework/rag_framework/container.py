"""
RAG 依赖注入容器

替代全局单例，统一管理所有组件的生命周期。
所有组件通过工厂注册表创建，支持热插拔替换实现类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# ── 触发所有实现类的自注册 ──────────────────────────────────────────
# 必须在导入 factories 之后、使用注册表之前，确保各后端已注册。
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

from rag_framework.core.config import RAGSettings
from rag_framework.core.factories import (
    embedder_registry,
    llm_registry,
    reranker_registry,
    retriever_registry,
    rewriter_registry,
    session_store_registry,
    vector_store_registry,
)
from rag_framework.core.lifecycle import Warmupable
from rag_framework.core.logger import setup_logging
from rag_framework.core.registry import get_domain, list_domains

if TYPE_CHECKING:
    from rag_framework.domain.base import DomainPlugin
    from rag_framework.embedding.base import Embedder
    from rag_framework.llm.base import LLMClient
    from rag_framework.rerank.base import Reranker
    from rag_framework.retrieval.base import Retriever, VectorStore
    from rag_framework.retrieval.query_rewriter.base import QueryRewriter
    from rag_framework.session.base import SessionStore


@dataclass(frozen=True, slots=True)
class RAGContainer:
    """
    RAG 依赖注入容器（不可变）。

    组合所有框架组件，提供统一访问入口。
    通过工厂注册表创建，各组件实现可热插拔替换。
    """

    settings: RAGSettings
    embedder: Embedder
    vector_store: VectorStore
    retriever: Retriever
    reranker: Reranker
    llm: LLMClient
    rewriter_llm: LLMClient
    session_store: SessionStore
    domain: DomainPlugin
    rule_rewriter: QueryRewriter | None = field(default=None, compare=False)
    llm_rewriter: QueryRewriter | None = field(default=None, compare=False)

    @classmethod
    def from_settings(
        cls,
        settings: RAGSettings | None = None,
        *,
        _domain_override: DomainPlugin | None = None,
    ) -> "RAGContainer":
        """
        从配置构建完整容器。

        自动初始化日志、通过工厂注册表创建各组件。
        若传入了 _domain_override 则跳过自动发现（用于测试）。
        """
        settings = settings or RAGSettings()
        setup_logging(level=settings.log_level)

        # ── 1. Embedder ──
        embedder = embedder_registry.create(
            settings.embed_backend,
            model_path=settings.embed_model_path,
            device=settings.embed_device,
            normalize=settings.embed_normalize,
        )

        # ── 2. Vector Store ──
        vector_store = vector_store_registry.create(
            settings.vector_store_backend,
            chroma_path=settings.chroma_db_path,
            embedder=embedder,
        )

        # ── 3. LLM（主 LLM，用于对话/生成/摘要） ──
        llm_kwargs = {
            "base_url": settings.llm_base_url,
            "api_key": settings.resolved_llm_api_key,
            "model": settings.llm_model,
            "backend": settings.llm_backend,
            "max_tokens": settings.llm_max_tokens,
            "max_concurrent": settings.llm_max_concurrent,
        }
        if settings.llm_backend == "local":
            llm_kwargs["model_path"] = settings.llm_local_model_path
        llm = llm_registry.create(
            settings.llm_backend,
            **llm_kwargs,
        )

        # ── 3.5 Rewriter LLM（查询改写专用，默认复用主 LLM 配置） ──
        rewriter_backend = settings.resolved_rewriter_llm_backend
        rewriter_llm_kwargs = {
            "base_url": settings.rewriter_llm_base_url,
            "api_key": settings.rewriter_llm_api_key,
            "model": settings.rewriter_llm_model,
            "backend": rewriter_backend,
            "max_tokens": settings.rewriter_llm_max_tokens,
            "max_concurrent": settings.llm_max_concurrent,
        }
        if rewriter_backend == "local":
            rewriter_llm_kwargs["model_path"] = settings.rewriter_llm_local_model_path
        rewriter_llm = llm_registry.create(
            rewriter_backend,
            **rewriter_llm_kwargs,
        )

        # ── 4. Reranker ──
        reranker = reranker_registry.create(
            settings.reranker_backend,
            model_path=settings.reranker_model_path,
            max_length=settings.reranker_max_length,
            batch_size=settings.reranker_batch_size,
        )

        # ── 5. Session Store ──
        session_store = session_store_registry.create(
            settings.session_store_backend,
            default_budget=settings.default_token_budget,
        )

        # ── 6. Domain ──
        if _domain_override is not None:
            domain = _domain_override
        else:
            # 兼容旧逻辑：若未注册则尝试自动发现并注册
            if settings.active_domain not in list_domains():
                try:
                    import importlib
                    import sys
                    from pathlib import Path

                    repo_root = Path(__file__).resolve().parents[2]
                    domain_pkg = repo_root / "domains" / settings.active_domain
                    if domain_pkg.is_dir():
                        pkg_path = str(domain_pkg)
                        if pkg_path not in sys.path:
                            sys.path.insert(0, pkg_path)
                    mod = importlib.import_module(f"{settings.active_domain}_domain")
                    # 自动查找并注册 DomainPlugin 子类
                    from rag_framework.core.registry import register_domain
                    from rag_framework.domain.base import DomainPlugin

                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if (
                            isinstance(attr, type)
                            and issubclass(attr, DomainPlugin)
                            and attr is not DomainPlugin
                        ):
                            register_domain(attr)
                            break
                except Exception:
                    pass
            domain = get_domain(settings.active_domain)

        # ── 7. Rewriters ──
        rule_rewriter = None
        try:
            rule_rewriter = rewriter_registry.create("rule", domain=domain)
        except Exception:
            pass

        llm_rewriter = None
        if rewriter_registry.is_registered("llm"):
            try:
                llm_rewriter = rewriter_registry.create("llm", llm=rewriter_llm)
            except Exception:
                pass

        # ── 8. Retriever ──
        retriever = retriever_registry.create(
            settings.retriever_backend,
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            reranker=reranker,
            domain=domain,
            domain_filter=domain.name if domain else "",
        )

        return cls(
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            retriever=retriever,
            reranker=reranker,
            llm=llm,
            rewriter_llm=rewriter_llm,
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
        根据 DomainPlugin 的分级规则生成多路检索路由（同步，供 ai_app2/ai_app3 直接调用）。

        Returns:
            list[QueryRoute]，第一条为原始/改写后的主 query。
        """
        # SessionManager 中已包含 rewriter 逻辑，此处复用其内部实现
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
        return manager._build_routes(query, history)

    # ─── 生命周期 ───────────────────────────────────────────────────────────────

    def warmup_targets(self) -> list[Warmupable]:
        """返回所有支持预热的组件（供 lifespan 统一调用）。"""
        targets: list[Warmupable] = []
        for comp in (self.embedder, self.reranker, self.vector_store, self.llm, self.rewriter_llm):
            if isinstance(comp, Warmupable):
                targets.append(comp)
        return targets
