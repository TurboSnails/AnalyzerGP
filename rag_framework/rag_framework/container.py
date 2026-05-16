"""
RAG 依赖注入容器

替代全局单例，统一管理所有组件的生命周期。
所有组件通过工厂注册表创建，支持热插拔替换实现类。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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
    llamaindex_retriever_registry,
    llm_registry,
    reranker_registry,
    retriever_registry,
    rewriter_registry,
    session_store_registry,
    torch_model_registry,
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
    llamaindex_retriever: Retriever | None = field(default=None, compare=False)
    torch_models: dict[str, Any] = field(default_factory=dict, compare=False)

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

        # ── 9. LlamaIndex Retriever（可选，由配置开关） ──
        llamaindex_retriever = None
        if settings.llamaindex_enabled and domain and domain.llamaindex_config:
            # 触发自注册（import-time side effect）
            from rag_framework.llamaindex import base as _li_base  # noqa: F401
            li_cfg = domain.llamaindex_config
            from rag_framework.llamaindex.index_config import IndexDescription

            idx_desc = IndexDescription(
                domain_name=domain.name,
                index_type=li_cfg.index_type,
                persist_dir=li_cfg.persist_dir or settings.llamaindex_index_dir,
                doc_paths=li_cfg.doc_paths,
                response_mode=li_cfg.response_mode or settings.llamaindex_response_mode,
                similarity_top_k=li_cfg.similarity_top_k or settings.llamaindex_similarity_top_k,
                enable_hybrid=li_cfg.enable_hybrid or settings.llamaindex_enable_hybrid,
                node_parser=li_cfg.node_parser,
            )
            if llamaindex_retriever_registry.is_registered("default"):
                try:
                    llamaindex_retriever = llamaindex_retriever_registry.create(
                        "default",
                        settings=settings,
                        embedder=embedder,
                        index_description=idx_desc,
                    )
                except Exception as exc:
                    retrieval_logger = __import__(
                        "rag_framework.core.logger", fromlist=["retrieval_logger"]
                    ).retrieval_logger
                    retrieval_logger.warning(f"LlamaIndex Retriever 创建失败: {exc}")

        # ── 10. PyTorch 任务模型（可选，由 DomainPlugin 声明） ──
        torch_models: dict[str, Any] = {}
        if domain and domain.torch_tasks:
            # 触发自注册（import-time side effect）
            from rag_framework.torch_models import (  # noqa: F401
                intent_classifier,
                sentiment_analyzer,
                entity_recognizer,
            )
            for task_cfg in domain.torch_tasks:
                task_name = task_cfg.task_name
                if not task_name:
                    continue
                model_key = task_cfg.model_path or task_cfg.model_id or task_name
                if torch_model_registry.is_registered(task_name):
                    try:
                        model = torch_model_registry.create(
                            task_name,
                            model_path=model_key,
                            device=settings.torch_device,
                            batch_size=task_cfg.batch_size,
                            **task_cfg.kwargs,
                        )
                        torch_models[task_name] = model
                    except Exception as exc:
                        logger = __import__(
                            "rag_framework.core.logger", fromlist=["get_logger"]
                        ).get_logger("rag.container")
                        logger.warning(f"PyTorch 模型 '{task_name}' 创建失败: {exc}")
                else:
                    # fallback：尝试用通用名注册的后端
                    for fallback_key in ("intent_classifier", "sentiment_analyzer", "entity_recognizer"):
                        if torch_model_registry.is_registered(fallback_key) and task_name.startswith(fallback_key.replace("_", "").replace("analyzer", "").replace("classifier", "").replace("recognizer", "")):
                            try:
                                model = torch_model_registry.create(
                                    fallback_key,
                                    model_path=model_key,
                                    device=settings.torch_device,
                                    batch_size=task_cfg.batch_size,
                                    **task_cfg.kwargs,
                                )
                                torch_models[task_name] = model
                                break
                            except Exception as exc:
                                logger = __import__(
                                    "rag_framework.core.logger", fromlist=["get_logger"]
                                ).get_logger("rag.container")
                                logger.warning(f"PyTorch 模型 fallback '{fallback_key}' 创建失败: {exc}")

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
            llamaindex_retriever=llamaindex_retriever,
            torch_models=torch_models,
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
        # LlamaIndex Retriever 预热
        if self.llamaindex_retriever is not None and isinstance(self.llamaindex_retriever, Warmupable):
            targets.append(self.llamaindex_retriever)
        # PyTorch 任务模型预热
        for model in self.torch_models.values():
            if isinstance(model, Warmupable):
                targets.append(model)
        return targets
