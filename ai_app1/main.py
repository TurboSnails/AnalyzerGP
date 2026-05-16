"""
ai_app1 — 多领域 RAG 助手（薄应用层）

基于 rag_framework 构建，支持同时加载多个 DomainPlugin。
各领域共享 Embedding、LLM、Reranker、ChromaDB 向量库与 BM25 稀疏索引，
通过统一的 knowledge_base collection + domain metadata 实现领域隔离。
SessionStore / Retriever 按领域隔离（各自 domain_filter）。

生命周期：
  1. lifespan 显式注册所有领域插件
  2. 创建基础容器（共享重型组件）
  3. 创建共享 BM25Store（统一索引，所有领域共用）
  4. 为每个领域派生独立容器（独立 session_store / retriever / domain_filter）
  5. 统一预热所有 Warmupable 组件（自动去重，避免重复加载模型）
  6. shutdown 时释放所有 Closable 组件（自动去重）
"""
import asyncio
import os
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from rag_framework.container import RAGContainer
from rag_framework.core.config import RAGSettings
from rag_framework.core.lifecycle import Warmupable, Closable

# ── 触发所有实现类的自注册（子模块导入不会自动执行 __init__.py） ──
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

from ai_app1.api.chat import router as chat_router

# ── 配置：要同时加载的领域插件 ───────────────────────────────────────────────
# 新增加领域时，在这里导入并添加到 _DOMAIN_CLASSES 即可
# 索引脚本负责在 metadata 中写入对应 domain 标记，无需修改任何 collection 名称
_DOMAIN_CLASSES = []

try:
    from msmarco_domain.plugin import MSMarcoDomainPlugin
    _DOMAIN_CLASSES.append(MSMarcoDomainPlugin)
except Exception:
    pass

try:
    from android_domain.plugin import AndroidDomainPlugin
    _DOMAIN_CLASSES.append(AndroidDomainPlugin)
except Exception:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时注册领域插件、创建多容器、并发预热；关闭时释放资源。"""
    from rag_framework.core.registry import register_domain, get_domain, list_domains
    from rag_framework.core.factories import (
        retriever_registry,
        session_store_registry,
        rewriter_registry,
    )
    from rag_framework.retrieval.sparse import BM25Store

    # 1. 显式注册所有领域插件到全局注册表（去 import-time 副作用）
    for cls in _DOMAIN_CLASSES:
        register_domain(cls)

    registered = list_domains()
    if not registered:
        raise RuntimeError("没有成功注册任何领域插件，请检查依赖与导入路径")

    # 2. 确定默认激活领域（可通过环境变量覆盖）
    default_domain = os.getenv("RAG_ACTIVE_DOMAIN", registered[0])
    if default_domain not in registered:
        default_domain = registered[0]

    # 3. 创建基础容器（包含共享的 embedder / vector_store / llm / reranker）
    base_settings = RAGSettings()
    base_settings = base_settings.model_copy(update={"active_domain": default_domain})
    base_container = RAGContainer.from_settings(base_settings)
    base_chroma = Path(base_settings.chroma_db_path)

    # 4. 创建共享 BM25Store（统一索引，所有领域共用同一个 tantivy 目录）
    #    从统一的 knowledge_base collection 构建，内部文档带 domain 字段
    shared_bm25 = BM25Store(
        index_dir=base_settings.bm25_index_dir,
        chroma_path=base_settings.chroma_db_path,
        collection_name="knowledge_base",
    )

    # 5. 为每个注册领域创建派生容器
    #    - 共享 embedder / vector_store / llm / reranker / bm25（避免重复加载模型）
    #    - 独立 session_store / retriever（带各自的 domain_filter）
    containers: dict[str, RAGContainer] = {}

    for name in registered:
        domain_settings = base_settings.model_copy(update={"active_domain": name})

        domain = get_domain(name)
        retriever = retriever_registry.create(
            domain_settings.retriever_backend,
            settings=domain_settings,
            embedder=base_container.embedder,
            vector_store=base_container.vector_store,
            reranker=base_container.reranker,
            domain=domain,
            sparse_store=shared_bm25,
            domain_filter=domain.name if domain else "",
        )
        session_store = session_store_registry.create(
            domain_settings.session_store_backend,
            default_budget=domain_settings.default_token_budget,
        )
        rule_rewriter = None
        try:
            rule_rewriter = rewriter_registry.create("rule", domain=domain)
        except Exception:
            pass

        if name == default_domain:
            containers[name] = replace(
                base_container,
                settings=domain_settings,
                domain=domain,
                retriever=retriever,
                session_store=session_store,
                rule_rewriter=rule_rewriter,
            )
        else:
            containers[name] = replace(
                base_container,
                settings=domain_settings,
                domain=domain,
                retriever=retriever,
                session_store=session_store,
                rule_rewriter=rule_rewriter,
            )

    app.state.containers = containers
    # 保留兼容：app.state.container 指向默认领域容器
    app.state.container = containers.get(default_domain)

    # 6. 并发预热所有 Warmupable 组件（按对象 id 去重，避免重复预热）
    seen_ids: set[int] = set()
    warmup_tasks = []
    for c in containers.values():
        for comp in c.warmup_targets():
            cid = id(comp)
            if cid not in seen_ids:
                seen_ids.add(cid)
                warmup_tasks.append(comp.warmup())

    if warmup_tasks:
        await asyncio.gather(*warmup_tasks)

    print(f"[startup] 已加载领域: {list(containers.keys())}，默认: {default_domain}")
    print("[startup] 所有模型和索引预热完成")
    yield

    # 7. shutdown：关闭所有 Closable 组件（按对象 id 去重）
    seen_ids = set()
    for c in containers.values():
        for comp in (c.embedder, c.reranker, c.vector_store, c.llm):
            cid = id(comp)
            if cid not in seen_ids and isinstance(comp, Closable):
                seen_ids.add(cid)
                try:
                    await comp.shutdown()
                except Exception as e:
                    print(f"[shutdown] 关闭组件出错: {e}")


app = FastAPI(title="多领域 RAG 助手", version="2.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

_static = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_static, html=True), name="static")


@app.get("/")
def root():
    return FileResponse(_static / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.3.0"}
