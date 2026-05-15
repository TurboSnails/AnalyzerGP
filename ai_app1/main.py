"""
ai_app1 — Android 开发助手（薄应用层）

基于 rag_framework + AndroidDomainPlugin 构建。
仅负责 HTTP 服务启动、静态文件、CORS 等 Web 层职责。

生命周期：
  1. lifespan 显式注册领域插件
  2. 通过工厂注册表创建 RAGContainer
  3. 统一预热所有 Warmupable 组件
  4. shutdown 时关闭所有 Closable 组件
"""
import asyncio
from contextlib import asynccontextmanager
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时注册领域插件、创建容器、并发预热；关闭时释放资源。"""
    from android_domain.plugin import AndroidDomainPlugin

    # 1. 显式注册领域插件到全局注册表（去 import-time 副作用）
    from rag_framework.core.registry import register_domain
    register_domain(AndroidDomainPlugin)

    # 2. 创建容器（通过工厂注册表，所有组件可插拔）
    #    若测试已注入容器，则复用，避免加载真实模型
    container = getattr(app.state, "container", None)
    if container is None:
        container = RAGContainer.from_settings(RAGSettings())
        app.state.container = container

    # 3. 并发预热所有 Warmupable 组件
    warmup_tasks = []
    for comp in container.warmup_targets():
        warmup_tasks.append(comp.warmup())

    if warmup_tasks:
        await asyncio.gather(*warmup_tasks)

    print("[startup] 所有模型和索引预热完成")
    yield

    # 4. shutdown：关闭所有 Closable 组件
    for comp in (
        container.embedder,
        container.reranker,
        container.vector_store,
        container.llm,
    ):
        if isinstance(comp, Closable):
            try:
                await comp.shutdown()
            except Exception as e:
                print(f"[shutdown] 关闭组件出错: {e}")


app = FastAPI(title="Android 开发助手", version="2.1.0", lifespan=lifespan)

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
    return {"status": "ok", "version": "2.1.0"}
