"""
ai_app1 — Android 开发助手（薄应用层）

基于 rag_framework + AndroidDomainPlugin 构建。
仅负责 HTTP 服务启动、静态文件、CORS 等 Web 层职责。
"""
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app1.api.chat import router as chat_router

app = FastAPI(title="Android 开发助手", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)

_static = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_static, html=True), name="static")


@app.on_event("startup")
async def preload_models():
    """启动时注册领域插件并预热模型和索引。"""
    from rag_framework.container import RAGContainer
    from rag_framework.core.config import get_settings
    from rag_framework.core.registry import register_domain
    from android_domain import AndroidDomainPlugin

    # 注册 Android 领域插件
    register_domain(AndroidDomainPlugin)

    settings = get_settings()
    container = RAGContainer.from_settings(settings)

    # 预热 embedding
    await asyncio.to_thread(container.embedder._ensure_model)
    # 预热 reranker
    await asyncio.to_thread(container.reranker._ensure_model)
    # 预热 BM25
    await asyncio.to_thread(container.sparse_store._ensure_loaded)

    print("[startup] 所有模型和索引预热完成")


@app.get("/")
def root():
    return FileResponse(_static / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}
