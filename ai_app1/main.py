import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app1.api.chat import router as chat_router

app = FastAPI()

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
    """启动时预热所有模型和索引，避免首个请求承担懒加载成本。"""
    from ai_app1.service.embedding import get_embedding_service
    from ai_app1.service.reranker import _get_reranker_service
    from ai_app1.service.bm25_store import search as bm25_search

    from ai_app1.service.query_rewriter import _get_service as _get_rewriter_service

    await asyncio.to_thread(get_embedding_service()._ensure_model)
    await asyncio.to_thread(_get_reranker_service()._ensure_model)
    await asyncio.to_thread(_get_rewriter_service()._ensure_model)
    # 触发 BM25 索引加载（空查询会快速返回，但会完成索引初始化）
    await asyncio.to_thread(bm25_search, "", 1)

    print("[startup] 所有模型和索引预热完成")


@app.get("/")
def root():
    return FileResponse(_static / "index.html")