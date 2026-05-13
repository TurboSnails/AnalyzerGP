"""
ai_app1 — Android 开发助手（薄应用层）

基于 rag_framework + AndroidDomainPlugin 构建。
仅负责 HTTP 服务启动、静态文件、CORS 等 Web 层职责。
"""
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app1.api.chat import router as chat_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    """启动时注册领域插件并并发预热所有模型。"""
    from rag_framework.core.registry import register_domain
    from android_domain import AndroidDomainPlugin

    # 注册 Android 领域插件（必须在 get_container() 之前完成）
    register_domain(AndroidDomainPlugin)

    # 使用 chat 模块的单例容器——确保预热的和请求用的是同一个实例
    from ai_app1.api.chat import get_container
    container = get_container()

    # 并发预热 embedding / reranker / BM25（互相独立，节省启动时间）
    await asyncio.gather(
        asyncio.to_thread(container.embedder._ensure_model),
        asyncio.to_thread(container.reranker._ensure_model),
        asyncio.to_thread(container.sparse_store._ensure_loaded),
    )

    # 预热 Qwen 改写器（避免首条用户请求承担 ~3s 加载延迟）
    if hasattr(container.llm_rewriter, "_ensure_loaded"):
        await asyncio.to_thread(container.llm_rewriter._ensure_loaded)

    print("[startup] 所有模型和索引预热完成")
    yield
    # shutdown：无需显式清理（进程退出时自动释放）


app = FastAPI(title="Android 开发助手", version="2.0.0", lifespan=lifespan)

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
    return {"status": "ok", "version": "2.0.0"}
