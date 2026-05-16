"""
ai_app4 — 商业化客服系统 FastAPI 入口（端口 8004）。

基于 LangGraph + LlamaIndex + PyTorch 的第三代 RAG 客服系统。
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app4.api.chat import router as chat_router
from ai_app4.lifespan import lifespan

app = FastAPI(
    title="ai_app4 — 商业化客服系统",
    description="基于 LangGraph + LlamaIndex + PyTorch 的智能客服系统",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/api")

_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/ui", StaticFiles(directory=_static, html=True), name="static")


@app.get("/")
def root():
    return {"message": "ai_app4 客服系统已启动", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
