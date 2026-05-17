"""
ai_app4 — Wealth AI Agent v4.0 FastAPI 入口（端口 8004）。

基于 LangGraph 的全球资产与宏观经济多步推演智能助理。
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app4.api.chat import router as chat_router
from ai_app4.lifespan import lifespan

app = FastAPI(
    title="ai_app4 — Wealth AI Agent",
    description="基于 LangGraph 的全球资产与宏观经济多步推演智能助理",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(chat_router, prefix="/api")

_static = Path(__file__).parent / "static"
if _static.is_dir():
    app.mount("/ui", StaticFiles(directory=_static, html=True), name="static")


@app.get("/")
def root():
    return {"message": "Wealth AI Agent v4.0 已启动", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}
