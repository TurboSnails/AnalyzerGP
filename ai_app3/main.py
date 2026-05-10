from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ai_app3.api.chat import router as chat_router

app = FastAPI(title="ai_app3 - Agentic RAG")

app.include_router(chat_router)

_static = Path(__file__).parent / "static"
app.mount("/ui", StaticFiles(directory=_static, html=True), name="static")


@app.get("/")
def root():
    return FileResponse(_static / "index.html")
