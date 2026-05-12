import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DB_PATH = "/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db"

_AI_APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_AI_APP_ROOT)


def _resolve_bge_m3_path() -> str:
    """与 download_bge_m3.py 默认输出一致：优先含权重的目录。"""

    def has_weights(directory: str) -> bool:
        return os.path.isfile(os.path.join(directory, "pytorch_model.bin")) or os.path.isfile(
            os.path.join(directory, "model.safetensors")
        )

    for base in (_REPO_ROOT, _AI_APP_ROOT):
        candidate = os.path.join(base, "models", "bge-m3")
        if os.path.isdir(candidate) and has_weights(candidate):
            return candidate
    return os.path.join(_AI_APP_ROOT, "models", "bge-m3")


BGE_M3_PATH = os.getenv("BGE_M3_PATH", "").strip() or _resolve_bge_m3_path()


def _resolve_reranker_path() -> str:
    """优先使用本地已下载的 bge-reranker-base，避免网络/token 问题。"""

    def has_weights(directory: str) -> bool:
        return os.path.isfile(os.path.join(directory, "pytorch_model.bin")) or os.path.isfile(
            os.path.join(directory, "model.safetensors")
        )

    for base in (_REPO_ROOT, _AI_APP_ROOT):
        candidate = os.path.join(base, "models", "bge-reranker-base")
        if os.path.isdir(candidate) and has_weights(candidate):
            return candidate
    return "BAAI/bge-reranker-base"


# ─── CrossEncoder Reranker 模型 ─────────────────────────────────────────────
# 优先本地路径；否则从 HuggingFace Hub 自动下载
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "").strip() or _resolve_reranker_path()


def _resolve_query_rewriter_path() -> str:
    """优先使用本地已下载的 Qwen2.5-1.5B-Instruct，避免网络依赖。"""

    def has_weights(directory: str) -> bool:
        return os.path.isfile(os.path.join(directory, "pytorch_model.bin")) or os.path.isfile(
            os.path.join(directory, "model.safetensors")
        )

    for base in (_REPO_ROOT, _AI_APP_ROOT):
        candidate = os.path.join(base, "models", "qwen2.5-1.5b-instruct")
        if os.path.isdir(candidate) and has_weights(candidate):
            return candidate
    return "Qwen/Qwen2.5-1.5B-Instruct"


# ─── Query Rewriter 模型（Qwen2.5-1.5B-Instruct） ────────────────────────────
QUERY_REWRITER_MODEL = os.getenv("QUERY_REWRITER_MODEL", "").strip() or _resolve_query_rewriter_path()