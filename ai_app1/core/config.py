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