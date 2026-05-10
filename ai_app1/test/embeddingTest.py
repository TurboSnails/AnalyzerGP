import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token

from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parent

# 本地模型目录（download_bge_m3.py 默认下载到 test/models/<预设名>/）
_USE_LOCAL = os.getenv("EMBEDDING_MODEL_LOCAL", "1").strip().lower() not in ("0", "false", "no")

# bge-base-zh-v1.5（已下载）
_MODEL_DIR = _ROOT / "models" / "bge-base-zh-v1.5"

# 改用 bge-m3：先下载 → 再把上面改成 bge-m3 目录
#   在 ai_app1 下: uv run python test/download_bge_m3.py --preset bge-m3
#   或仓库根:     uv run python ai_app1/test/download_bge_m3.py --preset bge-m3
# _MODEL_DIR = _ROOT / "models" / "bge-m3"

# 强制走 Hub 模型名（不走本地）：export EMBEDDING_MODEL_LOCAL=0
# model = SentenceTransformer("BAAI/bge-base-zh-v1.5")

if _USE_LOCAL:
    if not _MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"未找到本地模型目录: {_MODEL_DIR}\n"
            "先执行: uv run python test/download_bge_m3.py   （默认拉 bge-base-zh）"
        )
    model = SentenceTransformer(str(_MODEL_DIR))
else:
    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-base-zh-v1.5"))

text1 = "空指针异常"
text2 = "对象为空导致闪退"

emb1 = model.encode(text1)
emb2 = model.encode(text2)

print(emb1)
print(len(emb1))

print(emb2)
print(len(emb2))
