import os
from dotenv import load_dotenv

# 先尝试 ai_app2/.env，再回退到 ai_app1/.env
_current_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_current_dir, "..", ".env")
_ai_app1_env = os.path.join(_current_dir, "..", "..", "ai_app1", ".env")

for path in (_env_path, _ai_app1_env):
    if os.path.exists(path):
        load_dotenv(path)
        break

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 复用 ai_app1 构建好的 ChromaDB 索引（动态计算路径，避免硬编码）
CHROMA_DB_PATH = os.path.normpath(
    os.path.join(_current_dir, "..", "..", "ai_app1", "pre", "chroma_db")
)

# 会话默认值
DEFAULT_TOKEN_BUDGET = 4096
MAX_HISTORY = 4  # 保留最近对话轮数（每轮 user+assistant 各一条）
MAX_STEPS = 10   # Agent tool calling 最大步数

SYSTEM_PROMPT = "你是一个专业的Android开发助手，回答要简洁、准确"
