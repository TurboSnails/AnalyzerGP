import os
from dotenv import load_dotenv

_current_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_current_dir, "..", ".env")
_ai_app1_env = os.path.join(_current_dir, "..", "..", "ai_app1", ".env")

for path in (_env_path, _ai_app1_env):
    if os.path.exists(path):
        load_dotenv(path)
        break

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 复用 rag_framework 的路径解析逻辑（自动识别 ai_app1/data/chroma_db）
from rag_framework.core.config import get_settings as _rag_get_settings
CHROMA_DB_PATH = _rag_get_settings().chroma_db_path

# ── 会话与预算 ──
DEFAULT_TOKEN_BUDGET = 4096
MAX_HISTORY = 4
MAX_STEPS = 10

SYSTEM_PROMPT = (
    "你是 Android 开发专家助手，回答要简洁、准确、专业。"
    "当参考资料不足以回答时，请明确指出，不要编造内容。"
)

# ── Agentic RAG 超参数 ──
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.65  # 检索质量置信度阈值
MAX_REWRITE_ITERATIONS = 2             # 最大查询改写轮数
MAX_SUB_QUERIES = 3                    # 单轮最大子查询数
SUB_QUERY_MIN_CONFIDENCE = 0.55        # 子查询最低置信度
ENABLE_KNOWLEDGE_GRAPH = True          # 是否启用轻量知识图谱增强
