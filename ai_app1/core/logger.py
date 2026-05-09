"""
日志模块，统一管理各子模块的 logger 实例。
仅负责创建 logger，实际的 basicConfig 在模块导入时执行一次。
"""
import logging
import sys

# 全局配置：所有 logger 共用同一套配置
# format 中包含时间、级别、logger 名、消息，便于在 uvicorn 输出中区分来源
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# 各子模块独立的 logger，支持独立控制级别或输出目标
ai_client_logger = logging.getLogger("ai_client")
session_logger = logging.getLogger("session")
chat_logger = logging.getLogger("chat")

# Phase 1-3 新增模块 logger
vector_store_logger = logging.getLogger("vector_store")
bm25_logger = logging.getLogger("bm25_store")
reranker_logger = logging.getLogger("reranker")