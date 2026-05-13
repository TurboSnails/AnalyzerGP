"""
RAG Framework 日志模块

统一管理各子模块 logger 实例，支持结构化日志输出。
实际 basicConfig 在模块导入时执行一次。
"""
import logging
import sys
from typing import Literal

from rag_framework.core.config import get_settings


_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 logger 实例。

    首次调用时初始化全局 logging 配置，后续复用。
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    _LOGGERS[name] = logger
    return logger


def setup_logging(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] | None = None,
    fmt: str | None = None,
) -> None:
    """
    初始化全局 logging 配置。

    从 RAGSettings 读取默认配置，支持运行时覆盖。
    幂等：多次调用不会重复添加 handler。
    """
    if level is None:
        settings = get_settings()
        level = settings.log_level
    if fmt is None:
        settings = get_settings()
        fmt = settings.log_format

    numeric_level = getattr(logging, level)

    root = logging.getLogger()
    if root.handlers:
        # 已初始化过，仅更新级别
        root.setLevel(numeric_level)
        for h in root.handlers:
            h.setLevel(numeric_level)
        return

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


# 常用 logger 快捷访问
ai_client_logger = get_logger("ai_client")
session_logger = get_logger("session")
chat_logger = get_logger("chat")
vector_store_logger = get_logger("vector_store")
bm25_logger = get_logger("bm25_store")
reranker_logger = get_logger("reranker")
retrieval_logger = get_logger("retrieval")
embed_logger = get_logger("embedding")
indexing_logger = get_logger("indexing")
eval_logger = get_logger("eval")
