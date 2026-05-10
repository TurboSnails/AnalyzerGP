"""
日志模块，统一管理各子模块的 logger 实例。
"""
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

graph_logger = logging.getLogger("graph")
retrieve_logger = logging.getLogger("retrieve")
eval_logger = logging.getLogger("evaluate")
chat_logger = logging.getLogger("chat")
compress_logger = logging.getLogger("compress")
kg_logger = logging.getLogger("knowledge_graph")
