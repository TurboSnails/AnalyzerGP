"""
检索包装模块：复用 ai_app1 的完整混合检索管道。

直接透传 ai_app1.service.vector_store.query_db，保持接口完全一致。
"""
from ai_app1.service.vector_store import query_db

__all__ = ["query_db"]
