"""
内存 Session Store 实现

进程内字典存储，重启丢失。适合单实例部署。
"""
from __future__ import annotations

import threading

from rag_framework.core.factories import register_session_store
from rag_framework.core.logger import session_logger
from rag_framework.session.base import SessionStore, SessionData


class MemorySessionStore(SessionStore):
    """进程内内存会话存储。"""

    def __init__(self, default_budget: int = 4096) -> None:
        self._data: dict[str, SessionData] = {}
        self._default_budget = default_budget
        self._lock = threading.Lock()

    def get(self, user_id: str) -> SessionData:
        with self._lock:
            if user_id not in self._data:
                self._data[user_id] = SessionData(
                    user_id=user_id,
                    token_budget=self._default_budget,
                )
                session_logger.info(f"创建新会话: user_id={user_id}")
            else:
                session_logger.debug(
                    f"复用已有会话: user_id={user_id}, "
                    f"history_len={len(self._data[user_id].history)}"
                )
            return self._data[user_id]

    def save(self, session: SessionData) -> None:
        with self._lock:
            self._data[session.user_id] = session

    def delete(self, user_id: str) -> None:
        with self._lock:
            self._data.pop(user_id, None)

    def list_users(self) -> list[str]:
        with self._lock:
            return list(self._data.keys())


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_memory_session_store(
    default_budget: int = 4096,
) -> MemorySessionStore:
    return MemorySessionStore(default_budget=default_budget)


register_session_store("memory", _create_memory_session_store)
