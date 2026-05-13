"""
Session Store 抽象基类

管理用户会话的生命周期：创建、读取、更新、删除。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionData:
    """会话数据。"""
    user_id: str
    history: list[dict] = field(default_factory=list)
    summary: str = ""
    trimmed: list[dict] = field(default_factory=list)
    token_budget: int = 4096
    metadata: dict = field(default_factory=dict)


class SessionStore(ABC):
    """会话存储抽象基类。"""

    @abstractmethod
    def get(self, user_id: str) -> SessionData:
        """获取或创建会话。"""
        ...

    @abstractmethod
    def save(self, session: SessionData) -> None:
        """保存会话。"""
        ...

    @abstractmethod
    def delete(self, user_id: str) -> None:
        """删除会话。"""
        ...

    @abstractmethod
    def list_users(self) -> list[str]:
        """列出所有用户 ID。"""
        ...
