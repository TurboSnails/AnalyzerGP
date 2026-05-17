"""
ai_app4 内存缓存层。

为外部数据源（Yahoo Finance、FRED API 等）提供结构化数据的本地缓存，
降低 API 调用频率和成本。

设计原则：
  - 简单：纯内存 dict，无需 Redis（单进程 FastAPI 场景足够）
  - 线程安全：asyncio.Lock 保护读写
  - TTL 自动过期：每次读取时检查过期时间，惰性清理
  - 降级友好：缓存未命中时直接返回 None，调用方自行降级
"""
from __future__ import annotations

import asyncio
import time
from typing import Any


class MemoryCache:
    """
    异步安全内存缓存，支持 TTL 过期。

    适用于：
      - 股价数据（缓存 5 分钟）
      - 宏观指标（缓存 1 小时）
      - 搜索结果（不缓存或极短 TTL）
    """

    def __init__(self, default_ttl: int = 300) -> None:
        """
        Args:
            default_ttl: 默认 TTL（秒），未指定时缓存 5 分钟
        """
        self._default_ttl = default_ttl
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expire_at)
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """
        获取缓存值。

        如果 key 不存在或已过期，返回 None。
        过期条目会在读取时惰性清理。
        """
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            value, expire_at = entry
            if time.time() > expire_at:
                # 惰性清理过期条目
                del self._store[key]
                return None

            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """
        设置缓存值。

        Args:
            key: 缓存键
            value: 缓存值（任意类型）
            ttl: 过期时间（秒），None 时使用 default_ttl
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expire_at = time.time() + effective_ttl

        async with self._lock:
            self._store[key] = (value, expire_at)

    async def delete(self, key: str) -> bool:
        """删除指定 key，返回是否成功删除。"""
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> None:
        """清空所有缓存。"""
        async with self._lock:
            self._store.clear()

    async def expire_all(self) -> int:
        """主动清理所有过期条目，返回清理数量。"""
        now = time.time()
        removed = 0
        async with self._lock:
            expired_keys = [k for k, (_, exp) in self._store.items() if now > exp]
            for k in expired_keys:
                del self._store[k]
                removed += 1
        return removed

    async def keys(self) -> list[str]:
        """返回当前所有未过期的 key 列表。"""
        now = time.time()
        async with self._lock:
            return [k for k, (_, exp) in self._store.items() if now <= exp]

    async def stats(self) -> dict[str, Any]:
        """返回缓存统计信息。"""
        now = time.time()
        async with self._lock:
            total = len(self._store)
            expired = sum(1 for _, exp in self._store.values() if now > exp)
            return {
                "total_keys": total,
                "expired_keys": expired,
                "active_keys": total - expired,
                "default_ttl_seconds": self._default_ttl,
            }


# ── 全局单例（进程内共享）─────────────────────────────────────────────────

_default_cache: MemoryCache | None = None
_cache_lock = asyncio.Lock()


async def get_default_cache() -> MemoryCache:
    """获取全局默认缓存实例（惰性初始化）。"""
    global _default_cache
    if _default_cache is None:
        async with _cache_lock:
            if _default_cache is None:
                _default_cache = MemoryCache(default_ttl=300)
    return _default_cache
