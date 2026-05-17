"""
ai_app4 API 配额与限流管理器。

商业级别下，三方 API（Tavily、Alpha Vantage 等）按调用计费，
必须严格控制每个用户/租户的调用配额，防止账单爆炸。

设计原则：
  - 内存级配额跟踪（进程内），无需 Redis（单进程 FastAPI 足够）
  - 按租户和用户两级配额
  - 硬配额（拒绝）+ 软配额（警告但允许）
  - 支持按数据源独立配额
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuotaConfig:
    """单个数据源的配额配置。"""

    # 每日硬上限（超过后拒绝调用）
    daily_hard_limit: int = 100
    # 每日软上限（超过后记录警告但仍允许）
    daily_soft_limit: int = 80
    # 每小时上限（防止短时 burst）
    hourly_limit: int = 20
    # 单次请求最大结果数
    max_results_per_call: int = 5


@dataclass
class QuotaUsage:
    """单个用户/租户在某数据源上的配额使用情况。"""

    count_today: int = 0
    count_this_hour: int = 0
    last_call_time: float = 0.0
    # 记录每次调用的时间戳（用于滑动窗口清理）
    call_history: list[float] = field(default_factory=list)

    def record_call(self) -> None:
        """记录一次调用。"""
        now = time.time()
        self.call_history.append(now)
        self.last_call_time = now
        self._cleanup(now)
        self.count_today = len(self.call_history)
        self.count_this_hour = sum(1 for t in self.call_history if now - t < 3600)

    def _cleanup(self, now: float) -> None:
        """清理超过 24 小时的旧记录。"""
        cutoff = now - 86400
        self.call_history = [t for t in self.call_history if t > cutoff]

    def is_hard_limited(self, config: QuotaConfig) -> bool:
        """是否已达到每日硬上限。"""
        return self.count_today >= config.daily_hard_limit

    def is_soft_limited(self, config: QuotaConfig) -> bool:
        """是否已达到每日软上限（可警告但仍允许）。"""
        return self.count_today >= config.daily_soft_limit

    def is_hourly_limited(self, config: QuotaConfig) -> bool:
        """是否已达到每小时上限。"""
        return self.count_this_hour >= config.hourly_limit


class QuotaManager:
    """
    API 配额管理器。

    按 tenant_id + user_id + source_name 三级维度追踪配额使用。
    """

    def __init__(self) -> None:
        # (tenant_id, user_id, source_name) -> QuotaUsage
        self._usage: dict[tuple[str, str, str], QuotaUsage] = {}
        # source_name -> QuotaConfig
        self._configs: dict[str, QuotaConfig] = {}
        self._lock = asyncio.Lock()

    def register_config(self, source_name: str, config: QuotaConfig) -> None:
        """为数据源注册配额配置。"""
        self._configs[source_name] = config

    def get_config(self, source_name: str) -> QuotaConfig:
        """获取数据源的配额配置（不存在时返回默认值）。"""
        return self._configs.get(source_name, QuotaConfig())

    async def check_and_consume(
        self,
        tenant_id: str,
        user_id: str,
        source_name: str,
        requested_results: int = 1,
    ) -> tuple[bool, dict[str, Any]]:
        """
        检查配额并消费一次调用。

        Args:
            tenant_id: 租户标识
            user_id: 用户标识
            source_name: 数据源名称（如 "tavily_search"）
            requested_results: 本次请求的结果数

        Returns:
            (allowed, info_dict)
            - allowed: True 表示允许调用，False 表示已超配额
            - info_dict: 包含配额状态信息，用于 trace 记录
        """
        config = self.get_config(source_name)
        key = (tenant_id, user_id, source_name)

        async with self._lock:
            usage = self._usage.get(key)
            if usage is None:
                usage = QuotaUsage()
                self._usage[key] = usage

            # 强制清理（防止长期未访问的条目膨胀）
            usage._cleanup(time.time())

            info = {
                "source": source_name,
                "count_today": usage.count_today,
                "count_this_hour": usage.count_this_hour,
                "daily_hard_limit": config.daily_hard_limit,
                "daily_soft_limit": config.daily_soft_limit,
                "hourly_limit": config.hourly_limit,
                "requested_results": requested_results,
            }

            # 检查硬配额
            if usage.is_hard_limited(config):
                info["allowed"] = False
                info["reason"] = "daily_hard_limit_exceeded"
                return False, info

            # 检查小时配额
            if usage.is_hourly_limited(config):
                info["allowed"] = False
                info["reason"] = "hourly_limit_exceeded"
                return False, info

            # 检查单次结果数上限
            if requested_results > config.max_results_per_call:
                info["allowed"] = False
                info["reason"] = "max_results_per_call_exceeded"
                return False, info

            # 记录调用
            usage.record_call()
            info["allowed"] = True
            info["soft_warning"] = usage.is_soft_limited(config)
            info["count_after"] = usage.count_today

            return True, info

    async def get_usage_summary(
        self,
        tenant_id: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        """
        获取配额使用汇总。

        用于成本监控 Dashboard。
        """
        summary: dict[str, Any] = {}
        async with self._lock:
            for (t, u, source), usage in self._usage.items():
                if tenant_id and t != tenant_id:
                    continue
                if user_id and u != user_id:
                    continue

                config = self.get_config(source)
                summary[f"{t}:{u}:{source}"] = {
                    "tenant_id": t,
                    "user_id": u,
                    "source": source,
                    "count_today": usage.count_today,
                    "count_this_hour": usage.count_this_hour,
                    "daily_hard_limit": config.daily_hard_limit,
                    "daily_soft_limit": config.daily_soft_limit,
                    "hourly_limit": config.hourly_limit,
                    "remaining_today": max(0, config.daily_hard_limit - usage.count_today),
                    "last_call": usage.last_call_time,
                }
        return summary

    async def reset(self, tenant_id: str = "", user_id: str = "", source_name: str = "") -> int:
        """
        重置配额（管理用途）。

        如果指定了 tenant_id / user_id / source_name，只重置匹配的条目。
        如果全部为空，重置所有条目。

        Returns:
            重置的条目数量
        """
        removed = 0
        async with self._lock:
            keys_to_remove = []
            for key in list(self._usage.keys()):
                t, u, s = key
                if tenant_id and t != tenant_id:
                    continue
                if user_id and u != user_id:
                    continue
                if source_name and s != source_name:
                    continue
                keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._usage[key]
                removed += 1

        return removed


# ── 全局单例 ──────────────────────────────────────────────────────────────

_default_quota_manager: QuotaManager | None = None
_quota_lock = asyncio.Lock()


async def get_default_quota_manager() -> QuotaManager:
    """获取全局默认配额管理器（惰性初始化）。"""
    global _default_quota_manager
    if _default_quota_manager is None:
        async with _quota_lock:
            if _default_quota_manager is None:
                _default_quota_manager = QuotaManager()
                # 注册默认配额配置
                _default_quota_manager.register_config(
                    "tavily_search",
                    QuotaConfig(
                        daily_hard_limit=50,
                        daily_soft_limit=40,
                        hourly_limit=10,
                        max_results_per_call=5,
                    ),
                )
                _default_quota_manager.register_config(
                    "yahoo_finance",
                    QuotaConfig(
                        daily_hard_limit=500,
                        daily_soft_limit=400,
                        hourly_limit=100,
                        max_results_per_call=3,
                    ),
                )
                _default_quota_manager.register_config(
                    "fred_api",
                    QuotaConfig(
                        daily_hard_limit=200,
                        daily_soft_limit=150,
                        hourly_limit=50,
                        max_results_per_call=3,
                    ),
                )
    return _default_quota_manager
