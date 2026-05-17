"""
ai_app4 Wealth AI Agent FastAPI 生命周期管理。

流程：
  1. 加载 WealthSettings
  2. 构建 WealthContainer（触发模型加载 + 索引预热）
  3. 初始化三轨融合检索器（ThreeTrackRetriever）
  4. 初始化缓存（MemoryCache）和配额管理器（QuotaManager）
  5. 预热所有 Warmupable 组件
  6. 关闭时释放资源
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.service.tools import register_wealth_tools
from ai_app4.service.cache import MemoryCache
from ai_app4.service.quota import QuotaManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 上下文管理器。"""
    settings = WealthSettings()
    container = WealthContainer.from_settings(settings)
    app.state.container = container
    app.state.settings = settings

    # 初始化缓存和配额管理器
    cache = MemoryCache(default_ttl=300)
    quota_manager = QuotaManager()
    # 注册默认配额配置
    from ai_app4.service.quota import QuotaConfig
    quota_manager.register_config(
        "tavily_search",
        QuotaConfig(daily_hard_limit=50, daily_soft_limit=40, hourly_limit=10, max_results_per_call=5),
    )
    quota_manager.register_config(
        "yahoo_finance",
        QuotaConfig(daily_hard_limit=500, daily_soft_limit=400, hourly_limit=100, max_results_per_call=3),
    )
    quota_manager.register_config(
        "fred_api",
        QuotaConfig(daily_hard_limit=200, daily_soft_limit=150, hourly_limit=50, max_results_per_call=3),
    )

    # 同步到全局上下文（供 LangGraph 节点使用，避免循环导入）
    context.set_container(container)
    context.set_settings(settings)
    context.set_cache(cache)
    context.set_quota_manager(quota_manager)

    # 初始化三轨融合检索器（如果启用）
    if settings.three_track_enabled:
        from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever
        from ai_app4.service.datasources.yahoo_finance import YahooFinanceSource
        from ai_app4.service.datasources.fred_api import FredAPISource
        from ai_app4.service.datasources.tavily_search import TavilySearchSource

        data_sources = []
        if settings.yahoo_finance_enabled:
            data_sources.append(YahooFinanceSource(cache=cache))
        if settings.fred_api_enabled and settings.fred_api_key:
            data_sources.append(FredAPISource(api_key=settings.fred_api_key, cache=cache))
        if settings.tavily_search_enabled and settings.tavily_api_key:
            data_sources.append(
                TavilySearchSource(
                    api_key=settings.tavily_api_key,
                    search_depth=settings.tavily_search_depth,
                    max_results=settings.tavily_max_results,
                )
            )

        three_track = ThreeTrackRetriever(
            settings=settings,
            local_retriever=container.retriever,
            data_sources=data_sources,
            quota_manager=quota_manager,
        )
        app.state.three_track_retriever = three_track
        context.set_three_track_retriever(three_track)

    # 注册 Wealth AI 数学计算工具
    register_wealth_tools()

    # 预热所有支持 Warmupable 的组件
    for target in container.warmup_targets():
        try:
            target.warmup()
        except Exception as exc:
            import logging
            logging.getLogger("ai_app4.lifespan").warning(f"预热失败: {exc}")

    yield

    # 关闭时释放资源（如需）
    # TODO: 释放 GPU 缓存等
