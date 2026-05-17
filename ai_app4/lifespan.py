"""
ai_app4 Wealth AI Agent FastAPI 生命周期管理。

流程：
  1. 加载 WealthSettings
  2. 构建 WealthContainer（触发模型加载 + 索引预热）
  3. 预热所有 Warmupable 组件
  4. 关闭时释放资源
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.service.tools import register_wealth_tools


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 上下文管理器。"""
    settings = WealthSettings()
    container = WealthContainer.from_settings(settings)
    app.state.container = container
    app.state.settings = settings

    # 同步到全局上下文（供 LangGraph 节点使用，避免循环导入）
    context.set_container(container)
    context.set_settings(settings)

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
