"""
ai_app4 FastAPI 生命周期管理。

流程：
  1. 加载 CS4Settings
  2. 构建 CS4Container（触发 LlamaIndex 索引加载 + PyTorch 模型加载）
  3. 预热所有 Warmupable 组件
  4. 关闭时释放资源
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ai_app4.core.config import CS4Settings
from ai_app4.core.container import CS4Container
from ai_app4.core import context


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI lifespan 上下文管理器。"""
    settings = CS4Settings()
    container = CS4Container.from_settings(settings)
    app.state.container = container
    app.state.settings = settings

    # 同步到全局上下文（供 LangGraph 节点使用，避免循环导入）
    context.set_container(container)
    context.set_settings(settings)

    # 预热所有支持 Warmupable 的组件
    for target in container.warmup_targets():
        try:
            target.warmup()
        except Exception as exc:
            import logging
            logging.getLogger("ai_app4.lifespan").warning(f"预热失败: {exc}")

    yield

    # 关闭时释放资源（如需）
    # TODO: 关闭 LlamaIndex StorageContext、释放 PyTorch GPU 缓存等
