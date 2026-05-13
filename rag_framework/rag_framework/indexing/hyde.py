"""
HyDE (Hypothetical Document Embeddings) 问题生成器

对每个文档 chunk 使用 LLM 生成假设性问题，写入 HyDE collection。
查询时将用户 query 嵌入后在 HyDE collection 中检索，以弥补
query 与文档语言风格差异带来的语义鸿沟。
"""
from __future__ import annotations

import asyncio
from typing import Callable

from rag_framework.core.logger import get_logger
from rag_framework.domain.base import DomainPlugin
from rag_framework.llm.base import LLMClient

_logger = get_logger("rag.indexing.hyde")


async def generate_hyde_questions(
    chunks: list[str],
    domain: DomainPlugin,
    llm: LLMClient,
    batch_size: int = 8,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """
    为每个 chunk 异步生成 HyDE 问题。

    Args:
        chunks: 文档片段列表
        domain: 领域插件（提供 hyde prompt 模板）
        llm: LLM 客户端
        batch_size: 并发批次大小（控制 LLM 并发压力）
        on_progress: 进度回调 (done, total)

    Returns:
        与 chunks 等长的问题字符串列表（失败的位置返回空字符串）
    """
    total = len(chunks)
    results: list[str] = [""] * total

    for start in range(0, total, batch_size):
        batch = chunks[start: start + batch_size]
        tasks = [_generate_one(chunk, domain, llm) for chunk in batch]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, r in enumerate(batch_results):
            idx = start + i
            if isinstance(r, Exception):
                _logger.warning(f"HyDE 生成失败 chunk[{idx}]: {r}")
                results[idx] = ""
            else:
                results[idx] = str(r)

        done = min(start + batch_size, total)
        if on_progress:
            on_progress(done, total)
        _logger.info(f"HyDE 进度: {done}/{total}")

    return results


async def _generate_one(chunk: str, domain: DomainPlugin, llm: LLMClient) -> str:
    prompt = domain.get_hyde_prompt(chunk)
    return await llm.chat([{"role": "user", "content": prompt}])
