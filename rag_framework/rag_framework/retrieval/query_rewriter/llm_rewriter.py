"""
LLM-Based Query Rewriter

使用 LLM 将含代词、上下文依赖或模糊表述的查询改写为独立清晰的检索 query。
适用于 level-2 改写（见 DomainPlugin.rewrite_router_rules）。

同步接口内部通过独立线程运行 async LLM 调用，避免与外层 event loop 冲突。
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from rag_framework.core.logger import get_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.llm.base import LLMClient
from rag_framework.retrieval.query_rewriter.base import QueryRewriter

_logger = get_logger("rag.rewriter.llm")

_SYSTEM_PROMPT = (
    "你是查询改写助手。根据对话上下文，将用户问题改写为 2-3 个独立的检索 query，"
    "每行一个，不含编号，不含额外解释。"
    "第一条必须是用完整语义表达的独立问题（消解代词和指代）。"
)


class LLMQueryRewriter(QueryRewriter):
    """基于 LLM 的查询改写器。"""

    def __init__(self, llm: LLMClient, max_tokens: int = 128) -> None:
        self._llm = llm
        self._max_tokens = max_tokens
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="llm_rewriter",
        )

    def rewrite(self, query: str, history: list[dict]) -> list[QueryRoute]:
        messages = self._build_messages(query, history)
        try:
            raw = self._run_sync(self._llm.chat(messages))
        except Exception as exc:
            _logger.warning(f"LLM 改写失败，降级返回原始 query: {exc}")
            return [QueryRoute(text=query, type="original", weight=1.0)]

        lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
        if not lines:
            return [QueryRoute(text=query, type="original", weight=1.0)]

        routes: list[QueryRoute] = [
            QueryRoute(text=query, type="original", weight=1.0),
        ]
        for i, line in enumerate(lines[:3]):
            routes.append(
                QueryRoute(
                    text=line,
                    type="semantic",
                    weight=round(0.90 - i * 0.10, 2),
                    routes=["dense", "bm25"],
                )
            )
        _logger.debug(f"LLM 改写: {query!r} → {len(routes)} 条")
        return routes

    def _build_messages(self, query: str, history: list[dict]) -> list[dict]:
        ctx_lines: list[str] = []
        for msg in (history[-4:] if len(history) > 4 else history):
            role = "用户" if msg.get("role") == "user" else "AI"
            ctx_lines.append(f"{role}: {str(msg.get('content', ''))[:80]}")

        user_content = (
            f"对话上下文：\n{''.join(ctx_lines)}\n\n问题：{query}"
            if ctx_lines
            else f"问题：{query}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _run_sync(self, coro) -> str:
        """在独立线程的新 event loop 中执行协程，避免 nested loop 报错。"""
        def _target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        return self._executor.submit(_target).result(timeout=15)
