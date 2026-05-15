"""
OpenAI 兼容 LLM Client 实现

支持 minimax / ollama / openai 等多后端，统一走 openai SDK。
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import openai

from rag_framework.core.factories import register_llm
from rag_framework.core.lifecycle import Closable
from rag_framework.core.logger import ai_client_logger
from rag_framework.llm.base import LLMClient


class OpenAILLMClient(LLMClient, Closable):
    """
    多后端 LLM 客户端（OpenAI 兼容协议）。

    支持流式/非流式对话、工具调用、摘要生成。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        backend: str = "openai",
        max_tokens: int = 512,
        timeout: float = 120.0,
        max_concurrent: int = 3,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._backend = backend
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_concurrent = max_concurrent
        self._semaphore: asyncio.Semaphore | None = None
        self.client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        ai_client_logger.info(
            f"OpenAILLMClient 初始化: backend={backend}, base_url={base_url}, "
            f"model={model}, max_concurrent={max_concurrent}"
        )

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def model(self) -> str:
        return self._model

    def _get_semaphore(self) -> asyncio.Semaphore:
        """懒初始化 Semaphore（必须在 async 上下文中首次调用）。"""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    # ─── 非流式 ─────────────────────────────────────────────────────────────────

    async def chat(self, messages: list[dict], use_tools: bool = False) -> str:
        kwargs = self._build_kwargs(messages, use_tools, stream=False)
        start = time.monotonic()
        async with self._get_semaphore():
            response = await self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        ai_client_logger.info(
            f"[{self._backend}] chat 完成: content_len={len(content)}, "
            f"耗时={time.monotonic() - start:.2f}s"
        )
        return content

    async def summarize(self, history: list[dict]) -> str:
        history_text = "\n".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in history
        )
        prompt = [
            {"role": "user", "content": "请总结以下对话的关键信息，用于后续对话参考"},
            {"role": "user", "content": history_text},
        ]
        return await self.chat(prompt, use_tools=False)

    async def run_agent(self, messages: list[dict]) -> str:
        """非流式工具增强对话。"""
        MAX_STEPS = 10
        for _ in range(MAX_STEPS):
            kwargs = self._build_kwargs(messages, use_tools=True, stream=False)
            response = await self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                return msg.content or ""

            tool_results = self._execute_tools(tool_calls)
            messages.append(msg.model_dump())
            messages.extend(tool_results)

        ai_client_logger.warning(f"[{self._backend}] run_agent 达到最大步数: {MAX_STEPS}")
        return "Agent stopped: max steps reached"

    # ─── 流式 ───────────────────────────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[dict], use_tools: bool = False
    ) -> AsyncIterator[str]:
        kwargs = self._build_kwargs(messages, use_tools, stream=True)
        t0 = time.perf_counter()
        # Semaphore 只门控连接建立，不持锁贯穿整个流式读取，避免长占槽位
        async with self._get_semaphore():
            response = await self.client.chat.completions.create(**kwargs)

        first_token_ms = None
        char_count = 0
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - t0) * 1000
                char_count += len(delta.content)
                yield delta.content

        total_ms = (time.perf_counter() - t0) * 1000
        gen_ms = total_ms - (first_token_ms or 0)
        cps = (char_count / gen_ms * 1000) if gen_ms > 0 else 0
        ai_client_logger.info(
            f"[{self._backend}] LLM 流式完成: TTFT={first_token_ms or 0:.0f}ms, "
            f"生成={gen_ms:.0f}ms, 总={total_ms:.0f}ms, "
            f"输出={char_count}字符 ({cps:.0f}字符/秒)"
        )

    async def run_agent_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """流式工具增强对话，增量收集 tool_calls。"""
        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            kwargs = self._build_kwargs(messages, use_tools=True, stream=True)
            t0 = time.perf_counter()
            response = await self.client.chat.completions.create(**kwargs)

            full = ""
            tool_calls: list[dict] = []
            first_token_ms = None

            async for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if delta.content:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    full += delta.content
                    yield delta.content

                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = getattr(tc, "index", 0)
                        while len(tool_calls) <= idx:
                            tool_calls.append({
                                "id": "", "function": {"name": "", "arguments": ""},
                                "type": "function",
                            })
                        if tc.id:
                            tool_calls[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls[idx]["function"]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls[idx]["function"]["arguments"] += tc.function.arguments

            total_ms = (time.perf_counter() - t0) * 1000
            ai_client_logger.info(
                f"[{self._backend}] LLM step={step + 1}: "
                f"TTFT={first_token_ms or 0:.0f}ms, "
                f"总={total_ms:.0f}ms, tool_calls={len(tool_calls)}"
            )

            if not tool_calls:
                break

            tool_results = self._execute_tools(tool_calls)
            messages.append({
                "role": "assistant",
                "content": full,
                "tool_calls": tool_calls,
            })
            messages.extend(tool_results)

    # ─── 内部辅助 ───────────────────────────────────────────────────────────────

    def _build_kwargs(
        self, messages: list[dict], use_tools: bool, stream: bool
    ) -> dict:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
        }
        if self._max_tokens > 0:
            kwargs["max_tokens"] = self._max_tokens
            if not stream:
                kwargs["max_completion_tokens"] = self._max_tokens
        if use_tools:
            from rag_framework.llm.tool_registry import get_tool_definitions
            tools = get_tool_definitions()
            if tools:
                kwargs["tools"] = tools
        return kwargs

    async def shutdown(self) -> None:
        """关闭底层 HTTP 客户端。"""
        await self.client.close()

    @staticmethod
    def _execute_tools(tool_calls: list[dict]) -> list[dict]:
        from rag_framework.llm.tool_registry import execute_tool
        results = []
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            try:
                func_args = json.loads(tc["function"]["arguments"])
                result = execute_tool(func_name, func_args)
            except Exception as e:
                result = f"Error: {e}"
            results.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "content": str(result),
            })
        return results


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_openai_llm(
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    backend: str = "openai",
    max_tokens: int = 512,
    max_concurrent: int = 3,
) -> OpenAILLMClient:
    return OpenAILLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        backend=backend,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
    )


register_llm("openai", _create_openai_llm)
register_llm("minimax", _create_openai_llm)
register_llm("ollama", _create_openai_llm)
