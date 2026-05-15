"""
Local Transformers LLM Client

使用本地 transformers 模型（如 Qwen2.5-1.5B-Instruct）进行 LLM 推理。
支持流式/非流式对话、摘要生成。

首次调用时懒加载模型，之后复用同一实例。
所有同步推理通过 asyncio.to_thread() 放到线程池执行，避免阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncIterator

from rag_framework.core.factories import register_llm
from rag_framework.core.lifecycle import Warmupable
from rag_framework.core.logger import ai_client_logger
from rag_framework.llm.base import LLMClient


class LocalLLMClient(LLMClient, Warmupable):
    """
    本地 transformers LLM 客户端。

    使用 Hugging Face transformers 直接在本地运行模型推理，
    无需外部 API（ollama / openai / minimax）。
    """

    def __init__(
        self,
        model_path: str,
        max_tokens: int = 512,
        max_concurrent: int = 1,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self._model_path = model_path
        self._max_tokens = max_tokens
        self._max_concurrent = max_concurrent
        self._device_override = device
        self._dtype_override = dtype
        self._semaphore: asyncio.Semaphore | None = None

        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._lock = threading.Lock()

        ai_client_logger.info(
            f"LocalLLMClient 初始化: model_path={model_path}, "
            f"max_tokens={max_tokens}, max_concurrent={max_concurrent}"
        )

    # ─── 属性 ───────────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        return "local"

    @property
    def model(self) -> str:
        return self._model_path.split("/")[-1]

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._max_concurrent)
        return self._semaphore

    # ─── 懒加载 ─────────────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        with self._lock:
            if self._tokenizer is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self._device_override:
                device = self._device_override
            elif torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

            if self._dtype_override:
                dtype = getattr(torch, self._dtype_override)
            elif device in ("cuda", "mps"):
                dtype = torch.float16
            else:
                dtype = torch.float32

            ai_client_logger.info(
                f"本地 LLM 加载中: path={self._model_path!r}, device={device}, dtype={dtype}"
            )
            t0 = time.monotonic()

            tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            # 使用 from_pretrained 加载权重（low_cpu_mem_usage=False 避免延迟加载）
            # 但 transformers 对缺失的 tied weight 仍会创建 meta tensor，
            # 因此加载后需显式 materialize，再调用 .to(device)
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            model.tie_weights()

            # 处理残留的 meta tensor（如缺失的 lm_head.weight）
            # tie_weights() 理论上会共享 embed_tokens.weight，但某些版本 transformers
            # 不会更新 _parameters 中的 meta tensor 引用，导致 .to(device) 报错
            for name, param in list(model.named_parameters()):
                if param.device.type == "meta":
                    # 对 tied embeddings 的 lm_head，直接共享 embed_tokens 权重
                    if "lm_head.weight" in name and getattr(model.config, "tie_word_embeddings", False):
                        embed = model.get_input_embeddings()
                        if embed is not None and embed.weight.device.type != "meta":
                            model.lm_head.weight = embed.weight
                            continue
                    # 其他 meta tensor：用空张量 materialize（通常不会发生）
                    parent_name = name.rsplit(".", 1)[0]
                    attr_name = name.rsplit(".", 1)[1]
                    module = model.get_submodule(parent_name)
                    empty = torch.empty_like(param, device="cpu")
                    module._parameters[attr_name] = torch.nn.Parameter(
                        empty, requires_grad=param.requires_grad
                    )

            model = model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device

            ai_client_logger.info(
                f"本地 LLM 加载完成: device={device}, dtype={dtype}, "
                f"耗时={time.monotonic() - t0:.1f}s"
            )

    # ─── 非流式 ─────────────────────────────────────────────────────────────────

    async def chat(self, messages: list[dict], use_tools: bool = False) -> str:
        self._ensure_loaded()
        async with self._get_semaphore():
            text = await asyncio.to_thread(self._generate_sync, messages)
        return text

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
        """非流式工具增强对话（本地模型暂不支持工具调用，降级为普通 chat）。"""
        return await self.chat(messages, use_tools=False)

    # ─── 流式 ───────────────────────────────────────────────────────────────────

    async def chat_stream(
        self, messages: list[dict], use_tools: bool = False
    ) -> AsyncIterator[str]:
        self._ensure_loaded()
        t0 = time.perf_counter()
        first_token_ms = None
        char_count = 0

        async with self._get_semaphore():
            async for chunk in self._generate_stream(messages):
                if chunk:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - t0) * 1000
                    char_count += len(chunk)
                    yield chunk

        total_ms = (time.perf_counter() - t0) * 1000
        gen_ms = total_ms - (first_token_ms or 0)
        cps = (char_count / gen_ms * 1000) if gen_ms > 0 else 0
        ai_client_logger.info(
            f"[local] LLM 流式完成: TTFT={first_token_ms or 0:.0f}ms, "
            f"生成={gen_ms:.0f}ms, 总={total_ms:.0f}ms, "
            f"输出={char_count}字符 ({cps:.0f}字符/秒)"
        )

    async def run_agent_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """流式工具增强对话（本地模型暂不支持工具调用，降级为普通 chat_stream）。"""
        async for chunk in self.chat_stream(messages, use_tools=False):
            yield chunk

    # ─── 内部同步生成 ───────────────────────────────────────────────────────────

    def _generate_sync(self, messages: list[dict]) -> str:
        import torch
        from transformers import TextIteratorStreamer

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            streamer=streamer,
            max_new_tokens=self._max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        result_parts = []
        for new_text in streamer:
            result_parts.append(new_text)

        thread.join()
        return "".join(result_parts)

    async def _generate_stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """异步包装 TextIteratorStreamer 的同步迭代。"""
        import torch
        from transformers import TextIteratorStreamer

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=inputs.input_ids,
            streamer=streamer,
            max_new_tokens=self._max_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        loop = asyncio.get_running_loop()
        iterator = iter(streamer)

        def _next_safe():
            """安全地调用 next，将 StopIteration 转为 None 标记。"""
            try:
                return next(iterator)
            except StopIteration:
                return None

        try:
            while True:
                new_text = await loop.run_in_executor(None, _next_safe)
                if new_text is None:
                    break
                if new_text:
                    yield new_text
        finally:
            thread.join()

    async def warmup(self) -> None:
        """异步预热：加载 tokenizer 和 model。"""
        await asyncio.to_thread(self._ensure_loaded)


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_local_llm(
    model_path: str = "",
    max_tokens: int = 512,
    max_concurrent: int = 1,
    **_ignored: object,
) -> LocalLLMClient:
    # 默认使用 CPU + float32，避免 MPS (macOS Metal) 上 Qwen 模型的 dtype 崩溃
    return LocalLLMClient(
        model_path=model_path,
        max_tokens=max_tokens,
        max_concurrent=max_concurrent,
        device="cpu",
        dtype="float32",
    )


register_llm("local", _create_local_llm)
