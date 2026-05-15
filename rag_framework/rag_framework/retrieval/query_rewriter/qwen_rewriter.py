"""
Local Qwen Query Rewriter

使用本地 Qwen2.5-1.5B-Instruct 模型将用户查询改写为 2-3 条检索 query。
首次调用时懒加载模型（避免拖慢 FastAPI 启动），之后复用同一实例。
"""
from __future__ import annotations

import json
import re
import threading
import time

from rag_framework.core.factories import register_rewriter
from rag_framework.core.lifecycle import Warmupable
from rag_framework.core.logger import get_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.retrieval.query_rewriter.base import QueryRewriter

_logger = get_logger("rag.rewriter.qwen")

_SYSTEM_PROMPT = (
    "你是 Android 开发知识库的检索优化专家。\n"
    "任务：根据当前问题（以及对话上下文），生成 2~3 条适合向量检索的 query。\n"
    "输出格式（严格遵守）：只输出一个 JSON 数组，每个元素必须是字符串，不加任何解释。\n"
    '正确示例：["Handler 内存泄漏怎么解决", "Android Handler memory leak fix", "WeakReference 防泄漏"]\n'
    "错误示例（禁止）：带编号的列表、嵌套数组、Markdown 代码块"
)


class QwenQueryRewriter(QueryRewriter, Warmupable):
    """
    本地 Qwen2.5-1.5B-Instruct 查询改写器。

    线程安全懒加载：首次 rewrite() 调用时加载模型，之后复用。
    """

    def __init__(self, model_path: str, max_new_tokens: int = 128) -> None:
        self._model_path = model_path
        self._max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._device: str | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        with self._lock:
            if self._tokenizer is not None:
                return

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if torch.cuda.is_available():
                device, dtype = "cuda", torch.float16
            elif torch.backends.mps.is_available():
                device, dtype = "mps", torch.float16
            else:
                device, dtype = "cpu", torch.float32

            _logger.info(f"Qwen 改写器加载中: path={self._model_path!r}, device={device}")
            t0 = time.monotonic()

            tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            # device_map 需要 accelerate 处理设备放置和 tied weight 解析
            model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                dtype=dtype,
                device_map=device,
            )
            # Qwen2 lm_head.weight 是 tied weight，LOAD REPORT 中显示 MISSING 属正常；
            # 但 accelerate 路径不保证自动调用 tie_weights()，须显式绑定，
            # 否则 lm_head 使用随机初始化权重，推理输出为乱码。
            model.tie_weights()
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._device = device

            _logger.info(
                f"Qwen 改写器加载完成: device={device}, dtype={dtype}, "
                f"耗时={time.monotonic()-t0:.1f}s"
            )

    def rewrite(self, query: str, history: list[dict]) -> list[QueryRoute]:
        self._ensure_loaded()

        model_label = f"qwen(local:{self._model_path.split('/')[-1]})"
        _logger.info(f"LLM 改写开始: model={model_label!r}, query={query!r}")
        t0 = time.monotonic()

        messages = self._build_messages(query, history)
        try:
            raw = self._generate(messages)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _logger.warning(
                f"LLM 改写失败 ({elapsed*1000:.0f}ms), model={model_label!r}: {exc}，"
                f"降级返回原始 query"
            )
            return [QueryRoute(text=query, type="original", weight=1.0)]

        elapsed = time.monotonic() - t0
        lines = self._parse_output(raw, query)

        if not lines:
            _logger.warning(
                f"LLM 改写返回空结果 ({elapsed*1000:.0f}ms), model={model_label!r}，"
                f"raw={raw!r}，降级返回原始 query"
            )
            return [QueryRoute(text=query, type="original", weight=1.0)]

        routes: list[QueryRoute] = [QueryRoute(text=query, type="original", weight=1.0)]
        for i, line in enumerate(lines[:3]):
            routes.append(
                QueryRoute(
                    text=line,
                    type="semantic",
                    weight=round(0.90 - i * 0.10, 2),
                    routes=["dense", "bm25"],
                )
            )

        _logger.info(
            f"LLM 改写完成 ({elapsed*1000:.0f}ms): model={model_label!r}, "
            f"{query!r} → {len(routes)-1} 条扩写: {[r.text for r in routes[1:]]}"
        )
        return routes

    def _build_messages(self, query: str, history: list[dict]) -> list[dict]:
        ctx_lines: list[str] = []
        for msg in (history[-4:] if len(history) > 4 else history):
            role = "用户" if msg.get("role") == "user" else "AI"
            ctx_lines.append(f"{role}: {str(msg.get('content', ''))[:80]}")

        user_content = (
            f"对话上下文：\n{''.join(ctx_lines)}\n\n【当前问题】\n{query}"
            if ctx_lines
            else f"【当前问题】\n{query}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: list[dict]) -> str:
        import torch

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,   # 防止小模型陷入 token 重复循环输出乱码
            )

        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _parse_output(raw: str, fallback: str) -> list[str]:
        # 去除可能的 markdown 代码块
        cleaned = re.sub(r"```json|```", "", raw).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                result = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
                if result:
                    return result
        except json.JSONDecodeError:
            pass
        # 降级：按行解析
        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            # 提取 Markdown 链接文本 [text](url) → text
            ln = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", ln)
            # 去除编号前缀：1. / 1、 / 1) 以及 [1. ...]
            ln = re.sub(r"^\s*\[?\s*[\d]+[.、)]\s*", "", ln).strip()
            # 去除符号前缀
            ln = ln.lstrip("-•·").strip()
            if ln:
                lines.append(ln)
        return lines

    async def warmup(self) -> None:
        """异步预热：加载 Qwen 模型。"""
        import asyncio
        await asyncio.to_thread(self._ensure_loaded)


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_qwen_rewriter(
    model_path: str = "",
    max_new_tokens: int = 128,
) -> QwenQueryRewriter:
    from rag_framework.core.config import _resolve_rewriter_path
    path = model_path or _resolve_rewriter_path()
    return QwenQueryRewriter(model_path=path, max_new_tokens=max_new_tokens)


register_rewriter("qwen", _create_qwen_rewriter)
register_rewriter("local_rewriter", _create_qwen_rewriter)
