"""
ai_app3 LLM 统一入口。

从 RAGSettings 读取配置，消除各 service / graph 模块中的硬编码 ChatOpenAI。
提供两个级别的实例：
  • get_chat_llm()      — 主 LLM（对话/生成/评估/压缩）
  • get_rewriter_llm()  — 查询改写/意图分析专用 LLM

注意：ai_app3 基于 LangChain ChatOpenAI，要求后端为 OpenAI 兼容协议。
若 rewriter_llm_backend="local"（transformers 直接加载），建议改用 ollama
提供兼容端点，或在 rag_framework 层完成改写（见 container.rewriter_llm）。
"""
from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from rag_framework.core.config import get_settings
from rag_framework.core.logger import get_logger

_logger = get_logger("ai_app3.llm_provider")
_settings = get_settings()


def get_chat_llm(temperature: float = 0.2, **kwargs: Any) -> ChatOpenAI:
    """
    获取主 LLM（对话/生成/评估/压缩）。

    配置来源：RAGSettings 的 llm_* 字段，默认对应远程 minimax。
    """
    return ChatOpenAI(
        model=_settings.llm_model,
        base_url=_settings.llm_base_url,
        api_key=_settings.resolved_llm_api_key,
        temperature=temperature,
        **kwargs,
    )


def get_rewriter_llm(temperature: float = 0.2, **kwargs: Any) -> ChatOpenAI:
    """
    获取查询改写/意图分析专用 LLM。

    配置来源：RAGSettings 的 rewriter_llm_* 字段。
    若 rewriter_llm_backend 为 "local" 且缺少兼容端点，
    则回退到主 LLM 并记录警告（保证可用性）。
    """
    backend = _settings.resolved_rewriter_llm_backend
    base_url = _settings.rewriter_llm_base_url
    api_key = _settings.rewriter_llm_api_key
    model = _settings.rewriter_llm_model

    # local backend 默认无 OpenAI 兼容端点，回退到主 LLM
    if backend == "local" and not base_url:
        _logger.warning(
            f"rewriter_llm_backend='local' 但缺少 OpenAI 兼容端点，"
            f"回退到主 LLM (backend={_settings.llm_backend})"
        )
        return get_chat_llm(temperature=temperature, **kwargs)

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )
