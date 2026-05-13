"""
LLM Client 抽象基类

支持流式/非流式对话、工具调用、摘要生成。
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any


class LLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @property
    @abstractmethod
    def backend(self) -> str:
        """后端标识，如 minimax / ollama / openai。"""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """当前使用的模型名。"""
        ...

    @abstractmethod
    async def chat(self, messages: list[dict], use_tools: bool = False) -> str:
        """
        非流式对话。

        Args:
            messages: OpenAI 格式的消息列表
            use_tools: 是否启用工具调用

        Returns:
            AI 生成的文本回复
        """
        ...

    @abstractmethod
    async def chat_stream(
        self, messages: list[dict], use_tools: bool = False
    ) -> AsyncIterator[str]:
        """
        流式对话，yield 每个 token chunk。

        Args:
            messages: OpenAI 格式的消息列表
            use_tools: 是否启用工具调用
        """
        ...

    @abstractmethod
    async def summarize(self, history: list[dict]) -> str:
        """
        将对话历史压缩为摘要。

        Args:
            history: 不含 system prompt 的对话历史

        Returns:
            摘要文本
        """
        ...

    @abstractmethod
    async def run_agent(self, messages: list[dict]) -> str:
        """
        工具增强型对话（非流式，多轮工具调用）。

        Args:
            messages: 完整消息列表

        Returns:
            最终文本回复
        """
        ...

    @abstractmethod
    async def run_agent_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """
        流式工具增强型对话。

        在流式响应中直接收集 tool_calls，执行工具后继续流式返回。
        """
        ...
