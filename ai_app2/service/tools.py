"""
工具定义模块。

复用 rag_framework.llm.tool_registry 注册工具，供 LLM agent 调用。
Schema 由注册中心统一管理，OpenAILLMClient 自动注入。
"""
from __future__ import annotations

from rag_framework.llm.tool_registry import register_tool


def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积"""
    return a * b


register_tool(
    name="multiply",
    func=multiply,
    description="计算两个数字的乘积",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
)

# 导出工具名列表（供文档/测试引用）
TOOL_NAMES = ["multiply"]
