"""
工具注册中心

统一管理可用工具的定义与执行函数，支持动态注册。
"""
from __future__ import annotations

from typing import Callable, Any

# {name: callable}
_TOOL_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def register_tool(name: str, func: Callable[..., Any], description: str,
                  parameters: dict | None = None) -> None:
    """
    注册一个工具函数。

    Args:
        name: 工具名（LLM 调用时使用的 function name）
        func: 实际执行的 Python 函数
        description: 工具描述
        parameters: JSON Schema 格式的参数定义
    """
    _TOOL_FUNCTIONS[name] = func
    _TOOL_SCHEMAS[name] = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}, "required": []},
        },
    }


_TOOL_SCHEMAS: dict[str, dict] = {}


def get_tool_definitions() -> list[dict]:
    """获取所有已注册工具的 OpenAI 格式定义列表。"""
    return list(_TOOL_SCHEMAS.values())


def execute_tool(name: str, args: dict) -> Any:
    """执行指定工具。"""
    if name not in _TOOL_FUNCTIONS:
        return f"Unknown tool: {name}"
    return _TOOL_FUNCTIONS[name](**args)


def list_tools() -> list[str]:
    """列出所有已注册工具名。"""
    return list(_TOOL_FUNCTIONS.keys())


# ─── 默认工具 ─────────────────────────────────────────────────────────────────

def multiply(a: int, b: int) -> int:
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
