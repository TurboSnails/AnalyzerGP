from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积"""
    return a * b


TOOLS = [multiply]
TOOL_MAP = {t.name: t for t in TOOLS}
