"""
Context Compressor — 自适应上下文压缩与结构化摘要。

能力：
1. compress_context   : 当上下文超长时，压缩保留关键信息
2. extract_key_facts  : 从上下文中提取结构化关键事实（JSON）
3. build_prompt_context: 组装最终 LLM prompt，支持层次化引用
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ai_app3.core.config import OPENAI_API_KEY, DEFAULT_TOKEN_BUDGET
from ai_app3.core.logger import compress_logger

_llm = ChatOpenAI(
    model="MiniMax-M2.7",
    base_url="https://api.minimaxi.com/v1",
    api_key=OPENAI_API_KEY or "",
    temperature=0.1,
)


def _estimate_tokens(text: str) -> int:
    cn_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
    other_chars = len(text) - cn_chars
    return int(cn_chars * 1.5 + other_chars * 0.5)


def compress_context(query: str, context: str, target_budget: int = DEFAULT_TOKEN_BUDGET // 2) -> str:
    """
    若上下文 token 数超过 target_budget，则压缩为保留关键信息的摘要。
    """
    tokens = _estimate_tokens(context)
    if tokens <= target_budget:
        compress_logger.debug(f"上下文无需压缩: tokens={tokens}")
        return context

    prompt = [
        SystemMessage(content="你是一个上下文压缩专家。请压缩以下文档，保留与查询最相关的技术细节、代码示例和关键结论。输出纯文本，不要解释。"),
        HumanMessage(
            content=f"查询：{query}\n\n"
                    f"原始上下文（{tokens} tokens，目标压缩到 {target_budget} tokens）：\n\n"
                    f"{context[:4000]}\n\n"
                    f"请输出压缩后的上下文："
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        compressed = (resp.content or context).strip()
        new_tokens = _estimate_tokens(compressed)
        compress_logger.info(f"上下文压缩: {tokens} → {new_tokens} tokens")
        return compressed
    except Exception as e:
        compress_logger.error(f"压缩失败: {e}，返回截断上下文")
        # 保守回退：截断到目标预算对应的字符数（粗略估算）
        approx_chars = int(target_budget / 0.8)
        return context[:approx_chars] + "\n...[内容截断]"


def extract_key_facts(context: str) -> list[dict]:
    """
    从上下文中提取结构化关键事实，用于生成带引用的回答。
    返回 [{"fact": str, "source": str, "relevance": float}, ...]
    """
    prompt = [
        SystemMessage(content="你是一个信息提取专家。从文档中提取最多5条关键事实，输出合法 JSON 数组。"),
        HumanMessage(
            content=f'请提取关键事实，格式：[{{"fact": "...", "source": "段落摘要", "relevance": 0.95}}]\n\n'
                    f"文档：\n{context[:3000]}"
        ),
    ]
    try:
        resp = _llm.invoke(prompt)
        raw = resp.content or "[]"
        raw = re.sub(r"```json|```", "", raw).strip()
        facts = json.loads(raw)
        if not isinstance(facts, list):
            facts = []
    except Exception as e:
        compress_logger.warning(f"extract_key_facts 解析失败: {e}")
        facts = []

    compress_logger.info(f"提取关键事实: {len(facts)} 条")
    return facts


def build_prompt_context(query: str, context: str, facts: list[dict] | None = None) -> str:
    """
    构建带层次化引用的最终 prompt 上下文。
    """
    parts = ["【参考资料】\n" + context]
    if facts:
        parts.append("\n【关键事实提炼】")
        for i, f in enumerate(facts[:5], 1):
            parts.append(f"{i}. {f.get('fact', '')} (来源: {f.get('source', '文档')})")
    parts.append(f"\n【用户问题】\n{query}")
    return "\n".join(parts)
