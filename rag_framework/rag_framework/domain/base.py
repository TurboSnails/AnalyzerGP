"""
DomainPlugin 抽象基类

领域插件负责注入领域特定的知识：系统提示、查询分类、Collection 命名、
评测集、术语映射等。框架通过此接口与领域解耦。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectionNames:
    """领域知识库的 Collection 名称集合。"""
    parent: str
    child: str
    hyde: str


@dataclass(frozen=True)
class QueryRoute:
    """查询分类结果，决定召回路径与权重。"""
    text: str
    type: str = "original"          # original | semantic | keyword | api
    weight: float = 1.0
    routes: list[str] = field(default_factory=lambda: ["dense", "hyde", "bm25"])


@dataclass(frozen=True)
class DomainPrompts:
    """领域相关的 prompt 模板。"""
    system: str = ""
    hyde: str = (
        "你是该领域的专家。以下是一段文档：\n\n{chunk}\n\n"
        "请生成3个用户可能会问的问题，这些问题可以通过上述文档内容回答。\n"
        "要求：直接输出3个问题，每行一个，不要编号，不要额外说明。"
    )
    summarize: str = "请总结以下对话的关键信息，用于后续对话参考。"


class DomainPlugin(ABC):
    """
    领域插件抽象基类。

    每个领域（Android、投资、医学等）实现一个子类，
    注册到 Registry 后框架自动加载。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """领域唯一标识，如 'android'、'investment'。"""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """系统提示词，注入 LLM 的 system 角色。"""
        ...

    @property
    def prompts(self) -> DomainPrompts:
        """Prompt 模板集合，子类可覆盖。"""
        return DomainPrompts(system=self.system_prompt)

    @abstractmethod
    def classify_query(self, query: str, history: list[dict]) -> QueryRoute:
        """
        对查询进行分类，决定召回策略。

        Args:
            query: 用户原始查询
            history: 会话历史（可选）

        Returns:
            QueryRoute，包含 type/weight/routes
        """
        ...

    @abstractmethod
    def get_collection_names(self) -> CollectionNames:
        """返回该领域在向量数据库中的 collection 名称。"""
        ...

    def get_hyde_prompt(self, chunk: str) -> str:
        """
        为 chunk 生成 HyDE 问题生成的 prompt。

        默认使用 prompts.hyde 模板，子类可覆盖。
        """
        return self.prompts.hyde.format(chunk=chunk)

    def get_eval_dataset(self) -> list[dict]:
        """
        返回该领域的评测集数据。

        默认返回空列表（未配置评测集的领域）。
        """
        return []

    def get_term_mapping(self) -> dict[str, str]:
        """
        返回中文 → 英文术语映射，用于 Query Rewriter 规则扩展。

        默认返回空 dict。
        """
        return {}

    def rewrite_router_rules(self, query: str, history: list[dict]) -> int | None:
        """
        领域专用的 Rewrite Router 规则。

        返回 rewrite level（0/1/2）或 None（使用框架默认规则）。
        """
        return None

    def fallback_response(self, reason: str = "low_confidence") -> str:
        """
        低置信度或检索失败时的兜底回复模板。

        Args:
            reason: "low_confidence" | "no_results" | "out_of_scope"
        """
        templates = {
            "low_confidence": (
                "抱歉，这个问题在当前知识库中未找到强相关内容，"
                "建议提供更多上下文或换个问法。"
            ),
            "no_results": (
                "抱歉，知识库目前没有相关资料，请稍后重试或换个问题。"
            ),
            "out_of_scope": (
                "抱歉，这个问题超出了我当前知识库的覆盖范围。"
            ),
        }
        return templates.get(reason, templates["out_of_scope"])

    def estimate_tokens(self, text: str) -> int:
        """
        估算文本 token 数。

        默认策略（中英文混合）：
          - 中文字符：~1.5 token/字
          - 英文/数字/符号：~0.5 token/字符

        子类可覆盖为更精确的实现。
        """
        import re
        cn_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other_chars = len(text) - cn_chars
        return int(cn_chars * 1.5 + other_chars * 0.5)
