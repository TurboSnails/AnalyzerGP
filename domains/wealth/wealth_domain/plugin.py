"""
WealthDomainPlugin 实现

集中全球资产配置领域专用的所有逻辑：
- 系统提示词（投资分析师风格）
- 查询分类器（区分宏观/财报/混合）
- Collection 命名（macro_econ + corp_earnings 双域）
- 中英文金融术语映射
- 重写路由规则
- 兜底回复模板
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from rag_framework.core.logger import retrieval_logger
from rag_framework.domain.base import (
    DomainPlugin,
    CollectionNames,
    QueryRoute,
    DomainPrompts,
)


# ── 金融术语映射（中文 → 英文 keyword）─────────────────────────────────────────

_DEFAULT_TERMS: dict[str, str] = {
    "美联储": "Federal Reserve Fed",
    "加息": "rate hike interest rate increase",
    "降息": "rate cut interest rate decrease",
    "议息": "FOMC meeting interest rate decision",
    "CPI": "CPI consumer price index inflation",
    "PPI": "PPI producer price index",
    "VIX": "VIX volatility index fear gauge",
    "通胀": "inflation",
    "通缩": "deflation",
    "量化宽松": "quantitative easing QE",
    "缩表": "balance sheet reduction tapering",
    "GDP": "GDP gross domestic product",
    "非农就业": "non-farm payrolls NFP",
    "失业率": "unemployment rate",
    "财报": "earnings report financial results",
    "营收": "revenue sales",
    "净利润": "net profit earnings",
    "毛利率": "gross margin",
    "指引": "guidance outlook forecast",
    "分红": "dividend",
    "回购": "share buyback stock repurchase",
    "每股收益": "EPS earnings per share",
    "市盈率": "P/E ratio price earnings",
    "市净率": "P/B ratio price book",
    "NVIDIA": "NVIDIA NVDA",
    "英伟达": "NVIDIA NVDA",
    "比亚迪": "BYD",
    "特斯拉": "Tesla TSLA",
    "苹果": "Apple AAPL",
    "微软": "Microsoft MSFT",
    "谷歌": "Google GOOGL Alphabet",
    "亚马逊": "Amazon AMZN",
    "网格交易": "grid trading",
    "做T": "intraday trading T+0",
    "凯利公式": "Kelly criterion",
    "仓位": "position sizing allocation",
    "止损": "stop loss",
    "止盈": "take profit",
    "波动率": "volatility",
    "夏普比率": "Sharpe ratio",
    "最大回撤": "maximum drawdown",
    "贝塔": "beta",
    "阿尔法": "alpha",
}

# ── Rewrite Router 规则词表 ──────────────────────────────────────────────────

_CONTEXT_REFS = ("它", "这个", "那个", "上面", "之前", "刚才")
_VAGUE_TERMS = ("怎么样", "如何", "怎么看", "有什么影响", "分析一下", "评价一下")


class WealthDomainPlugin(DomainPlugin):
    """全球资产与宏观经济投资分析领域插件。"""

    def __init__(self) -> None:
        self._base_dir = Path(__file__).parent
        self._terms = _DEFAULT_TERMS.copy()

    @property
    def name(self) -> str:
        return "wealth"

    @property
    def system_prompt(self) -> str:
        return (
            "你是一位专业的全球资产配置分析师，精通美股科技股、A股/港股、"
            "宏观经济（美联储政策、CPI/PPI、VIX）以及量化交易策略（网格交易、凯利公式）。"
            "回答基于检索到的财报和宏观数据，严禁凭空编造数字。"
            "涉及仓位计算时，必须调用工具函数得出精确结果，不可口算。"
        )

    @property
    def prompts(self) -> DomainPrompts:
        return DomainPrompts(
            system=self.system_prompt,
            hyde=(
                "你是金融分析专家。以下是一段投资相关文档：\n\n{chunk}\n\n"
                "请生成3个投资者可能会问的问题，这些问题可以通过上述文档内容回答。"
                "要求：直接输出3个问题，每行一个，不要编号，不要额外说明。"
            ),
        )

    # ── Collection 命名 ──────────────────────────────────────────────────────

    def get_collection_names(self) -> CollectionNames:
        """返回默认/统一的 collection 名称（双域共用）。"""
        return CollectionNames(
            parent="wealth_parent",
            child="wealth_child",
            hyde="wealth_hyde",
        )

    def get_macro_collection_names(self) -> CollectionNames:
        """返回宏观经济域的 collection 名称。"""
        return CollectionNames(
            parent="macro_econ_parent",
            child="macro_econ_child",
            hyde="macro_econ_hyde",
        )

    def get_corp_collection_names(self) -> CollectionNames:
        """返回企业财报域的 collection 名称。"""
        return CollectionNames(
            parent="corp_earnings_parent",
            child="corp_earnings_child",
            hyde="corp_earnings_hyde",
        )

    # ── 查询分类 ──────────────────────────────────────────────────────────────

    def classify_query(self, query: str, history: list[dict]) -> QueryRoute:
        """
        投资领域查询分类。

        根据关键词特征判断最适合的召回策略。
        """
        q = query.lower()

        # 宏观关键词
        macro_keywords = [
            "fed", "federal reserve", "美联储", "加息", "降息", "cpi", "ppi",
            "vix", "通胀", "通缩", "gdp", "非农", "就业", "失业率",
            "量化宽松", "缩表", "fomc", "宏观", "经济",
        ]
        # 财报关键词
        corp_keywords = [
            "earnings", "revenue", "guidance", "outlook", "forecast",
            "财报", "营收", "净利润", "毛利率", "指引", "eps",
            "dividend", "分红", "buyback", "回购", "nvidia", "英伟达",
            "byd", "比亚迪", "tsla", "特斯拉", "aapl", "apple", "苹果",
        ]

        has_macro = any(kw in q for kw in macro_keywords)
        has_corp = any(kw in q for kw in corp_keywords)

        if has_macro and has_corp:
            # 混合问题：优先 dense + bm25 双路，权重均衡
            return QueryRoute(
                text=query, type="mixed", weight=0.85,
                routes=["dense", "bm25", "hyde"],
            )
        if has_macro:
            return QueryRoute(
                text=query, type="macro", weight=0.80,
                routes=["dense", "bm25"],
            )
        if has_corp:
            return QueryRoute(
                text=query, type="corp", weight=0.80,
                routes=["dense", "bm25"],
            )

        # 默认语义检索
        return QueryRoute(
            text=query, type="semantic", weight=0.90,
            routes=["dense", "hyde"],
        )

    # ── 术语映射 ──────────────────────────────────────────────────────────────

    def get_term_mapping(self) -> dict[str, str]:
        """返回中文 → 英文金融术语映射。"""
        return self._terms.copy()

    # ── Rewrite Router ────────────────────────────────────────────────────────

    def rewrite_router_rules(self, query: str, history: list[dict]) -> int | None:
        """
        投资领域 Rewrite Router。

        返回 rewrite level：
          0 = 不 rewrite（明确股票代码/指标名，可直接命中）
          1 = 规则扩展（命中中文术语映射，翻译为英文关键词）
          2 = LLM rewrite（含代词/模糊词/长复合句）
        """
        # Level 2：含代词引用、模糊表达、长复合句
        if history and any(w in query for w in _CONTEXT_REFS):
            return 2
        if any(w in query for w in _VAGUE_TERMS) and len(query) >= 20:
            return 2
        if len(query) >= 30:
            return 2

        # Level 1：命中中文术语映射
        if any(term in query for term in self._terms):
            return 1

        # Level 0：含明确股票代码或财务指标缩写
        if re.search(r"\b[A-Z]{1,5}\b", query):
            return 0
        if any(kw in query.upper() for kw in ["CPI", "PPI", "GDP", "VIX", "EPS", "PE", "PB", "FOMC"]):
            return 0

        return 0

    # ── 兜底回复 ──────────────────────────────────────────────────────────────

    def fallback_response(self, reason: str = "low_confidence") -> str:
        templates = {
            "low_confidence": (
                "【知识库提示】本次问题在投资分析知识库中未找到强相关内容。"
                "请直接回复用户：『抱歉，当前知识库未覆盖该问题，"
                "建议提供更具体的公司名称、指标或时间范围。』"
                "不要凭通用知识展开回答。"
            ),
            "no_results": (
                "【知识库提示】检索引擎未返回任何文档。"
                "请直接回复用户：『抱歉，知识库目前没有相关资料，"
                "请稍后重试或换个问题。』"
            ),
            "out_of_scope": (
                "抱歉，这个问题超出了我当前投资知识库的覆盖范围。"
            ),
        }
        return templates.get(reason, templates["out_of_scope"])
