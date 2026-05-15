"""
Rule-Based Query Rewriter

利用领域术语映射对中文查询做词典增强：将命中的中文术语对应的英文
追加到 query 末尾，提升 BM25 的英文命中率。

输出 2 条路由：
  [0] original — 原始 query，weight=1.0
  [1] keyword  — 追加英文术语，weight=0.85（仅在有命中时输出）
"""
from __future__ import annotations

import time

from rag_framework.core.factories import register_rewriter
from rag_framework.core.logger import get_logger
from rag_framework.domain.base import DomainPlugin, QueryRoute
from rag_framework.retrieval.query_rewriter.base import QueryRewriter

_logger = get_logger("rag.rewriter.rule")


class RuleQueryRewriter(QueryRewriter):
    """规则驱动的查询改写器。"""

    def __init__(self, domain: DomainPlugin) -> None:
        self._terms: dict[str, str] = domain.get_term_mapping()

    def rewrite(self, query: str, history: list[dict]) -> list[QueryRoute]:
        t0 = time.monotonic()
        original = QueryRoute(text=query, type="original", weight=1.0)

        matched_en = [en for zh, en in self._terms.items() if zh in query]
        if not matched_en:
            _logger.debug(f"规则改写 ({(time.monotonic()-t0)*1000:.0f}ms): 无命中术语，返回原始 query={query!r}")
            return [original]

        expanded = query + " " + " ".join(matched_en)
        elapsed = time.monotonic() - t0
        _logger.info(f"规则改写 ({elapsed*1000:.0f}ms): {query!r} + {matched_en} → {expanded!r}")

        return [
            original,
            QueryRoute(
                text=expanded,
                type="keyword",
                weight=0.85,
                routes=["bm25", "dense"],
            ),
        ]


# ─── 工厂函数与自注册 ──────────────────────────────────────────
def _create_rule_rewriter(domain: DomainPlugin) -> RuleQueryRewriter:
    return RuleQueryRewriter(domain=domain)


register_rewriter("rule", _create_rule_rewriter)
