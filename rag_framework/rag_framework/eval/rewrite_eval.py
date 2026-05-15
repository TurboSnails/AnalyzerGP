"""
Rewrite 效果评测器（Rewrite Evaluation）

评估 query rewrite 是否真正提升了检索质量。
核心对比指标：
  - rewrite 前 vs rewrite 后的 recall@5 / hit@1 / mrr
  - 每条 query 的 delta（变化量）
  - 提升/下降/持平 的数量统计
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.eval.hit_judge import ground_truth_ids
from rag_framework.eval.metrics import reciprocal_rank, recall_at_k, hit_at_k
from rag_framework.retrieval.base import RetrievedDoc
from rag_framework.retrieval.fusion import HybridConfig


@dataclass
class RewriteComparison:
    """单条 query 的 rewrite 前后对比结果。"""
    query: str = ""
    rewritten: str = ""
    before_recall: float = 0.0
    after_recall: float = 0.0
    before_hit1: float = 0.0
    after_hit1: float = 0.0
    before_mrr: float = 0.0
    after_mrr: float = 0.0
    delta_recall: float = 0.0
    delta_hit1: float = 0.0
    delta_mrr: float = 0.0
    is_improved: bool = False
    is_degraded: bool = False


async def _retrieve_raw(
    query: str,
    container: RAGContainer,
    config: HybridConfig | None = None,
) -> list[RetrievedDoc]:
    """
    直接对原始 query 执行检索（不走 rewrite/classify）。
    同步调用，用于评测。
    """
    cfg = config or HybridConfig()
    old_cfg = container.retriever._cfg
    try:
        container.retriever._cfg = cfg
        result = await container.retriever.retrieve([QueryRoute(text=query)])
    finally:
        container.retriever._cfg = old_cfg
    return result.docs


async def _retrieve_rewritten(
    query: str,
    container: RAGContainer,
    config: HybridConfig | None = None,
) -> tuple[str, list[RetrievedDoc]]:
    """
    先走 rewrite + classify，再检索。
    返回 (rewritten_query, docs)。
    """
    from rag_framework.session.manager import SessionManager

    cfg = config or HybridConfig()
    old_cfg = container.retriever._cfg
    manager = SessionManager(
        store=container.session_store,
        llm=container.llm,
        retriever=container.retriever,
        domain=container.domain,
        settings=container.settings,
        rule_rewriter=container.rule_rewriter,
        llm_rewriter=container.llm_rewriter,
    )
    try:
        container.retriever._cfg = cfg
        routes = manager._build_routes(query, [])
        rewritten = routes[0].text if routes else query
        result = await container.retriever.retrieve(routes)
    finally:
        container.retriever._cfg = old_cfg
    return rewritten, result.docs


def _compute_metrics(rank_list: list[str], gt_ids: set[str]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(rank_list, gt_ids, 5),
        "hit@1": hit_at_k(rank_list, gt_ids, 1),
        "mrr": reciprocal_rank(rank_list, gt_ids),
    }


async def compare_rewrite_single(
    item: dict,
    container: RAGContainer,
    config: HybridConfig | None = None,
) -> RewriteComparison:
    """
    对单条 query 做 rewrite 前后对比。
    """
    query = item["query"]
    cfg = config or HybridConfig()

    # 1. 获取 ground truth
    # 先做一次完整检索拿到所有 chunks 用于 hit_judge
    all_docs = await _retrieve_raw(query, container, cfg)
    chunks_raw = [{"id": d.id, "text": d.text} for d in all_docs]
    gt_ids = ground_truth_ids(item, chunks_raw)

    # 2. 原始 query 检索（不走 rewrite）
    docs_before = await _retrieve_raw(query, container, cfg)
    rank_before = [d.id for d in docs_before]
    metrics_before = _compute_metrics(rank_before, gt_ids)

    # 3. rewrite 后检索
    rewritten, docs_after = await _retrieve_rewritten(query, container, cfg)
    rank_after = [d.id for d in docs_after]
    metrics_after = _compute_metrics(rank_after, gt_ids)

    delta_recall = metrics_after["recall@5"] - metrics_before["recall@5"]
    delta_hit1 = metrics_after["hit@1"] - metrics_before["hit@1"]
    delta_mrr = metrics_after["mrr"] - metrics_before["mrr"]

    return RewriteComparison(
        query=query,
        rewritten=rewritten,
        before_recall=metrics_before["recall@5"],
        after_recall=metrics_after["recall@5"],
        before_hit1=metrics_before["hit@1"],
        after_hit1=metrics_after["hit@1"],
        before_mrr=metrics_before["mrr"],
        after_mrr=metrics_after["mrr"],
        delta_recall=delta_recall,
        delta_hit1=delta_hit1,
        delta_mrr=delta_mrr,
        is_improved=(delta_recall > 0 or delta_hit1 > 0 or delta_mrr > 0),
        is_degraded=(delta_recall < 0 or delta_hit1 < 0 or delta_mrr < 0),
    )


async def run_rewrite_eval(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    config: HybridConfig | None = None,
    verbose: bool = True,
) -> dict:
    """
    批量评测 rewrite 效果，输出对比报告。
    """
    if container is None:
        container = RAGContainer.from_settings(get_settings())

    if dataset_path is None:
        dataset = container.domain.get_eval_dataset()
        if not dataset:
            eval_logger.warning("Domain 未提供评测集，返回空报告")
            return {}
    else:
        import json
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        dataset = [item for item in dataset if "query" in item]

    total = len(dataset)
    comparisons: list[RewriteComparison] = []

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Rewrite Evaluation   共 {total} 条")
        print(f"{'─'*70}\n")

    for i, item in enumerate(dataset, 1):
        comp = await compare_rewrite_single(item, container, config)
        comparisons.append(comp)

        if verbose:
            marker = "📈" if comp.is_improved else ("📉" if comp.is_degraded else "➖")
            print(
                f"[{i:02d}/{total}] {marker} recall {comp.before_recall:.0%} → "
                f"{comp.after_recall:.0%}  hit@1 {comp.before_hit1:.0%} → {comp.after_hit1:.0%}"
            )
            print(f"  原始: {comp.query[:60]}")
            if comp.rewritten != comp.query:
                print(f"  改写: {comp.rewritten[:60]}")
            print()

    improved = sum(1 for c in comparisons if c.is_improved)
    degraded = sum(1 for c in comparisons if c.is_degraded)
    unchanged = total - improved - degraded

    avg_delta_recall = sum(c.delta_recall for c in comparisons) / total if total else 0.0
    avg_delta_hit1 = sum(c.delta_hit1 for c in comparisons) / total if total else 0.0
    avg_delta_mrr = sum(c.delta_mrr for c in comparisons) / total if total else 0.0

    report = {
        "total": total,
        "improved": improved,
        "degraded": degraded,
        "unchanged": unchanged,
        "avg_delta_recall": round(avg_delta_recall, 4),
        "avg_delta_hit1": round(avg_delta_hit1, 4),
        "avg_delta_mrr": round(avg_delta_mrr, 4),
        "details": [
            {
                "query": c.query,
                "rewritten": c.rewritten,
                "before_recall": c.before_recall,
                "after_recall": c.after_recall,
                "before_hit1": c.before_hit1,
                "after_hit1": c.after_hit1,
                "before_mrr": c.before_mrr,
                "after_mrr": c.after_mrr,
                "delta_recall": c.delta_recall,
                "delta_hit1": c.delta_hit1,
                "delta_mrr": c.delta_mrr,
            }
            for c in comparisons
        ],
    }

    if verbose:
        print(f"{'─'*70}")
        print(f"  Rewrite 综合报告")
        print(f"    提升 {improved} 条 | 下降 {degraded} 条 | 持平 {unchanged} 条")
        print(f"    ΔRecall@5 = {avg_delta_recall:+.4f}")
        print(f"    ΔHit@1    = {avg_delta_hit1:+.4f}")
        print(f"    ΔMRR      = {avg_delta_mrr:+.4f}")
        print(f"{'─'*70}\n")

    return report
