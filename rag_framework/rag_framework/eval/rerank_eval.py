"""
Rerank 效果评测器（Rerank Evaluation）

验证 CrossEncoder 是否真的把正确 chunk 排到了 top1。

核心指标：
  - rerank_win:   正确 chunk 从非 top1 被推到 top1 的次数
  - rerank_loss:  正确 chunk 从 top1 被挤下去的次数
  - rerank_hold:  原本 top1 且 rerank 后仍是 top1 的次数
  - avg_rank_delta: rerank 前后正确 chunk 的平均排名变化（负数=上升）
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.eval.hit_judge import ground_truth_ids
from rag_framework.rerank.base import RankedDoc
from rag_framework.retrieval.base import RetrievedDoc
from rag_framework.retrieval.fusion import HybridConfig, HybridRetriever


@dataclass
class RerankComparison:
    """单条 query 的 rerank 前后对比。"""
    query: str = ""
    gt_ids: set[str] = field(default_factory=set)
    before_rank: int = 999          # rerank 前第一个 gt 的位置（1-based）
    after_rank: int = 999           # rerank 后第一个 gt 的位置
    before_top1_id: str = ""        # rerank 前 top1 的 chunk id
    after_top1_id: str = ""         # rerank 后 top1 的 chunk id
    top1_is_hit_before: bool = False
    top1_is_hit_after: bool = False
    rank_delta: int = 0             # after - before（负数=上升）
    is_win: bool = False            # 非 top1 → top1
    is_loss: bool = False           # top1 → 非 top1
    is_hold: bool = False           # top1 → top1
    ce_scores: list[tuple[str, float]] = field(default_factory=list)  # (id, ce_score)


def _find_first_gt_rank(rank_list: list[str], gt_ids: set[str]) -> int:
    for idx, doc_id in enumerate(rank_list, start=1):
        if doc_id in gt_ids:
            return idx
    return 999


async def _retrieve_rerank_stages(
    query: str,
    container: RAGContainer,
    config: HybridConfig | None = None,
) -> tuple[list[RankedDoc], list[RankedDoc], set[str]]:
    """
    获取 rerank 前（RRF 后）和 rerank 后的候选列表，以及 gt_ids。

    Returns:
        (before_rerank_candidates, after_rerank_candidates, gt_ids)
    """
    cfg = config or HybridConfig()
    retriever: HybridRetriever = container.retriever  # type: ignore[assignment]
    old_cfg = retriever._cfg

    try:
        retriever._cfg = cfg
        collections = container.domain.get_collection_names()

        # 检查 collections
        col_parent = retriever._dense.get_collection(collections.parent)
        col_child = retriever._dense.get_collection(collections.child)
        col_hyde = retriever._dense.get_collection(collections.hyde)
        v2_ready = all(c is not None for c in [col_parent, col_child, col_hyde])

        if not v2_ready:
            raise RuntimeError("v2 collections 未就绪，无法评测 rerank")

        queries = [QueryRoute(text=query)]

        # ── RRF 前 ───────────────────────────────────────────────────────────
        weighted_lists, pid_best_dense, pid_best_bm25, _branch_traces = await (
            retriever._async_multi_route_fetch(queries, collections)
        )
        rrf_results = retriever._rrf_merge(weighted_lists)
        rrf_score_map = {pid: score for pid, score in rrf_results}
        top20_ids = [pid for pid, _ in rrf_results[:20]]

        parent_texts = await asyncio.to_thread(
            retriever._dense.fetch_parents, top20_ids, collections.parent
        )

        seen: set[str] = set()
        candidates: list[RankedDoc] = []
        for pid in top20_ids:
            if pid in parent_texts and pid not in seen:
                seen.add(pid)
                candidates.append(RankedDoc(
                    id=pid,
                    text=parent_texts[pid],
                    rrf_score=rrf_score_map[pid],
                    vector_rank=pid_best_dense.get(pid, 999),
                    bm25_rank=pid_best_bm25.get(pid, 999),
                ))

        # 获取 gt_ids（用 before 的候选）
        chunks_raw = [{"id": c.id, "text": c.text} for c in candidates]
        # 构造一个临时 item 用于 hit_judge（无 expected_chunk 时使用空）
        # 实际上 caller 会传入 item，这里我们先返回 candidates，gt 在外面算

        # ── Rerank 后 ────────────────────────────────────────────────────────
        if cfg.enable_rerank:
            reranked = await asyncio.to_thread(
                container.reranker.rerank,
                query, candidates, cfg.rerank_top_k,
            )
        else:
            for c in candidates:
                c.score = c.rrf_score
            reranked = sorted(candidates, key=lambda x: x.score, reverse=True)[:cfg.rerank_top_k]

        return candidates, reranked, set()

    finally:
        retriever._cfg = old_cfg


async def evaluate_rerank_single(
    item: dict,
    container: RAGContainer,
    config: HybridConfig | None = None,
) -> RerankComparison:
    """对单条 query 评测 rerank 效果。"""
    query = item["query"]
    cfg = config or HybridConfig()

    before, after, _ = await _retrieve_rerank_stages(query, container, cfg)

    # 计算 gt
    chunks_raw = [{"id": c.id, "text": c.text} for c in before]
    gt_ids = ground_truth_ids(item, chunks_raw)

    before_ids = [c.id for c in before]
    after_ids = [c.id for c in after]

    before_rank = _find_first_gt_rank(before_ids, gt_ids)
    after_rank = _find_first_gt_rank(after_ids, gt_ids)

    top1_is_hit_before = before_ids[0] in gt_ids if before_ids else False
    top1_is_hit_after = after_ids[0] in gt_ids if after_ids else False

    rank_delta = after_rank - before_rank

    is_win = (not top1_is_hit_before) and top1_is_hit_after
    is_loss = top1_is_hit_before and (not top1_is_hit_after)
    is_hold = top1_is_hit_before and top1_is_hit_after

    ce_scores = [(c.id, c.ce_score) for c in after]

    return RerankComparison(
        query=query,
        gt_ids=gt_ids,
        before_rank=before_rank,
        after_rank=after_rank,
        before_top1_id=before_ids[0] if before_ids else "",
        after_top1_id=after_ids[0] if after_ids else "",
        top1_is_hit_before=top1_is_hit_before,
        top1_is_hit_after=top1_is_hit_after,
        rank_delta=rank_delta,
        is_win=is_win,
        is_loss=is_loss,
        is_hold=is_hold,
        ce_scores=ce_scores,
    )


async def run_rerank_eval(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    config: HybridConfig | None = None,
    verbose: bool = True,
) -> dict:
    """
    批量评测 rerank 效果。
    """
    if container is None:
        container = RAGContainer.from_settings(get_settings())

    if dataset_path is None:
        dataset = container.domain.get_eval_dataset()
        if not dataset:
            eval_logger.warning("Domain 未提供评测集")
            return {}
    else:
        import json
        with open(dataset_path, encoding="utf-8") as f:
            dataset = json.load(f)
        dataset = [item for item in dataset if "query" in item]

    total = len(dataset)
    comparisons: list[RerankComparison] = []

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Rerank Evaluation   共 {total} 条")
        print(f"{'─'*70}\n")

    for i, item in enumerate(dataset, 1):
        comp = await evaluate_rerank_single(item, container, config)
        comparisons.append(comp)

        if verbose:
            marker = "🎯" if comp.is_win else ("💥" if comp.is_loss else "➖")
            print(
                f"[{i:02d}/{total}] {marker} rank {comp.before_rank} → {comp.after_rank}  "
                f"top1_hit={comp.top1_is_hit_before}→{comp.top1_is_hit_after}"
            )
            print(f"  Query: {comp.query[:60]}")
            if comp.is_win:
                print(f"  ✅ Rerank 把正确 chunk 推到了 top1")
            elif comp.is_loss:
                print(f"  ❌ Rerank 把正确 chunk 挤出了 top1")
            print()

    wins = sum(1 for c in comparisons if c.is_win)
    losses = sum(1 for c in comparisons if c.is_loss)
    holds = sum(1 for c in comparisons if c.is_hold)
    misses = total - wins - losses - holds  # 原本就没命中，rerank 后也没命中

    avg_rank_delta = sum(c.rank_delta for c in comparisons) / total if total else 0.0

    report = {
        "total": total,
        "wins": wins,
        "losses": losses,
        "holds": holds,
        "misses": misses,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "loss_rate": round(losses / total, 4) if total else 0.0,
        "hold_rate": round(holds / total, 4) if total else 0.0,
        "avg_rank_delta": round(avg_rank_delta, 2),
        "details": [
            {
                "query": c.query,
                "before_rank": c.before_rank,
                "after_rank": c.after_rank,
                "before_top1_id": c.before_top1_id,
                "after_top1_id": c.after_top1_id,
                "top1_is_hit_before": c.top1_is_hit_before,
                "top1_is_hit_after": c.top1_is_hit_after,
                "rank_delta": c.rank_delta,
                "is_win": c.is_win,
                "is_loss": c.is_loss,
                "ce_scores": c.ce_scores,
            }
            for c in comparisons
        ],
    }

    if verbose:
        print(f"{'─'*70}")
        print(f"  Rerank 综合报告")
        print(f"    🎯 Win   = {wins} ({wins/total:.1%})  正确 chunk 被推到 top1")
        print(f"    💥 Loss  = {losses} ({losses/total:.1%})  正确 chunk 被挤出 top1")
        print(f"    ➖ Hold  = {holds} ({holds/total:.1%})  原本 top1 保持 top1")
        print(f"    ❌ Miss  = {misses} ({misses/total:.1%})  原本未命中，rerank 后也未命中")
        print(f"    平均排名变化 = {avg_rank_delta:+.1f}（负数=上升）")
        print(f"{'─'*70}\n")

    return report
