"""
通用排序质量评测器（Ranking Evaluator）

与领域无关，通过 DomainPlugin 获取评测集，
通过 Retriever 执行检索，输出 MRR / Hit@K / Recall@K / Latency 指标。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.domain.base import QueryRoute
from rag_framework.eval.hit_judge import ground_truth_ids
from rag_framework.eval.metrics import (
    EvalMetrics,
    aggregate_metrics,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_framework.retrieval.base import RetrievedDoc
from rag_framework.retrieval.fusion import HybridConfig


# ─── 评测集加载 ─────────────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [item for item in raw if "query" in item]


# ─── 单条 query 评测 ────────────────────────────────────────────────────────────

def evaluate_single_query(
    item: dict,
    container: RAGContainer,
    config: HybridConfig | None = None,
    enable_rewrite: bool = True,
) -> dict:
    """
    对单条 query 执行结构化检索，返回包含排序指标的详细结果。
    """
    query = item["query"]
    cfg = config or HybridConfig()

    # Query 扩写（可控开关）
    t_rewrite = 0.0
    if enable_rewrite:
        t0 = time.perf_counter()
        route = container.domain.classify_query(query, [])
        queries = [route]
        t_rewrite = (time.perf_counter() - t0) * 1000
    else:
        queries = [QueryRoute(text=query)]

    # 临时修改 retriever 配置
    old_cfg = container.retriever._cfg
    try:
        container.retriever._cfg = cfg
        result = container.retriever.retrieve(queries)
    finally:
        container.retriever._cfg = old_cfg

    rank_list = [d.id for d in result.docs]
    chunks_raw = [{"id": d.id, "text": d.text} for d in result.docs]
    gt_ids = ground_truth_ids(item, chunks_raw)

    # 计算排名指标
    first_rank = 999
    for idx, doc_id in enumerate(rank_list, start=1):
        if doc_id in gt_ids:
            first_rank = idx
            break

    rr = reciprocal_rank(rank_list, gt_ids)
    h1 = hit_at_k(rank_list, gt_ids, 1)
    h3 = hit_at_k(rank_list, gt_ids, 3)
    h5 = hit_at_k(rank_list, gt_ids, 5)
    r5 = recall_at_k(rank_list, gt_ids, 5)

    total_latency = result.latency_ms + t_rewrite

    top_chunks = [
        {
            "pos": i + 1,
            "id": d.id,
            "is_hit": d.id in gt_ids,
            "final_score": round(d.score, 4),
            "ce_score": d.metadata.get("ce_score", 0.0),
            "rrf_score": d.metadata.get("rrf_score", 0.0),
            "preview": d.text[:80].replace("\n", " "),
        }
        for i, d in enumerate(result.docs[:5])
    ]

    return {
        "query": query,
        "hit": h5 > 0,
        "rank": first_rank,
        "rr": round(rr, 4),
        "hit@1": h1,
        "hit@3": h3,
        "hit@5": h5,
        "recall@5": r5,
        "latency_ms": round(total_latency, 2),
        "latency_breakdown": {"rewrite": round(t_rewrite, 2), "total": round(total_latency, 2)},
        "matched_ids": list(gt_ids),
        "top_chunks": top_chunks,
    }


# ─── 批量评测 ───────────────────────────────────────────────────────────────────

def run_ranking_eval(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    config: HybridConfig | None = None,
    enable_rewrite: bool = True,
    verbose: bool = True,
) -> EvalMetrics:
    """
    对评测集批量执行排序质量评测。

    Args:
        dataset_path: 评测集路径，默认使用 domain 的 get_eval_dataset()
        container: RAG 容器，默认自动构建
        config: 检索配置（用于消融实验）
        enable_rewrite: 是否启用 query 扩写
        verbose: 是否打印逐条结果

    Returns:
        EvalMetrics 综合指标
    """
    if container is None:
        container = RAGContainer.from_settings(get_settings())

    # 获取评测集
    if dataset_path is None:
        dataset = container.domain.get_eval_dataset()
        if not dataset:
            eval_logger.warning("Domain 未提供评测集，返回空指标")
            return EvalMetrics()
    else:
        dataset = load_dataset(dataset_path)

    total = len(dataset)
    rank_lists: list[list[str]] = []
    ground_truths: list[set[str]] = []
    latencies: list[float] = []
    details: list[dict] = []

    if verbose:
        name = dataset_path.name if dataset_path else "domain_eval"
        print(f"\n{'─'*70}")
        print(f"  Ranking Evaluation   评测集: {name}  共 {total} 条")
        print(f"{'─'*70}\n")

    for i, item in enumerate(dataset, 1):
        res = evaluate_single_query(item, container, config=config, enable_rewrite=enable_rewrite)

        rank_lists.append([c["id"] for c in res["top_chunks"]])
        ground_truths.append(set(res["matched_ids"]))
        latencies.append(res["latency_ms"])
        details.append(res)

        if verbose:
            rank_str = f"rank={res['rank']}" if res["hit"] else "rank=MISS"
            print(f"[{i:02d}/{total}] {'✅' if res['hit'] else '❌'} {rank_str:12s}  ({res['latency_ms']:.0f}ms)")
            print(f"  Query : {res['query'][:65]}")
            for tc in res["top_chunks"][:3]:
                marker = "  **" if tc["is_hit"] else "    "
                print(f"{marker}[{tc['pos']}] {tc['id'][:30]:30s}  final={tc['final_score']:.3f}  {tc['preview'][:50]!r}")
            print()

    cfg = config or HybridConfig()
    flags = []
    if not cfg.enable_hyde:
        flags.append("-hyde")
    if not cfg.enable_bm25:
        flags.append("-bm25")
    if not cfg.enable_rerank:
        flags.append("-rerank")
    if not cfg.enable_lost_in_middle:
        flags.append("-LiM")
    config_summary = " | ".join(flags) if flags else "baseline"

    metrics = aggregate_metrics(
        rank_lists=rank_lists,
        ground_truths=ground_truths,
        latencies_ms=latencies,
        config_summary=config_summary,
    )

    if verbose:
        print(f"{'─'*70}")
        print(f"  综合指标")
        print(f"    Recall@5  = {metrics.recall_at_5:.0%}")
        print(f"    MRR       = {metrics.mrr:.3f}")
        print(f"    Hit@1     = {metrics.hit_at_1:.0%}")
        print(f"    Hit@3     = {metrics.hit_at_3:.0%}")
        print(f"    Hit@5     = {metrics.hit_at_5:.0%}")
        print(f"    平均延迟   = {metrics.latency.mean:.0f}ms  (P95={metrics.latency.p95:.0f}ms)")
        print(f"{'─'*70}\n")

    return metrics
