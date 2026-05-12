"""
Ranking Evaluation — 排序质量评测
==================================

评测维度：
  - MRR（Mean Reciprocal Rank）：排序质量核心指标
  - Hit@1/3/5：正确 chunk 是否出现在 Top-K
  - Recall@K：传统召回指标（兼容旧版）
  - 逐 query 排名可视化：输出正确 chunk 在每个 query 中的具体排名

与 evaluate.py 的区别：
  - evaluate.py 只关心"有没有召回"（Recall）
  - ranking_eval.py 关心"召回了排第几"（MRR / Hit）

运行方式：
    uv run python -m ai_app1.eval.ranking_eval
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Iterable

# 兼容直接运行与模块运行
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_app1.eval.metrics import (
    aggregate_metrics,
    EvalMetrics,
    recall_at_k,
    reciprocal_rank,
    hit_at_k,
)
from ai_app1.retrieval.vector_store import query_db_structured, RetrievalConfig
from ai_app1.retrieval.query_rewriter import rewrite_queries


# ─── 评测集加载 ───────────────────────────────────────────────────────────────

_EVAL_FILE = Path(__file__).parent / "评测集"
_HARD_CASES_FILE = Path(__file__).parent / "data" / "hard_cases.json"


def _load_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return [item for item in raw if "query" in item]


# ─── 命中判断（复用 evaluate.py 逻辑，但返回布尔值）──────────────────────────

def _key_phrases(text: str, min_len: int = 4) -> list[str]:
    import re
    segments = re.split(r'[，。？！、：“”‘’（）【】\s]', text)
    return [s.strip() for s in segments if len(s.strip()) >= min_len]


def _ngrams(text: str, n: int = 4) -> list[str]:
    t = text.replace(" ", "")
    return [t[i:i+n] for i in range(len(t) - n + 1)]


def _is_hit(result_text: str, expected_chunk: str, evidence: str) -> bool:
    """简化版命中判断，返回 bool（用于 ranking 阶段的 label 匹配）。"""
    if not result_text:
        return False
    if evidence and evidence in result_text:
        return True
    if evidence:
        for phrase in _key_phrases(evidence):
            if phrase in result_text:
                return True
        for clause in _key_phrases(evidence, min_len=6):
            for gram in _ngrams(clause, n=4):
                if gram in result_text:
                    return True
    labels = [lbl.strip() for lbl in expected_chunk.split("/")]
    for lbl in labels:
        if lbl and lbl in result_text:
            return True
    return False


def _ground_truth_ids(item: dict, chunks: list) -> set[str]:
    """
    根据 label / evidence 匹配，返回正确 chunk 的 id 集合。

    多跳题（expected_chunk 含 "/"）允许命中任意一个分支。
    """
    expected = item.get("expected_chunk", "")
    raw_ev = item.get("evidence", "")
    evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

    matched = set()
    for c in chunks:
        if _is_hit(c["text"], expected, evidence):
            matched.add(c["id"])
    return matched


# ─── 单次 query 评测 ──────────────────────────────────────────────────────────

def evaluate_single_query(
    item: dict,
    config: RetrievalConfig | None = None,
    enable_rewrite: bool = True,
) -> dict:
    """
    对单条 query 执行结构化检索，返回包含排序指标的详细结果。

    Returns:
        {
            "query": str,
            "hit": bool,               # 是否在 Top-5 召回中命中
            "rank": int,               # 第一个命中 chunk 的排名（1-based，999=未命中）
            "rr": float,               # Reciprocal Rank
            "hit@1": float,
            "hit@3": float,
            "hit@5": float,
            "recall@5": float,
            "latency_ms": float,
            "latency_breakdown": dict,
            "matched_ids": list[str],  # 命中的 chunk id 列表
            "top_chunks": list[dict],  # Top-5 chunk 的摘要（用于可视化）
        }
    """
    query = item["query"]
    cfg = config or RetrievalConfig()

    # Query 扩写（可控开关）
    t_rewrite = 0.0
    if enable_rewrite and cfg.enable_rewrite:
        t0 = time.perf_counter()
        queries = rewrite_queries(query, history=[])
        t_rewrite = (time.perf_counter() - t0) * 1000
    else:
        from ai_app1.retrieval.query_rewriter import RewriteQuery
        queries = [RewriteQuery(text=query, type="original", weight=1.0,
                                routes=["dense", "hyde", "bm25"])]

    # 结构化检索
    result = query_db_structured(queries, config=cfg)
    rank_list = [c.id for c in result.chunks]

    # 计算 ground truth（基于 evidence/label 匹配 chunk 文本）
    chunks_raw = [{"id": c.id, "text": c.text} for c in result.chunks]
    gt_ids = _ground_truth_ids(item, chunks_raw)

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

    # Top-5 可视化摘要
    top_chunks = [
        {
            "pos": c.final_position,
            "id": c.id,
            "is_hit": c.id in gt_ids,
            "final_score": round(c.final_score, 4),
            "ce_score": round(c.ce_score, 4),
            "rrf_score": round(c.rrf_score, 4),
            "preview": c.text[:80].replace("\n", " "),
        }
        for c in result.chunks[:5]
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
        "latency_ms": round(result.latency_ms + t_rewrite, 2),
        "latency_breakdown": {**result.latency_breakdown, "rewrite": round(t_rewrite, 2)},
        "matched_ids": list(gt_ids),
        "top_chunks": top_chunks,
    }


# ─── 批量评测 ─────────────────────────────────────────────────────────────────

def run_ranking_eval(
    dataset_path: Path | None = None,
    config: RetrievalConfig | None = None,
    enable_rewrite: bool = True,
    verbose: bool = True,
) -> EvalMetrics:
    """
    对评测集批量执行排序质量评测。

    Args:
        dataset_path : 评测集路径，默认 evaluate.py 的 评测集
        config       : 检索配置（用于消融实验）
        enable_rewrite: 是否启用 query 扩写
        verbose      : 是否打印逐条结果

    Returns:
        EvalMetrics 综合指标
    """
    dataset = _load_dataset(dataset_path or _EVAL_FILE)
    total = len(dataset)

    rank_lists: list[list[str]] = []
    ground_truths: list[set[str]] = []
    latencies: list[float] = []
    details: list[dict] = []

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  Ranking Evaluation   评测集: {dataset_path.name if dataset_path else '评测集'}  共 {total} 条")
        print(f"  Config: {(config or RetrievalConfig()).summary()}")
        print(f"{'─'*70}\n")

    for i, item in enumerate(dataset, 1):
        res = evaluate_single_query(item, config=config, enable_rewrite=enable_rewrite)

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

    metrics = aggregate_metrics(
        rank_lists=rank_lists,
        ground_truths=ground_truths,
        latencies_ms=latencies,
        config_summary=(config or RetrievalConfig()).summary(),
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


# ─── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ranking_eval()
