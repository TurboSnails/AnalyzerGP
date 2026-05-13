"""
RAG 评测指标库（Retrieval & Ranking Metrics）

纯函数，无外部依赖。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ─── 召回指标 ───────────────────────────────────────────────────────────────────

def recall_at_k(rank_list: list[str], ground_truth: set[str], k: int = 5) -> float:
    if not rank_list or not ground_truth:
        return 0.0
    return 1.0 if set(rank_list[:k]) & ground_truth else 0.0


# ─── 排序指标 ───────────────────────────────────────────────────────────────────

def reciprocal_rank(rank_list: list[str], ground_truth: set[str]) -> float:
    for idx, doc_id in enumerate(rank_list, start=1):
        if doc_id in ground_truth:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(rank_lists: list[list[str]], ground_truths: list[set[str]]) -> float:
    if not rank_lists:
        return 0.0
    rr_sum = sum(reciprocal_rank(rl, gt) for rl, gt in zip(rank_lists, ground_truths))
    return rr_sum / len(rank_lists)


def hit_at_k(rank_list: list[str], ground_truth: set[str], k: int) -> float:
    return recall_at_k(rank_list, ground_truth, k)


# ─── DCG / NDCG ─────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(rank_list: list[str], relevance_map: dict[str, float], k: int = 5) -> float:
    if not rank_list or not relevance_map:
        return 0.0
    relevances = [relevance_map.get(doc_id, 0.0) for doc_id in rank_list]
    ideal = sorted(relevance_map.values(), reverse=True)
    ideal_relevances = ideal + [0.0] * max(0, len(rank_list) - len(ideal))
    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


# ─── 延迟统计 ───────────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    idx = (n - 1) * p / 100.0
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - idx) + s[hi] * (idx - lo)


@dataclass
class LatencyStats:
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats()
    s = sorted(latencies_ms)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    std = math.sqrt(variance)
    return LatencyStats(
        mean=round(mean, 2),
        std=round(std, 2),
        p50=round(percentile(s, 50.0), 2),
        p95=round(percentile(s, 95.0), 2),
        p99=round(percentile(s, 99.0), 2),
        min=round(s[0], 2),
        max=round(s[-1], 2),
        count=n,
    )


# ─── 综合指标 ───────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """单次实验的综合评测结果。"""
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    mrr: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    ndcg_at_5: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)
    config_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "recall@1": round(self.recall_at_1, 3),
            "recall@3": round(self.recall_at_3, 3),
            "recall@5": round(self.recall_at_5, 3),
            "mrr": round(self.mrr, 3),
            "hit@1": round(self.hit_at_1, 3),
            "hit@3": round(self.hit_at_3, 3),
            "hit@5": round(self.hit_at_5, 3),
            "ndcg@5": round(self.ndcg_at_5, 3),
            "latency_ms": {
                "mean": self.latency.mean,
                "std": self.latency.std,
                "p50": self.latency.p50,
                "p95": self.latency.p95,
                "p99": self.latency.p99,
                "min": self.latency.min,
                "max": self.latency.max,
                "count": self.latency.count,
            },
            "config": self.config_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalMetrics":
        key_map = {
            "recall@1": "recall_at_1",
            "recall@3": "recall_at_3",
            "recall@5": "recall_at_5",
            "hit@1": "hit_at_1",
            "hit@3": "hit_at_3",
            "hit@5": "hit_at_5",
            "ndcg@5": "ndcg_at_5",
        }
        scalar_fields = {
            "recall_at_1", "recall_at_3", "recall_at_5",
            "mrr", "hit_at_1", "hit_at_3", "hit_at_5", "ndcg_at_5",
        }
        kwargs = {}
        for k, v in d.items():
            if k in ("latency_ms", "config"):
                continue
            mapped = key_map.get(k, k)
            if mapped in scalar_fields:
                kwargs[mapped] = v
        obj = cls(**kwargs)
        lat = d.get("latency_ms") or {}
        if lat:
            obj.latency = LatencyStats(
                mean=lat.get("mean", 0.0),
                std=lat.get("std", 0.0),
                p50=lat.get("p50", 0.0),
                p95=lat.get("p95", 0.0),
                p99=lat.get("p99", 0.0),
                min=lat.get("min", 0.0),
                max=lat.get("max", 0.0),
                count=lat.get("count", 0),
            )
        obj.config_summary = d.get("config", "")
        return obj

    def to_markdown_row(self) -> str:
        return (
            f"| {self.config_summary} | "
            f"{self.recall_at_5:.0%} | {self.mrr:.3f} | "
            f"{self.hit_at_1:.0%} | {self.hit_at_3:.0%} | {self.hit_at_5:.0%} | "
            f"{self.latency.mean:.0f}ms |"
        )

    @staticmethod
    def markdown_header() -> str:
        return (
            "| 配置 | Recall@5 | MRR | Hit@1 | Hit@3 | Hit@5 | 平均延迟 |\n"
            "|------|----------|-----|-------|-------|-------|----------|"
        )

    @classmethod
    def from_results(cls, results: list[dict]) -> "EvalMetrics":
        """从评测结果列表聚合指标。"""
        total = len(results)
        if not total:
            return cls()
        return cls(
            recall_at_5=sum(r.get("recall@5", 0.0) for r in results) / total,
            mrr=sum(r.get("rr", 0.0) for r in results) / total,
            hit_at_1=sum(r.get("hit@1", 0.0) for r in results) / total,
            hit_at_3=sum(r.get("hit@3", 0.0) for r in results) / total,
            hit_at_5=sum(r.get("hit@5", 0.0) for r in results) / total,
            latency=compute_latency_stats([r.get("latency_ms", 0.0) for r in results]),
        )


def aggregate_metrics(
    rank_lists: list[list[str]],
    ground_truths: list[set[str]],
    latencies_ms: list[float],
    config_summary: str = "",
) -> EvalMetrics:
    """聚合多组查询的评测指标。"""
    n = len(rank_lists)
    if n == 0:
        return EvalMetrics(config_summary=config_summary)

    recalls = {"recall_at_1": [], "recall_at_3": [], "recall_at_5": []}
    hits = {"hit_at_1": [], "hit_at_3": [], "hit_at_5": []}
    rr_list = []

    for rl, gt in zip(rank_lists, ground_truths):
        recalls["recall_at_1"].append(recall_at_k(rl, gt, 1))
        recalls["recall_at_3"].append(recall_at_k(rl, gt, 3))
        recalls["recall_at_5"].append(recall_at_k(rl, gt, 5))
        hits["hit_at_1"].append(hit_at_k(rl, gt, 1))
        hits["hit_at_3"].append(hit_at_k(rl, gt, 3))
        hits["hit_at_5"].append(hit_at_k(rl, gt, 5))
        rr_list.append(reciprocal_rank(rl, gt))

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    return EvalMetrics(
        recall_at_1=_mean(recalls["recall_at_1"]),
        recall_at_3=_mean(recalls["recall_at_3"]),
        recall_at_5=_mean(recalls["recall_at_5"]),
        mrr=_mean(rr_list),
        hit_at_1=_mean(hits["hit_at_1"]),
        hit_at_3=_mean(hits["hit_at_3"]),
        hit_at_5=_mean(hits["hit_at_5"]),
        latency=compute_latency_stats(latencies_ms),
        config_summary=config_summary,
    )
