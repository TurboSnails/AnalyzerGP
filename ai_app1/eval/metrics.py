"""
RAG 评测指标库（Retrieval & Ranking Metrics）
==============================================

提供标准化的检索与排序指标计算，支持：
  - Recall@K          : 正确 chunk 是否被召回
  - Hit@1/3/5         : 正确 chunk 是否出现在 Top-K
  - MRR               : Mean Reciprocal Rank（排序质量核心指标）
  - NDCG@K            : 归一化折损累积增益（考虑分级相关性）
  - Latency 统计      : P50 / P95 / P99 / Mean / Std

设计原则：
  1. 纯函数，无外部依赖，可单元测试
  2. 输入为 rank_list（候选 chunk id 的有序列表）与 ground_truth（正确 id 集合）
  3. rank 为 1-based（符合信息检索领域惯例）
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


# ───────────────────────────────────────────────────────────────────────────────
# 1. 召回指标
# ───────────────────────────────────────────────────────────────────────────────

def recall_at_k(rank_list: list[str], ground_truth: set[str], k: int = 5) -> float:
    """
    Recall@K：在 Top-K 召回列表中是否包含任意一个 ground_truth。

    Args:
        rank_list    : 按排序后的 chunk id 列表（越靠前越相关）
        ground_truth : 期望命中的 chunk id 集合（支持多跳题的多个分支）
        k            : 截断位置

    Returns:
        1.0 若命中任一 ground_truth，否则 0.0
    """
    if not rank_list or not ground_truth:
        return 0.0
    return 1.0 if set(rank_list[:k]) & ground_truth else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# 2. 排序指标（核心）
# ───────────────────────────────────────────────────────────────────────────────

def reciprocal_rank(rank_list: list[str], ground_truth: set[str]) -> float:
    """
    Reciprocal Rank（RR）：第一个命中的 ground_truth 的排名倒数。

    例如：
        rank_list = ["A", "B", "C"], ground_truth = {"B"}
        → 第一个命中在 rank=2 → RR = 1/2 = 0.5

    若未命中，返回 0.0。
    """
    for idx, doc_id in enumerate(rank_list, start=1):
        if doc_id in ground_truth:
            return 1.0 / idx
    return 0.0


def mean_reciprocal_rank(rank_lists: list[list[str]], ground_truths: list[set[str]]) -> float:
    """
    MRR（Mean Reciprocal Rank）：多个查询的 RR 均值。

    MRR 是 RAG 排序质量的核心指标，因为它直接反映：
      "正确 chunk 是否排在前面"

    与 Recall@5=100% 不同，MRR 能区分：
      - rank=1（MRR=1.0） → LLM 第一眼就看到正确答案
      - rank=5（MRR=0.2） → LLM 可能忽略被噪声压到后面的正确内容
    """
    if not rank_lists:
        return 0.0
    rr_sum = sum(
        reciprocal_rank(rl, gt)
        for rl, gt in zip(rank_lists, ground_truths)
    )
    return rr_sum / len(rank_lists)


def hit_at_k(rank_list: list[str], ground_truth: set[str], k: int) -> float:
    """
    Hit@K：正确 chunk 是否出现在 Top-K（二元指标，0 或 1）。

    Hit@1 尤其重要，因为 LLM 的注意力集中在上下文最前面。
    """
    return recall_at_k(rank_list, ground_truth, k)


# ───────────────────────────────────────────────────────────────────────────────
# 3. NDCG（Normalized Discounted Cumulative Gain）
# ───────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: list[float], k: int) -> float:
    """
    DCG@K：折损累积增益。

    位置越靠前，相关性贡献越大（log2 折损）。
    """
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(rank_list: list[str], relevance_map: dict[str, float], k: int = 5) -> float:
    """
    NDCG@K：归一化 DCG。

    Args:
        rank_list    : 排序后的 chunk id 列表
        relevance_map: {chunk_id: relevance_score}，score 越高表示越相关
        k            : 截断位置

    例如：
        若正确 chunk 的 relevance=3，排在第 5 位，则 NDCG 会惩罚其排名靠后。
    """
    if not rank_list or not relevance_map:
        return 0.0

    relevances = [relevance_map.get(doc_id, 0.0) for doc_id in rank_list]
    ideal = sorted(relevance_map.values(), reverse=True)
    ideal_relevances = ideal + [0.0] * max(0, len(rank_list) - len(ideal))

    dcg = dcg_at_k(relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


# ───────────────────────────────────────────────────────────────────────────────
# 4. 延迟统计
# ───────────────────────────────────────────────────────────────────────────────

def percentile(values: list[float], p: float) -> float:
    """计算百分位数（线性插值）。"""
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
    """延迟统计摘要。"""
    mean: float = 0.0
    std: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """
    计算一组延迟数据的统计摘要（毫秒）。

    TTFT（Time To First Token）工程中关注 P95 / P99，
    因为尾部延迟决定了用户体验的最差情况。
    """
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


# ───────────────────────────────────────────────────────────────────────────────
# 5. 综合指标聚合器
# ───────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """
    单次实验（一个配置）的综合评测结果。

    输出格式与 ablation study 表格对齐，可直接 Markdown 化。
    """
    # 召回
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0

    # 排序（核心）
    mrr: float = 0.0
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0

    # NDCG
    ndcg_at_5: float = 0.0

    # 延迟
    latency: LatencyStats = field(default_factory=LatencyStats)

    # 消融配置描述
    config_summary: str = ""

    def to_dict(self) -> dict:
        """序列化为字典（便于 JSON 持久化）。"""
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
        """
        从 to_dict() 的输出反序列化为 EvalMetrics。

        负责处理 `recall@K` ↔ `recall_at_K` 的 key 映射，
        消费者不必再自己维护一份 key map。
        """
        key_map = {
            "recall@1": "recall_at_1", "recall@3": "recall_at_3", "recall@5": "recall_at_5",
            "hit@1": "hit_at_1", "hit@3": "hit_at_3", "hit@5": "hit_at_5",
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
        """生成 Markdown 表格的一行。"""
        return (
            f"| {self.config_summary} | "
            f"{self.recall_at_5:.0%} | {self.mrr:.3f} | "
            f"{self.hit_at_1:.0%} | {self.hit_at_3:.0%} | {self.hit_at_5:.0%} | "
            f"{self.latency.mean:.0f}ms |"
        )

    @staticmethod
    def markdown_header() -> str:
        """Markdown 表头。"""
        return (
            "| 配置 | Recall@5 | MRR | Hit@1 | Hit@3 | Hit@5 | 平均延迟 |\n"
            "|------|----------|-----|-------|-------|-------|----------|"
        )


def aggregate_metrics(
    rank_lists: list[list[str]],
    ground_truths: list[set[str]],
    latencies_ms: list[float],
    config_summary: str = "",
) -> EvalMetrics:
    """
    聚合多组查询的评测指标。

    Args:
        rank_lists    : 每个查询的 chunk id 排序列表
        ground_truths : 每个查询的正确 chunk id 集合
        latencies_ms  : 每个查询的耗时（毫秒）
        config_summary: 本次实验的配置描述

    Returns:
        EvalMetrics 综合指标对象
    """
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
