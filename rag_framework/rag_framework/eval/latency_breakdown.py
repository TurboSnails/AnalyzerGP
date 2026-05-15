"""
Latency Breakdown — 检索延迟精细化拆解

统计每个阶段的耗时：
  rewrite → classify → dense → hyde → bm25 → fetch_parents → rerank → lost_in_middle

与 RetrievalTrace 配合使用，但更专注于数值聚合和瓶颈发现。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag_framework.eval.metrics import compute_latency_stats


@dataclass
class PhaseLatency:
    """单次检索的各阶段延迟。"""
    rewrite_ms: float = 0.0
    classify_ms: float = 0.0
    dense_ms: float = 0.0
    hyde_ms: float = 0.0
    bm25_ms: float = 0.0
    fetch_parents_ms: float = 0.0
    rrf_ms: float = 0.0
    rerank_ms: float = 0.0
    lim_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "rewrite": self.rewrite_ms,
            "classify": self.classify_ms,
            "dense": self.dense_ms,
            "hyde": self.hyde_ms,
            "bm25": self.bm25_ms,
            "fetch_parents": self.fetch_parents_ms,
            "rrf": self.rrf_ms,
            "rerank": self.rerank_ms,
            "lost_in_middle": self.lim_ms,
            "total": self.total_ms,
        }


@dataclass
class LatencyBreakdownReport:
    """批量检索的延迟拆解报告。"""
    phase_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    bottleneck: str = ""
    total_queries: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "bottleneck": self.bottleneck,
            "phase_stats": self.phase_stats,
        }

    def print_report(self) -> str:
        lines = [
            "",
            "📊 Latency Breakdown",
            "─" * 65,
            f"{'阶段':<18} {'平均(ms)':>10} {'P50':>8} {'P95':>8} {'P99':>8} {'占比':>8}",
            "─" * 65,
        ]
        total_mean = self.phase_stats.get("total", {}).get("mean", 1.0)
        for phase, stats in self.phase_stats.items():
            if phase == "total":
                continue
            ratio = stats["mean"] / total_mean if total_mean > 0 else 0.0
            lines.append(
                f"{phase:<18} {stats['mean']:>10.1f} {stats['p50']:>8.1f} "
                f"{stats['p95']:>8.1f} {stats['p99']:>8.1f} {ratio:>7.1%}"
            )
        lines.append("─" * 65)
        lines.append(
            f"{'total':<18} {self.phase_stats.get('total', {}).get('mean', 0):>10.1f} "
            f"{self.phase_stats.get('total', {}).get('p50', 0):>8.1f} "
            f"{self.phase_stats.get('total', {}).get('p95', 0):>8.1f} "
            f"{self.phase_stats.get('total', {}).get('p99', 0):>8.1f}"
        )
        lines.append(f"\n🔴 瓶颈阶段: {self.bottleneck}")
        return "\n".join(lines)


def aggregate_phase_latencies(phase_list: list[PhaseLatency]) -> LatencyBreakdownReport:
    """
    从多次检索的 PhaseLatency 列表中聚合出报告。
    """
    if not phase_list:
        return LatencyBreakdownReport()

    # 收集各阶段的原始数据
    raw: dict[str, list[float]] = {k: [] for k in PhaseLatency().to_dict().keys()}
    for pl in phase_list:
        d = pl.to_dict()
        for k, v in d.items():
            raw[k].append(v)

    phase_stats = {}
    max_mean = 0.0
    bottleneck = ""
    for phase, values in raw.items():
        stats = compute_latency_stats(values)
        phase_stats[phase] = {
            "mean": stats.mean,
            "std": stats.std,
            "p50": stats.p50,
            "p95": stats.p95,
            "p99": stats.p99,
            "min": stats.min,
            "max": stats.max,
            "count": stats.count,
        }
        if phase != "total" and stats.mean > max_mean:
            max_mean = stats.mean
            bottleneck = phase

    return LatencyBreakdownReport(
        phase_stats=phase_stats,
        bottleneck=bottleneck,
        total_queries=len(phase_list),
    )


# ─── 上下文管理器：方便在 retrieve() 中计时 ─────────────────────────────────────

class PhaseTimer:
    """
    简化版阶段计时器。

    Usage:
        timer = PhaseTimer()
        with timer.phase("rewrite"):
            do_rewrite()
        with timer.phase("dense"):
            do_dense()
        latency = timer.finish()
    """

    def __init__(self) -> None:
        self._phases: dict[str, float] = {}
        self._start: float | None = None
        self._current_phase: str | None = None

    def phase(self, name: str):
        class _Ctx:
            def __enter__(inner_self):
                self._current_phase = name
                self._start = time.perf_counter()
                return inner_self

            def __exit__(inner_self, *args):
                if self._start is not None:
                    self._phases[name] = (time.perf_counter() - self._start) * 1000
                self._start = None
                self._current_phase = None
        return _Ctx()

    def record(self, name: str, ms: float) -> None:
        """手动记录一个阶段的耗时。"""
        self._phases[name] = ms

    def finish(self) -> PhaseLatency:
        """返回 PhaseLatency，自动计算 total。"""
        total = sum(self._phases.values())
        kw = {k.replace("_ms", ""): v for k, v in self._phases.items()}
        kw["total_ms"] = total
        # 映射回 PhaseLatency 字段名
        return PhaseLatency(
            rewrite_ms=kw.get("rewrite", 0.0),
            classify_ms=kw.get("classify", 0.0),
            dense_ms=kw.get("dense", 0.0),
            hyde_ms=kw.get("hyde", 0.0),
            bm25_ms=kw.get("bm25", 0.0),
            fetch_parents_ms=kw.get("fetch_parents", 0.0),
            rrf_ms=kw.get("rrf", 0.0),
            rerank_ms=kw.get("rerank", 0.0),
            lim_ms=kw.get("lost_in_middle", 0.0),
            total_ms=total,
        )
