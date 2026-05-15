"""
Retrieval Trace — 检索全链路追踪

让系统从"黑盒"变成"白盒"。
每条 query 记录完整的检索 pipeline：
  query → rewrite → dense召回 → hyde召回 → bm25召回 → rrf融合 → rerank排序 → 最终chunk

输出格式：结构化 dict + 人类可读打印。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

from rag_framework.core.logger import retrieval_logger


@dataclass
class BranchTrace:
    """单路检索分支的 trace。"""
    kind: str = ""            # dense | hyde | bm25
    query_text: str = ""
    weight: float = 1.0
    status: str = "pending"   # pending | success | timeout | error
    latency_ms: float = 0.0
    result_count: int = 0
    top_ids: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class RerankTrace:
    """Rerank 阶段的 trace。"""
    status: str = "pending"
    latency_ms: float = 0.0
    input_count: int = 0
    output_count: int = 0
    top_ce_score: float = 0.0
    error: str | None = None


@dataclass
class RetrievalTrace:
    """
    单次检索的完整 trace。
    """
    query: str = ""
    original_query: str = ""
    rewritten_query: str = ""
    rewrite_type: str = "original"   # original | rule | llm
    rewrite_latency_ms: float = 0.0

    branches: list[BranchTrace] = field(default_factory=list)
    rrf_latency_ms: float = 0.0
    rrf_input_count: int = 0
    rrf_output_count: int = 0

    rerank: RerankTrace = field(default_factory=RerankTrace)
    lost_in_middle: bool = False

    final_latency_ms: float = 0.0
    final_chunk_count: int = 0
    final_top_ids: list[str] = field(default_factory=list)
    top_ce_score: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print_trace(self) -> str:
        """生成人类可读的 trace 字符串。"""
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════════════════╗",
            "║                         RETRIEVAL TRACE                              ║",
            "╠══════════════════════════════════════════════════════════════════════╣",
            f"║ 原始 Query : {self.original_query[:55]:55} ║",
        ]
        if self.rewritten_query and self.rewritten_query != self.original_query:
            lines.append(f"║ 改写 Query : {self.rewritten_query[:55]:55} ║")
            lines.append(f"║ 改写类型   : {self.rewrite_type:<10} 耗时: {self.rewrite_latency_ms:>7.0f}ms         ║")
        else:
            lines.append(f"║ 改写       : 无改写                               耗时: {self.rewrite_latency_ms:>7.0f}ms         ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════╣")
        lines.append("║ 多路召回                                                             ║")
        for b in self.branches:
            status_icon = "✅" if b.status == "success" else ("⏱" if b.status == "timeout" else "❌")
            lines.append(
                f"║   [{status_icon}] {b.kind:<7}  q={b.query_text[:30]:30}  "
                f"n={b.result_count:<3}  {b.latency_ms:>6.0f}ms          ║"
            )
            if b.error:
                lines.append(f"║       ⚠ {b.error[:50]:50}           ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════╣")
        lines.append(f"║ RRF 融合     : 输入 {self.rrf_input_count:<3} 路 → 输出 {self.rrf_output_count:<3} 条  "
                     f"{self.rrf_latency_ms:>6.0f}ms                    ║")

        r = self.rerank
        r_icon = "✅" if r.status == "success" else ("⏱" if r.status == "timeout" else "❌")
        lines.append(
            f"║ Rerank [{r_icon}]   : 输入 {r.input_count:<3} → 输出 {r.output_count:<3}  "
            f"top_ce={r.top_ce_score:.3f}  {r.latency_ms:>6.0f}ms          ║"
        )
        if r.error:
            lines.append(f"║       ⚠ {r.error[:50]:50}           ║")

        if self.lost_in_middle:
            lines.append("║ LiM 重排     : 已应用                                               ║")

        lines.append("╠══════════════════════════════════════════════════════════════════════╣")
        lines.append(
            f"║ 最终结果     : {self.final_chunk_count} 个片段  top_ce={self.top_ce_score:.3f}  "
            f"总耗时={self.final_latency_ms:>6.0f}ms                   ║"
        )
        lines.append(f"║ Top IDs      : {', '.join(self.final_top_ids[:5])[:55]:55} ║")
        lines.append("╚══════════════════════════════════════════════════════════════════════╝")
        lines.append("")
        return "\n".join(lines)


# ─── 全局 Trace 存储（用于失败分析收集）──────────────────────────────────────────

_active_traces: list[RetrievalTrace] = []
_MAX_TRACE_HISTORY = 1000


def record_trace(trace: RetrievalTrace) -> None:
    """记录 trace 到全局历史。"""
    _active_traces.append(trace)
    if len(_active_traces) > _MAX_TRACE_HISTORY:
        _active_traces.pop(0)


def get_recent_traces(n: int = 10) -> list[RetrievalTrace]:
    """获取最近 n 条 trace。"""
    return _active_traces[-n:]


def get_traces_by_query(query_substring: str) -> list[RetrievalTrace]:
    """按 query 子串搜索 trace。"""
    return [t for t in _active_traces if query_substring in t.query]


def clear_traces() -> None:
    """清空 trace 历史。"""
    _active_traces.clear()
