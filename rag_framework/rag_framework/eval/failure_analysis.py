"""
tonFailure Analysis System — 失败分析与数据闭环

自动收集并分类"问题 query"，为迭代优化提供数据基础。

收集维度：
  1. miss_query:        检索未命中（recall@5 = 0）
  2. low_ce_score:      top_ce < threshold（检索到但置信度低）
  3. low_relevance:     用户追问（暗示上一轮回答不够好）
  4. explicit_bad:      用户明确表达不满意（"不对"、"不是这个"等）
  5. rerank_loss:       rerank 把正确 chunk 挤出 top1
  6. rewrite_degrade:   rewrite 后指标下降

存储：JSON Lines 文件，支持增量追加。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from rag_framework.core.logger import eval_logger


# ─── 失败样本定义 ───────────────────────────────────────────────────────────────

@dataclass
class FailureCase:
    """单条失败样本。"""
    query: str = ""
    category: str = ""           # miss | low_ce | low_relevance | explicit_bad | rerank_loss | rewrite_degrade
    reason: str = ""             # 具体原因描述
    timestamp: str = ""
    session_id: str = ""
    trace: dict[str, Any] = field(default_factory=dict)   # 关联的 RetrievalTrace dict
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FailureCase":
        return cls(
            query=d.get("query", ""),
            category=d.get("category", ""),
            reason=d.get("reason", ""),
            timestamp=d.get("timestamp", ""),
            session_id=d.get("session_id", ""),
            trace=d.get("trace", {}),
            metadata=d.get("metadata", {}),
        )


# ─── 存储后端 ───────────────────────────────────────────────────────────────────

class FailureStore:
    """失败样本持久化存储（JSON Lines）。"""

    def __init__(self, path: Path | str = "reports/failure_cases.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[FailureCase] = []
        self._buffer_size = 10

    def append(self, case: FailureCase) -> None:
        """追加一条失败样本（先写 buffer，满则刷盘）。"""
        self._buffer.append(case)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    def flush(self) -> None:
        """将 buffer 刷盘。"""
        if not self._buffer:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for case in self._buffer:
                f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
        eval_logger.info(f"FailureStore: 已刷盘 {len(self._buffer)} 条")
        self._buffer.clear()

    def load_all(self) -> list[FailureCase]:
        """加载所有历史失败样本。"""
        cases: list[FailureCase] = []
        if not self.path.exists():
            return cases
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(FailureCase.from_dict(json.loads(line)))
                except Exception:
                    continue
        return cases

    def get_by_category(self, category: str) -> list[FailureCase]:
        """按分类过滤。"""
        return [c for c in self.load_all() if c.category == category]

    def summary(self) -> dict[str, int]:
        """按分类统计数量。"""
        counts: dict[str, int] = {}
        for c in self.load_all():
            counts[c.category] = counts.get(c.category, 0) + 1
        return counts

    def print_summary(self) -> str:
        """生成人类可读的汇总。"""
        counts = self.summary()
        total = sum(counts.values())
        lines = [
            "",
            "🔴 Failure Analysis Summary",
            "─" * 50,
            f"{'分类':<20} {'数量':>8} {'占比':>8}",
            "─" * 50,
        ]
        for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"{cat:<20} {cnt:>8} {cnt/total:>7.1%}")
        lines.append("─" * 50)
        lines.append(f"{'总计':<20} {total:>8}")
        return "\n".join(lines)


# ─── 收集器（供 pipeline 调用）───────────────────────────────────────────────────

class FailureCollector:
    """
    失败样本收集器。

    在检索 pipeline 和对话 pipeline 的关键节点调用，
    自动判断是否属于失败样本并记录。
    """

    def __init__(self, store: FailureStore | None = None) -> None:
        self.store = store or FailureStore()
        self.low_ce_threshold: float = 0.30
        self.explicit_bad_keywords = {
            "不对", "不是这个", "答非所问", "没回答", " unrelated",
            "跑题了", "不相关", "没用", "没解决",
        }
        self.followup_keywords = {
            "那", "还有", "另外", "补充", "追问", "进一步", "详细",
        }

    # ── 检索阶段收集 ───────────────────────────────────────────────────────────

    def collect_miss(
        self,
        query: str,
        trace: dict[str, Any],
        session_id: str = "",
    ) -> None:
        """检索未命中 ground truth 时调用。"""
        self.store.append(FailureCase(
            query=query,
            category="miss",
            reason="检索未命中任何 ground truth chunk",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            trace=trace,
        ))

    def collect_low_ce(
        self,
        query: str,
        top_ce: float,
        trace: dict[str, Any],
        session_id: str = "",
    ) -> None:
        """top_ce 低于阈值时调用。"""
        if top_ce >= self.low_ce_threshold:
            return
        self.store.append(FailureCase(
            query=query,
            category="low_ce",
            reason=f"top_ce={top_ce:.3f} < threshold={self.low_ce_threshold}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            trace=trace,
            metadata={"top_ce": top_ce, "threshold": self.low_ce_threshold},
        ))

    def collect_rerank_loss(
        self,
        query: str,
        before_rank: int,
        after_rank: int,
        trace: dict[str, Any],
        session_id: str = "",
    ) -> None:
        """rerank 把正确 chunk 挤出 top1 时调用。"""
        self.store.append(FailureCase(
            query=query,
            category="rerank_loss",
            reason=f"rerank 前 rank={before_rank}，rerank 后 rank={after_rank}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            trace=trace,
            metadata={"before_rank": before_rank, "after_rank": after_rank},
        ))

    def collect_rewrite_degrade(
        self,
        query: str,
        original: str,
        rewritten: str,
        delta_recall: float,
        session_id: str = "",
    ) -> None:
        """rewrite 后指标下降时调用。"""
        self.store.append(FailureCase(
            query=query,
            category="rewrite_degrade",
            reason=f"rewrite 后 recall 下降 {delta_recall:.4f}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            metadata={
                "original": original,
                "rewritten": rewritten,
                "delta_recall": delta_recall,
            },
        ))

    # ── 对话阶段收集 ───────────────────────────────────────────────────────────

    def collect_explicit_bad(
        self,
        query: str,
        user_message: str,
        session_id: str = "",
    ) -> None:
        """用户明确表达不满意时调用。"""
        matched = [k for k in self.explicit_bad_keywords if k in user_message]
        if not matched:
            return
        self.store.append(FailureCase(
            query=query,
            category="explicit_bad",
            reason=f"用户表达不满意，命中关键词: {matched}",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            metadata={"user_message": user_message, "matched_keywords": matched},
        ))

    def collect_followup(
        self,
        prev_query: str,
        followup_query: str,
        session_id: str = "",
    ) -> None:
        """
        检测到用户追问时调用。
        追问信号：短 query 含 followup 关键词，或连续多轮问相似问题。
        """
        is_followup = any(k in followup_query for k in self.followup_keywords)
        is_short = len(followup_query) < 20
        if not (is_followup or is_short):
            return
        self.store.append(FailureCase(
            query=followup_query,
            category="followup",
            reason=f"检测到追问（前序 query: {prev_query[:40]}...）",
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            session_id=session_id,
            metadata={"prev_query": prev_query},
        ))

    def flush(self) -> None:
        self.store.flush()


# ─── 全局默认实例 ───────────────────────────────────────────────────────────────

_default_collector: FailureCollector | None = None


def get_failure_collector() -> FailureCollector:
    global _default_collector
    if _default_collector is None:
        _default_collector = FailureCollector()
    return _default_collector


def set_failure_collector(collector: FailureCollector) -> None:
    global _default_collector
    _default_collector = collector
