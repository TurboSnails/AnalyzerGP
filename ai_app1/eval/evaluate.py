"""
Retrieval Evaluation — Recall@K
================================
对评测集里每条 query 执行完整的 query_db() 检索流水线，判断是否命中
expected_chunk 所对应的证据片段。

命中规则（HIT）：
  1. evidence 子串出现在任意召回 chunk 中（精确句级匹配）
  2. expected_chunk 标签出现在任意召回 chunk 中（回落标签匹配）
  多跳题（expected_chunk 含 " / "）只需命中其中任意一个分支即可。

兼容模式：
  - run_eval()          : 旧版接口，使用 query_db()，输出 Recall@K
  - run_structured_eval(): 新版接口，使用 query_db_structured()，输出 Recall + MRR + Hit + Latency

运行方式：
    uv run python -m ai_app1.eval.evaluate
    uv run python -m ai_app1.eval.evaluate --structured
"""

import json
import sys
import time
from pathlib import Path

# 兼容直接运行（python evaluate.py）和模块运行（python -m ai_app1.eval.evaluate）
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 评测集路径 ─────────────────────────────────────────────────────────────────
_EVAL_FILE = Path(__file__).parent / "评测集"


def _load_dataset() -> list[dict]:
    with open(_EVAL_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    # 过滤掉纯注释条目（只有 json_comment 字段，没有 query）
    return [item for item in raw if "query" in item]


def _key_phrases(text: str, min_len: int = 4) -> list[str]:
    """按标点/空格切分文本，返回长度 >= min_len 的实质性片段。"""
    import re
    segments = re.split(r'[，。？！、：“”‘’（）【】\s]', text)
    return [s.strip() for s in segments if len(s.strip()) >= min_len]


def _ngrams(text: str, n: int = 4) -> list[str]:
    """返回文本中所有长度为 n 的字符滑窗（去除空格后）。"""
    t = text.replace(" ", "")
    return [t[i:i+n] for i in range(len(t) - n + 1)]


def _is_hit(result_text: str, expected_chunk: str, evidence: str) -> tuple[bool, str]:
    """
    Returns (hit: bool, reason: str).

    匹配规则（优先级由高到低）：
      1. evidence 全串精确子串匹配
      2. evidence 拆分子句（>=4 字）任意一个出现在召回文本中
      3. evidence 4-char 滑窗——任意 4 字连续片段出现在召回文本中
         （应对 evidence 是概括/意译、与原文有少量措辞差异的情况）
      4. expected_chunk 标签出现在召回文本中（多跳拆 "/"）
    """
    if not result_text:
        return False, "result_text 为空"

    # 规则 1：evidence 完整子串
    if evidence and evidence in result_text:
        return True, f"evidence 全串命中: {evidence[:40]!r}"

    if evidence:
        # 规则 2：子句短语匹配
        for phrase in _key_phrases(evidence):
            if phrase in result_text:
                return True, f"evidence 短语命中: {phrase!r}"

        # 规则 3：4-char 滑窗匹配（捕获措辞差异下的局部重叠）
        for clause in _key_phrases(evidence, min_len=6):
            for gram in _ngrams(clause, n=4):
                if gram in result_text:
                    return True, f"evidence 4-gram 命中: {gram!r}"

    # 规则 4：expected_chunk 标签（多跳拆分）
    labels = [lbl.strip() for lbl in expected_chunk.split("/")]
    for lbl in labels:
        if lbl and lbl in result_text:
            return True, f"label 命中: {lbl!r}"

    return False, f"MISS — expected={expected_chunk!r}"


# ═══════════════════════════════════════════════════════════════════════════════
# 旧版接口（向后兼容）
# ═══════════════════════════════════════════════════════════════════════════════

def run_eval(top_k_label: str = "5") -> None:
    dataset = _load_dataset()
    total = len(dataset)
    hits = 0

    print(f"\n{'─'*60}")
    print(f"  Retrieval Evaluation   评测集大小: {total} 条")
    print(f"{'─'*60}\n")

    # 延迟导入，避免模型在 import 时就加载
    from ai_app1.service.vector_store import query_db

    for i, item in enumerate(dataset, 1):
        query    = item["query"]
        expected = item.get("expected_chunk", "")
        raw_ev   = item.get("evidence", "")
        # evidence 可能是 str 或 list（多条证据）
        evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

        t0 = time.perf_counter()
        result_text = query_db(query) or ""
        elapsed = (time.perf_counter() - t0) * 1000

        hit, reason = _is_hit(result_text, expected, evidence)
        if hit:
            hits += 1
            status = "✅ HIT"
        else:
            status = "❌ MISS"

        print(f"[{i:02d}/{total}] {status}  ({elapsed:.0f}ms)")
        print(f"  Query   : {query[:60]}")
        print(f"  Expected: {expected}")
        print(f"  Reason  : {reason}")

        if not hit and result_text:
            # MISS 时展示实际召回的前 120 个字，方便分析
            preview = result_text[:120].replace("\n", " ")
            print(f"  Retrieved preview: {preview!r}")

        print()

    recall = hits / total if total else 0.0
    print(f"{'─'*60}")
    print(f"  Recall@{top_k_label}  =  {hits} / {total}  =  {recall:.0%}")
    print(f"{'─'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 新版结构化接口（集成 MRR / Hit@K / Latency）
# ═══════════════════════════════════════════════════════════════════════════════

def run_structured_eval(
    top_k_label: str = "5",
    config=None,
    enable_rewrite: bool = True,
) -> dict:
    """
    结构化召回评测，使用 query_db_structured() 获取完整排序与耗时信息。

    与 run_eval 的区别：
      - 输出 Recall@K + MRR + Hit@1/3/5 + Latency 统计
      - 支持消融配置（config / enable_rewrite）
      - 返回结构化 dict，可用于实验报告

    Returns:
        {
            "recall@k": float,
            "mrr": float,
            "hit@1": float,
            "hit@3": float,
            "hit@5": float,
            "latency_ms": {"mean": ..., "p95": ...},
            "details": [...],
        }
    """
    from ai_app1.service.vector_store import query_db_structured, RetrievalConfig
    from ai_app1.service.query_rewriter import rewrite_queries
    from ai_app1.eval.metrics import aggregate_metrics

    dataset = _load_dataset()
    total = len(dataset)
    cfg = config or RetrievalConfig()

    print(f"\n{'─'*60}")
    print(f"  Structured Retrieval Evaluation   评测集: {total} 条")
    print(f"  Config: {cfg.summary()}")
    print(f"{'─'*60}\n")

    rank_lists = []
    ground_truths = []
    latencies = []
    details = []

    for i, item in enumerate(dataset, 1):
        query = item["query"]
        expected = item.get("expected_chunk", "")
        raw_ev = item.get("evidence", "")
        evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

        # Query 扩写（可控）
        t_rewrite = 0.0
        if enable_rewrite and cfg.enable_rewrite:
            t0 = time.perf_counter()
            queries = rewrite_queries(query, history=[])
            t_rewrite = (time.perf_counter() - t0) * 1000
        else:
            from ai_app1.service.query_rewriter import RewriteQuery
            queries = [RewriteQuery(text=query, type="original", weight=1.0,
                                    routes=["dense", "hyde", "bm25"])]

        result = query_db_structured(queries, config=cfg)
        rank_list = [c.id for c in result.chunks]

        # 命中判断（基于 chunk 文本）
        gt_ids = set()
        for c in result.chunks:
            if _is_hit(c.text, expected, evidence)[0]:
                gt_ids.add(c.id)

        hit = len(gt_ids) > 0
        first_rank = 999
        for idx, doc_id in enumerate(rank_list, start=1):
            if doc_id in gt_ids:
                first_rank = idx
                break

        total_latency = result.latency_ms + t_rewrite
        rank_lists.append(rank_list)
        ground_truths.append(gt_ids)
        latencies.append(total_latency)
        details.append({
            "query": query,
            "hit": hit,
            "rank": first_rank,
            "matched_ids": list(gt_ids),
            "latency_ms": round(total_latency, 2),
        })

        status = "✅ HIT" if hit else "❌ MISS"
        print(f"[{i:02d}/{total}] {status}  rank={first_rank if hit else 'MISS'}  ({total_latency:.0f}ms)")
        print(f"  Query   : {query[:60]}")
        if not hit and result.ordered_text:
            preview = result.ordered_text[:120].replace("\n", " ")
            print(f"  Preview : {preview!r}")
        print()

    metrics = aggregate_metrics(
        rank_lists=rank_lists,
        ground_truths=ground_truths,
        latencies_ms=latencies,
        config_summary=cfg.summary(),
    )

    print(f"{'─'*60}")
    print(f"  Recall@5  = {metrics.recall_at_5:.0%}")
    print(f"  MRR       = {metrics.mrr:.3f}")
    print(f"  Hit@1     = {metrics.hit_at_1:.0%}")
    print(f"  Hit@3     = {metrics.hit_at_3:.0%}")
    print(f"  Hit@5     = {metrics.hit_at_5:.0%}")
    print(f"  平均延迟   = {metrics.latency.mean:.0f}ms  (P95={metrics.latency.p95:.0f}ms)")
    print(f"{'─'*60}\n")

    return {
        "recall@5": metrics.recall_at_5,
        "mrr": metrics.mrr,
        "hit@1": metrics.hit_at_1,
        "hit@3": metrics.hit_at_3,
        "hit@5": metrics.hit_at_5,
        "latency_ms": metrics.to_dict()["latency_ms"],
        "details": details,
    }


# ─── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retrieval Evaluation")
    parser.add_argument("--structured", action="store_true", help="使用结构化评测（含 MRR / Hit@K / Latency）")
    args = parser.parse_args()

    if args.structured:
        run_structured_eval()
    else:
        run_eval()
