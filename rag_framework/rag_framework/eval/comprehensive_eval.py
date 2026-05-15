"""
RAG 综合评测调度器（Comprehensive Eval）

整合所有评测维度的一站式调度器：
  1. Retrieval Ranking      → run_ranking_eval
  2. Query Classification   → 按类型统计 recall
  3. Rewrite Evaluation     → rewrite 前后对比
  4. Rerank Evaluation      → CrossEncoder top1 验证
  5. Ablation Study         → 消融实验
  6. Hard Cases             → 困难样本专项
  7. End-to-End QA          → run_qa_eval
  8. Latency Breakdown      → 阶段耗时分析
  9. Failure Analysis       → 失败样本收集

输出：统一 Markdown + JSON 报告。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Callable

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.eval.ablation import run_ablation_study
from rag_framework.eval.failure_analysis import FailureStore, get_failure_collector
from rag_framework.eval.latency_breakdown import PhaseLatency, aggregate_phase_latencies
from rag_framework.eval.metrics import EvalMetrics, LatencyStats
from rag_framework.eval.qa import run_qa_eval
from rag_framework.eval.query_classifier import aggregate_by_type, format_type_stats
from rag_framework.eval.ragas_eval import run_ragas_eval
from rag_framework.eval.ranking import run_ranking_eval
from rag_framework.eval.rerank_eval import run_rerank_eval
from rag_framework.eval.rewrite_eval import run_rewrite_eval


_REPORT_DIR = Path("reports")
_REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ─── 各评测模块包装 ─────────────────────────────────────────────────────────────

async def _run_ranking(container: RAGContainer | None = None) -> dict:
    metrics = await run_ranking_eval(container=container, verbose=False)
    return {"name": "ranking", "metrics": metrics.to_dict()}


async def _run_ablation(container: RAGContainer | None = None) -> dict:
    results = await run_ablation_study(container=container, verbose_per_query=False)
    return {"name": "ablation", "experiments": results}


async def _run_hard_cases(
    container: RAGContainer | None = None,
    hard_path: Path | None = None,
) -> dict:
    metrics = await run_ranking_eval(
        container=container, dataset_path=hard_path, verbose=False
    )
    return {"name": "hard_cases", "metrics": metrics.to_dict()}


async def _run_rewrite(
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
) -> dict:
    report = await run_rewrite_eval(
        dataset_path=dataset_path, container=container, verbose=False
    )
    return {"name": "rewrite", "report": report}


async def _run_rerank(
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
) -> dict:
    report = await run_rerank_eval(
        dataset_path=dataset_path, container=container, verbose=False
    )
    return {"name": "rerank", "report": report}


async def _run_qa(
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
) -> dict:
    report = await run_qa_eval(
        dataset_path=dataset_path, container=container, verbose=False
    )
    return {"name": "qa", "report": report}


async def _run_ragas(
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
) -> dict:
    report = await run_ragas_eval(
        dataset_path=dataset_path, container=container, verbose=False
    )
    return {"name": "ragas", "report": report}


# ─── 报告生成 ───────────────────────────────────────────────────────────────────

def _latency_from_dict(d: dict) -> LatencyStats:
    return LatencyStats(
        mean=d.get("mean", 0.0),
        std=d.get("std", 0.0),
        p50=d.get("p50", 0.0),
        p95=d.get("p95", 0.0),
        p99=d.get("p99", 0.0),
        min=d.get("min", 0.0),
        max=d.get("max", 0.0),
        count=d.get("count", 0),
    )


def _generate_full_report(results: list[dict]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# RAG Comprehensive Evaluation Report\n",
        f"**生成时间**: {ts}  ",
        "**项目**: RAG Framework Evaluation Platform\n",
        "---\n",
    ]

    for r in results:
        name = r["name"]
        lines.append(f"## {name.upper()}\n")

        if name in ("ranking", "hard_cases"):
            m = r.get("metrics", {})
            lines.append(f"- **Recall@5**: {m.get('recall@5', 0):.1%}")
            lines.append(f"- **MRR**: {m.get('mrr', 0):.3f}")
            lines.append(f"- **Hit@1**: {m.get('hit@1', 0):.1%}")
            lines.append(f"- **Hit@3**: {m.get('hit@3', 0):.1%}")
            lines.append(f"- **Hit@5**: {m.get('hit@5', 0):.1%}")
            lat = m.get("latency_ms", {})
            lines.append(
                f"- **平均延迟**: {lat.get('mean', 0):.0f}ms "
                f"(P95={lat.get('p95', 0):.0f}ms)"
            )

        elif name == "ablation":
            lines.append(EvalMetrics.markdown_header())
            for exp in r.get("experiments", []):
                md = EvalMetrics.from_dict(exp["metrics"])
                md.latency = _latency_from_dict(exp["metrics"]["latency_ms"])
                md.config_summary = exp["config"]
                lines.append(md.to_markdown_row())

        elif name == "rewrite":
            rep = r.get("report", {})
            lines.append(f"- **评测条数**: {rep.get('total', 0)}")
            lines.append(
                f"- **提升/下降/持平**: {rep.get('improved', 0)} / "
                f"{rep.get('degraded', 0)} / {rep.get('unchanged', 0)}"
            )
            lines.append(f"- **ΔRecall@5**: {rep.get('avg_delta_recall', 0):+.4f}")
            lines.append(f"- **ΔHit@1**: {rep.get('avg_delta_hit1', 0):+.4f}")
            lines.append(f"- **ΔMRR**: {rep.get('avg_delta_mrr', 0):+.4f}")

        elif name == "rerank":
            rep = r.get("report", {})
            lines.append(f"- **评测条数**: {rep.get('total', 0)}")
            lines.append(
                f"- **Win/Loss/Hold**: {rep.get('wins', 0)} / "
                f"{rep.get('losses', 0)} / {rep.get('holds', 0)}"
            )
            lines.append(f"- **Win Rate**: {rep.get('win_rate', 0):.1%}")
            lines.append(f"- **Loss Rate**: {rep.get('loss_rate', 0):.1%}")
            lines.append(
                f"- **平均排名变化**: {rep.get('avg_rank_delta', 0):+.1f}（负数=上升）"
            )

        elif name == "qa":
            rep = r.get("report", {})
            lines.append(f"- **平均 Overall**: {rep.get('avg_overall', 0):.3f}")
            lines.append(f"- **平均 Coverage**: {rep.get('avg_coverage', 0):.3f}")
            lines.append(
                f"- **平均 Hallucination**: {rep.get('avg_hallucination', 0):.3f}"
            )
            lines.append(f"- **平均延迟**: {rep.get('avg_latency_ms', 0):.0f}ms")

        elif name == "ragas":
            rep = r.get("report", {})
            summary = rep.get("summary", {})
            lines.append(f"- **模式**: {rep.get('mode', 'unknown')}")
            lines.append(f"- **Faithfulness**: {summary.get('faithfulness', 0):.3f}")
            lines.append(f"- **Answer Relevancy**: {summary.get('answer_relevancy', 0):.3f}")
            lines.append(f"- **Context Recall**: {summary.get('context_recall', 0):.3f}")
            lines.append(f"- **Context Precision**: {summary.get('context_precision', 0):.3f}")
            lines.append(f"- **Overall**: {summary.get('overall', 0):.3f}")

        lines.append("")

    lines.append("---\n")
    lines.append("*Report generated by rag_framework.eval.comprehensive_eval*\n")
    return "\n".join(lines)


# ─── 主调度器 ───────────────────────────────────────────────────────────────────

async def run_comprehensive_eval(
    command: str = "all",
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
    hard_cases_path: Path | None = None,
    qa_dataset_path: Path | None = None,
    enable_failure_store: bool = True,
) -> dict:
    """
    综合评测主调度器。

    Args:
        command: ranking | ablation | hard | rewrite | rerank | qa | all
        container: RAG 容器
        dataset_path: 评测集路径（rewrite / rerank 使用）
        hard_cases_path: 困难样本评测集路径
        qa_dataset_path: QA 评测集路径
        enable_failure_store: 是否启用失败样本收集

    Returns:
        完整报告字典
    """
    print(f"\n{'='*70}")
    print(f"  RAG Comprehensive Evaluation   command={command}")
    print(f"{'='*70}\n")

    start = time.perf_counter()
    results: list[dict] = []

    runners: dict[str, Callable] = {
        "ranking": lambda: _run_ranking(container),
        "ablation": lambda: _run_ablation(container),
        "hard": lambda: _run_hard_cases(container, hard_cases_path),
        "rewrite": lambda: _run_rewrite(container, dataset_path),
        "rerank": lambda: _run_rerank(container, dataset_path),
        "qa": lambda: _run_qa(container, qa_dataset_path),
        "ragas": lambda: _run_ragas(container, qa_dataset_path),
    }

    if command == "all":
        # 默认全量运行（不含 qa/ragas，因为需要 async + LLM-as-Judge）
        commands = ["ranking", "rewrite", "rerank", "ablation", "hard", "qa", "ragas"]
    else:
        commands = [command]

    for cmd in commands:
        print(f"▶ 执行: {cmd} ...")
        try:
            if cmd in runners:
                res = await runners[cmd]()
            else:
                print(f"  ⚠ 未知命令: {cmd}，跳过")
                continue
            results.append(res)
            print(f"  ✓ 完成\n")
        except Exception as e:
            print(f"  ✗ 失败: {e}\n")
            results.append({"name": cmd, "error": str(e)})

    total_elapsed = (time.perf_counter() - start) * 1000

    # 尝试加载 failure store 汇总
    failure_summary = {}
    if enable_failure_store:
        try:
            store = FailureStore()
            failure_summary = store.summary()
            print(store.print_summary())
        except Exception as e:
            eval_logger.warning(f"FailureStore 汇总失败: {e}")

    # 生成报告
    md_report = _generate_full_report(results)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = _REPORT_DIR / f"comprehensive_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    full_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_ms": round(total_elapsed, 2),
        "failure_summary": failure_summary,
        "results": results,
    }
    json_path = _REPORT_DIR / f"comprehensive_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"{'='*70}")
    print(f"  评测完成  总耗时: {total_elapsed:.0f}ms")
    print(f"  报告已保存:")
    print(f"    Markdown: {md_path}")
    print(f"    JSON:     {json_path}")
    print(f"{'='*70}\n")

    print(md_report)
    return full_report


# ─── CLI 入口 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG Comprehensive Evaluation — 生产级多维度评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m rag_framework.eval.comprehensive_eval ranking
  python -m rag_framework.eval.comprehensive_eval rewrite
  python -m rag_framework.eval.comprehensive_eval rerank
  python -m rag_framework.eval.comprehensive_eval all
        """,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["ranking", "ablation", "hard", "rewrite", "rerank", "qa", "ragas", "all"],
        help="要执行的评测命令（默认 all）",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="评测集路径（rewrite / rerank 使用）",
    )
    parser.add_argument(
        "--hard-cases",
        type=Path,
        dest="hard_cases",
        default=None,
        help="困难样本评测集路径",
    )
    parser.add_argument(
        "--qa-dataset",
        type=Path,
        dest="qa_dataset",
        default=None,
        help="QA 评测集路径",
    )
    args = parser.parse_args()
    asyncio.run(
        run_comprehensive_eval(
            command=args.command,
            dataset_path=args.dataset,
            hard_cases_path=args.hard_cases,
            qa_dataset_path=args.qa_dataset,
        )
    )


if __name__ == "__main__":
    main()
