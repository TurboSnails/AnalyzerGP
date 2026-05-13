"""
通用消融实验框架（Ablation Study）

通过系统性地关闭 pipeline 中的每个模块，量化各模块对检索质量的贡献。
与领域无关，通过 RAGContainer 获取组件，通过 HybridConfig 控制开关。
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Callable

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.eval.metrics import EvalMetrics
from rag_framework.eval.ranking import run_ranking_eval
from rag_framework.retrieval.fusion import HybridConfig


# ─── 实验配置定义 ───────────────────────────────────────────────────────────────

def _baseline_cfg() -> HybridConfig:
    """完整 pipeline（Baseline）。"""
    return HybridConfig()


def _ablation_configs() -> list[tuple[str, HybridConfig]]:
    """返回所有消融实验配置（名称, 配置）。"""
    baseline = _baseline_cfg()
    return [
        ("baseline", baseline),
        ("no_hyde", HybridConfig(enable_hyde=False)),
        ("no_bm25", HybridConfig(enable_bm25=False)),
        ("no_rerank", HybridConfig(enable_rerank=False)),
        ("no_lost_in_middle", HybridConfig(enable_lost_in_middle=False)),
        ("no_hyde_no_rerank", HybridConfig(enable_hyde=False, enable_rerank=False)),
        ("only_dense", HybridConfig(enable_hyde=False, enable_bm25=False, enable_rerank=False, enable_lost_in_middle=False)),
    ]


# ─── 单实验运行 ─────────────────────────────────────────────────────────────────

def run_single_experiment(
    name: str,
    config: HybridConfig,
    container: RAGContainer | None = None,
    dataset_path: Path | None = None,
    enable_rewrite: bool = True,
    verbose: bool = False,
) -> EvalMetrics:
    return run_ranking_eval(
        dataset_path=dataset_path,
        container=container,
        config=config,
        enable_rewrite=enable_rewrite,
        verbose=verbose,
    )


# ─── 批量消融实验 ───────────────────────────────────────────────────────────────

def run_ablation_study(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    output_dir: Path | None = None,
    verbose_per_query: bool = False,
) -> list[dict]:
    """
    执行完整消融实验，输出对比报告。

    Returns:
        每条实验结果的 dict 列表（含 config / metrics / 时间戳）
    """
    if output_dir is None:
        output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _ablation_configs()
    results: list[dict] = []

    print(f"\n{'='*70}")
    print(f"  Ablation Study   实验数: {len(configs)}")
    print(f"{'='*70}\n")

    for name, cfg in configs:
        summary = f"{name} ({cfg.enable_hyde=}, {cfg.enable_bm25=}, {cfg.enable_rerank=}, {cfg.enable_lost_in_middle=})"
        print(f"▶ 运行实验: {name}  ...  ({summary})")
        t0 = time.perf_counter()
        metrics = run_single_experiment(
            name=name,
            config=cfg,
            container=container,
            dataset_path=dataset_path,
            verbose=verbose_per_query,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        record = {
            "experiment": name,
            "config": summary,
            "metrics": metrics.to_dict(),
            "total_time_ms": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(record)

        print(f"  ✓ 完成  Recall@5={metrics.recall_at_5:.0%}  MRR={metrics.mrr:.3f}  "
              f"Hit@1={metrics.hit_at_1:.0%}  耗时={elapsed:.0f}ms\n")

        gc.collect()

    # Markdown 报告
    md_lines = [
        "# RAG Ablation Study Report\n",
        f"**评测集**: `{dataset_path.name if dataset_path else 'domain'}`  |  **实验数**: {len(configs)}  |  **时间**: {results[0]['timestamp']}\n",
        EvalMetrics.markdown_header(),
    ]
    for r in results:
        m = EvalMetrics.from_dict(r["metrics"])
        m.config_summary = r["config"]
        md_lines.append(m.to_markdown_row())

    md_lines.append("")
    md_lines.append("## 关键结论\n")
    md_lines.append(_generate_insights(results))

    md_path = output_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    json_path = output_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{'='*70}")
    print(f"  报告已保存:")
    print(f"    Markdown: {md_path}")
    print(f"    JSON:     {json_path}")
    print(f"{'='*70}\n")

    return results


# ─── 辅助函数 ───────────────────────────────────────────────────────────────────

def _generate_insights(results: list[dict]) -> str:
    """根据实验结果自动生成关键结论。"""
    baseline = next((r for r in results if r["experiment"] == "baseline"), None)
    if baseline is None:
        return "- Baseline 未找到，无法生成对比结论。\n"

    baseline_mrr = baseline["metrics"]["mrr"]
    baseline_latency = baseline["metrics"]["latency_ms"]["mean"]

    insights = []

    # 最大负面影响
    worst = min(results, key=lambda r: r["metrics"]["mrr"])
    if worst["experiment"] != "baseline":
        drop = baseline_mrr - worst["metrics"]["mrr"]
        insights.append(
            f"- **最大负面影响**: `{worst['experiment']}` 导致 MRR 下降 {drop:.3f} "
            f"(从 {baseline_mrr:.3f} → {worst['metrics']['mrr']:.3f})，"
            f"说明该模块对排序质量至关重要。"
        )

    # 延迟-质量权衡
    pareto = [
        r for r in results
        if r["metrics"]["mrr"] >= baseline_mrr * 0.95
        and r["metrics"]["latency_ms"]["mean"] < baseline_latency
    ]
    if pareto:
        names = ", ".join(f"`{r['experiment']}`" for r in pareto)
        insights.append(
            f"- **延迟-质量权衡**: {names} 在保持 MRR ≥ 95% baseline 的前提下降低了延迟，"
            f"可考虑作为线上轻量配置。"
        )

    # rerank 的价值
    no_rerank = next((r for r in results if r["experiment"] == "no_rerank"), None)
    if no_rerank:
        delta = baseline_mrr - no_rerank["metrics"]["mrr"]
        if delta > 0.03:
            insights.append(
                f"- **CrossEncoder Rerank 有效**: 关闭 rerank 后 MRR 下降 {delta:.3f}，"
                f"精排模块对提升 Top-1 命中率有直接帮助。"
            )

    if not insights:
        insights.append("- 各模块对指标影响较小，建议增加评测集规模或引入更困难的样本。")

    return "\n".join(insights) + "\n"
