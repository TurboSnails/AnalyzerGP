"""
Ablation Study — 消融实验框架
==============================

通过系统性地关闭 pipeline 中的每个模块，量化各模块对检索质量的贡献。

实验设计（对照 baseline + 单因素消融）：
  1. Baseline（全模块开启）
  2. -rewrite     : 关闭 query 扩写，只用原始 query
  3. -hyde        : 关闭 HyDE 召回路径
  4. -bm25        : 关闭 BM25 稀疏检索
  5. -rerank      : 关闭 CrossEncoder 精排（保留 RRF 排序）
  6. -LiM         : 关闭 Lost-in-Middle 重排
  7. +组合消融    : 关闭 rewrite + rerank（验证最差情况）

输出格式：
  Markdown 表格 + JSON 报告，可直接写入实验记录。

运行方式：
    uv run python -m ai_app1.eval.ablation
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path
from typing import Iterable

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_app1.retrieval.vector_store import RetrievalConfig
from ai_app1.eval.ranking_eval import run_ranking_eval, _EVAL_FILE
from ai_app1.eval.metrics import EvalMetrics


# ─── 实验配置定义 ─────────────────────────────────────────────────────────────

def _baseline_cfg() -> RetrievalConfig:
    """完整 pipeline（Baseline）。"""
    return RetrievalConfig()


def _ablation_configs() -> list[tuple[str, RetrievalConfig]]:
    """
    返回所有消融实验配置（名称, 配置）。

    命名约定：
      - baseline : 全开
      - no_xxx   : 单因素关闭 xxx
      - only_xxx : 仅保留 xxx（极端情况）
    """
    baseline = _baseline_cfg()
    return [
        ("baseline", baseline),
        ("no_rewrite", RetrievalConfig(enable_rewrite=False)),
        ("no_hyde", RetrievalConfig(enable_hyde=False)),
        ("no_bm25", RetrievalConfig(enable_bm25=False)),
        ("no_rerank", RetrievalConfig(enable_rerank=False)),
        ("no_lost_in_middle", RetrievalConfig(enable_lost_in_middle=False)),
        ("no_rewrite_no_rerank", RetrievalConfig(enable_rewrite=False, enable_rerank=False)),
        ("only_dense", RetrievalConfig(enable_hyde=False, enable_bm25=False, enable_rerank=False, enable_lost_in_middle=False)),
    ]


# ─── 单实验运行 ───────────────────────────────────────────────────────────────

def run_single_experiment(
    name: str,
    config: RetrievalConfig,
    dataset_path: Path = _EVAL_FILE,
    enable_rewrite: bool = True,
    verbose: bool = False,
) -> EvalMetrics:
    """
    运行单次消融实验。

    Args:
        name           : 实验名称（用于报告）
        config         : 检索配置
        dataset_path   : 评测集路径
        enable_rewrite : 是否在外层启用 query 扩写（与 config.enable_rewrite 联动）
        verbose        : 是否打印逐条详情

    Returns:
        EvalMetrics
    """
    # 如果 config 本身关闭了 rewrite，外层也同步关闭
    actual_rewrite = enable_rewrite and config.enable_rewrite
    return run_ranking_eval(
        dataset_path=dataset_path,
        config=config,
        enable_rewrite=actual_rewrite,
        verbose=verbose,
    )


# ─── 批量消融实验 ─────────────────────────────────────────────────────────────

def run_ablation_study(
    dataset_path: Path = _EVAL_FILE,
    output_dir: Path | None = None,
    verbose_per_query: bool = False,
) -> list[dict]:
    """
    执行完整消融实验，输出对比报告。

    Args:
        dataset_path    : 评测集路径
        output_dir      : 报告输出目录，默认 eval/ 同级目录
        verbose_per_query: 是否打印每个 query 的详细结果（建议 False，否则输出过长）

    Returns:
        每条实验结果的 dict 列表（含 config / metrics / 时间戳）
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _ablation_configs()
    results: list[dict] = []

    print(f"\n{'='*70}")
    print(f"  Ablation Study   实验数: {len(configs)}")
    print(f"{'='*70}\n")

    for name, cfg in configs:
        print(f"▶ 运行实验: {name}  ...  ({cfg.summary()})")
        t0 = time.perf_counter()
        metrics = run_single_experiment(
            name=name,
            config=cfg,
            dataset_path=dataset_path,
            verbose=verbose_per_query,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        record = {
            "experiment": name,
            "config": cfg.summary(),
            "metrics": metrics.to_dict(),
            "total_time_ms": round(elapsed, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        results.append(record)

        print(f"  ✓ 完成  Recall@5={metrics.recall_at_5:.0%}  MRR={metrics.mrr:.3f}  "
              f"Hit@1={metrics.hit_at_1:.0%}  耗时={elapsed:.0f}ms\n")

        gc.collect()

    # ── Markdown 报告 ────────────────────────────────────────────────────────
    md_lines = [
        "# RAG Ablation Study Report\n",
        f"**评测集**: `{dataset_path.name}`  |  **实验数**: {len(configs)}  |  **时间**: {results[0]['timestamp']}\n",
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

    # ── JSON 原始数据 ────────────────────────────────────────────────────────
    json_path = output_dir / f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"{'='*70}")
    print(f"  报告已保存:")
    print(f"    Markdown: {md_path}")
    print(f"    JSON:     {json_path}")
    print(f"{'='*70}\n")

    return results


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def _generate_insights(results: list[dict]) -> str:
    """
    根据实验结果自动生成关键结论。

    对比 baseline，找出 MRR 下降最多的配置，以及延迟最低的 Pareto 最优解。
    """
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

    # rewrite 的价值
    no_rewrite = next((r for r in results if r["experiment"] == "no_rewrite"), None)
    if no_rewrite:
        delta = baseline_mrr - no_rewrite["metrics"]["mrr"]
        if delta > 0.05:
            insights.append(
                f"- **Query Rewrite 价值显著**: 关闭 rewrite 后 MRR 下降 {delta:.3f}，"
                f"rewrite 对查询语义扩展贡献明显。"
            )
        elif delta < 0.02:
            insights.append(
                f"- **Query Rewrite 收益有限**: 关闭 rewrite 仅导致 MRR 下降 {delta:.3f}，"
                f"可考虑在延迟敏感场景关闭以节省 TTFT。"
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


# ─── 主入口 ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_ablation_study()
