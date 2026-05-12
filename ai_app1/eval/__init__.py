"""
ai_app1.eval — RAG Evaluation Platform
======================================

多维度评测体系，支撑 RAG 架构的量化迭代：

  - evaluate        : 传统召回评测（Recall@K）
  - ranking_eval    : 排序质量评测（MRR / Hit@K）
  - ablation        : 消融实验框架（Ablation Study）
  - qa_eval         : 端到端 QA 评测（LLM-as-Judge）
  - experiment      : 实验平台主入口（一键完整报告）
  - metrics         : 统一指标计算库（Recall / MRR / NDCG / Latency）

数据集：
  - 评测集          : 基础评测集（10 条标准 query）
  - hard_cases.json : 困难样本集（指代 / 模糊 / 错别字 / 中英混合 / 超长 / 多跳）
  - qa_benchmark.json: 端到端 QA 评测集（含 expected_answer + 关键点）

使用示例：
    # 1. 快速运行排序评测
    from ai_app1.eval.ranking_eval import run_ranking_eval
    metrics = run_ranking_eval()

    # 2. 运行消融实验
    from ai_app1.eval.ablation import run_ablation_study
    results = run_ablation_study()

    # 3. 一键完整实验
    # uv run python -m ai_app1.eval.experiment all

    # 4. 自定义配置评测
    from ai_app1.retrieval.vector_store import RetrievalConfig
    from ai_app1.eval.ranking_eval import run_ranking_eval
    cfg = RetrievalConfig(enable_hyde=False)
    metrics = run_ranking_eval(config=cfg)
"""

# 导出核心类型与函数，方便上层调用
from ai_app1.eval.metrics import (
    EvalMetrics,
    LatencyStats,
    aggregate_metrics,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    hit_at_k,
    ndcg_at_k,
    compute_latency_stats,
)

__all__ = [
    "EvalMetrics",
    "LatencyStats",
    "aggregate_metrics",
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "hit_at_k",
    "ndcg_at_k",
    "compute_latency_stats",
]
