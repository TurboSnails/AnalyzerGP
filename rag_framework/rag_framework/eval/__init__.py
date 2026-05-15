from rag_framework.eval.ablation import run_ablation_study
from rag_framework.eval.comprehensive_eval import run_comprehensive_eval
from rag_framework.eval.experiment import run_experiment
from rag_framework.eval.failure_analysis import (
    FailureCase,
    FailureCollector,
    FailureStore,
    get_failure_collector,
)
from rag_framework.eval.hit_judge import ground_truth_ids, is_hit, key_phrases, ngrams
from rag_framework.eval.latency_breakdown import (
    LatencyBreakdownReport,
    PhaseLatency,
    PhaseTimer,
    aggregate_phase_latencies,
)
from rag_framework.eval.metrics import (
    EvalMetrics,
    LatencyStats,
    aggregate_metrics,
    compute_latency_stats,
    dcg_at_k,
    hit_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from rag_framework.eval.qa import QAJudge, evaluate_single_qa, generate_answer, run_qa_eval
from rag_framework.eval.query_classifier import (
    aggregate_by_type,
    classify_query_type,
    classify_query_type_from_item,
    format_type_stats,
)
from rag_framework.eval.ranking import evaluate_single_query, load_dataset, run_ranking_eval
from rag_framework.eval.rerank_eval import run_rerank_eval
from rag_framework.eval.retrieval_trace import (
    BranchTrace,
    RerankTrace,
    RetrievalTrace,
    get_recent_traces,
    get_traces_by_query,
    record_trace,
)
from rag_framework.eval.rewrite_eval import run_rewrite_eval
from rag_framework.eval.ragas_eval import (
    RagasEvaluator,
    RagasMetrics,
    run_ragas_eval,
)

__all__ = [
    # metrics
    "recall_at_k",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "hit_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "percentile",
    "compute_latency_stats",
    "LatencyStats",
    "EvalMetrics",
    "aggregate_metrics",
    # hit judge
    "is_hit",
    "key_phrases",
    "ngrams",
    "ground_truth_ids",
    # ranking
    "load_dataset",
    "evaluate_single_query",
    "run_ranking_eval",
    # qa
    "generate_answer",
    "QAJudge",
    "evaluate_single_qa",
    "run_qa_eval",
    # ablation / experiment / comprehensive
    "run_ablation_study",
    "run_experiment",
    "run_comprehensive_eval",
    # query classifier
    "classify_query_type",
    "classify_query_type_from_item",
    "aggregate_by_type",
    "format_type_stats",
    # rewrite eval
    "run_rewrite_eval",
    # rerank eval
    "run_rerank_eval",
    # retrieval trace
    "RetrievalTrace",
    "BranchTrace",
    "RerankTrace",
    "record_trace",
    "get_recent_traces",
    "get_traces_by_query",
    # latency breakdown
    "PhaseLatency",
    "PhaseTimer",
    "LatencyBreakdownReport",
    "aggregate_phase_latencies",
    # ragas
    "RagasEvaluator",
    "RagasMetrics",
    "run_ragas_eval",
    # failure analysis
    "FailureCase",
    "FailureCollector",
    "FailureStore",
    "get_failure_collector",
    "set_failure_collector",
    # failure triage
    "generate_report",
    "TriageEngine",
    "ROOT_CAUSE_LABELS",
]
