"""
RAGAS 风格端到端评测模块

支持两种运行模式：
  1. Native 模式：使用原生 ragas 库计算指标（Faithfulness / Answer Relevancy / Context Recall / Context Precision）
  2. Fallback 模式：当 ragas 不可用时，使用 LLM-as-Judge 模拟等效指标

依赖（可选）：
  pip install ragas

输出：reports/ragas_score.md 与 JSON
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.eval.qa import generate_answer, load_qa_dataset
from rag_framework.llm.base import LLMClient

# ─── 尝试导入 ragas ─────────────────────────────────────────────────────────────

_RAGAS_AVAILABLE = False
try:
    from ragas.metrics import (
        Faithfulness,
        AnswerRelevancy,
        ContextRecall,
        ContextPrecision,
    )
    from ragas import evaluate as ragas_evaluate
    _RAGAS_AVAILABLE = True
except Exception:
    pass


# ─── Fallback Judge Prompts（ragas 不可用时使用）───────────────────────────────

_FAITHFULNESS_PROMPT = """\
你是 RAG 评测专家。请判断【模型回答】是否忠实于【检索上下文】。

评分标准（0~1，保留两位小数）：
- 1.0：回答中的所有事实均能在上下文中找到依据，无任何虚构。
- 0.5：部分事实有依据，但存在少量过度推断或无法验证的内容。
- 0.0：回答中包含上下文未提及的虚构信息，或明显与上下文矛盾。

仅输出一个 0~1 之间的数字，不要任何解释。
"""

_ANSWER_RELEVANCY_PROMPT = """\
你是 RAG 评测专家。请判断【模型回答】对【用户问题】的相关程度。

评分标准（0~1，保留两位小数）：
- 1.0：回答精准、完整地回答了用户问题，无冗余。
- 0.5：回答部分相关，但包含一些无关信息或遗漏了关键要点。
- 0.0：回答完全跑题或与问题无关。

仅输出一个 0~1 之间的数字，不要任何解释。
"""

_CONTEXT_RECALL_PROMPT = """\
你是 RAG 评测专家。请判断【检索上下文】是否覆盖了【参考答案】中的所有关键要点。

评分标准（0~1，保留两位小数）：
- 1.0：上下文包含了参考答案中的全部关键信息。
- 0.5：上下文包含部分关键信息，但遗漏了若干要点。
- 0.0：上下文几乎没有包含参考答案的关键信息。

仅输出一个 0~1 之间的数字，不要任何解释。
"""

_CONTEXT_PRECISION_PROMPT = """\
你是 RAG 评测专家。请判断【检索上下文】中有多少内容是与【用户问题】真正相关的。

评分标准（0~1，保留两位小数）：
- 1.0：上下文中的每一句话都与问题直接相关，无噪声。
- 0.5：上下文包含部分相关信息，但也混入了不少无关内容。
- 0.0：上下文几乎全是噪声，与问题无关。

仅输出一个 0~1 之间的数字，不要任何解释。
"""


# ─── RAGAS Evaluator ───────────────────────────────────────────────────────────

@dataclass
class RagasMetrics:
    """RAGAS 风格四维度指标。"""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevancy": round(self.answer_relevancy, 3),
            "context_recall": round(self.context_recall, 3),
            "context_precision": round(self.context_precision, 3),
            "overall": round(self.overall, 3),
        }


class RagasEvaluator:
    """
    RAGAS 风格评测器。

    如果 ragas 库可用则走 native 路径；否则使用 LLM-as-Judge fallback。
    """

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm
        self._native = _RAGAS_AVAILABLE
        if not self._native:
            eval_logger.info("ragas 库未安装，启用 LLM-as-Judge fallback 模式")

    async def _judge_score(self, prompt: str, context: str) -> float:
        """使用 LLM 输出 0~1 分数。"""
        if self._llm is None:
            return 0.0
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": context},
        ]
        try:
            raw = await self._llm.chat(messages, use_tools=False)
            # 提取第一个看起来像数字的部分
            for token in raw.replace(",", " ").split():
                try:
                    return max(0.0, min(1.0, float(token)))
                except ValueError:
                    continue
            return 0.0
        except Exception as e:
            eval_logger.warning(f"Judge 评分失败: {e}")
            return 0.0

    async def evaluate_native(
        self,
        dataset: list[dict],
    ) -> list[RagasMetrics]:
        """使用原生 ragas 库评测。"""
        from datasets import Dataset as HFDataset  # type: ignore[import-untyped]

        ragas_data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        for item in dataset:
            ragas_data["question"].append(item["query"])
            ragas_data["answer"].append(item.get("model_answer", ""))
            ragas_data["contexts"].append(item.get("contexts", []))
            ragas_data["ground_truth"].append(item.get("expected_answer", ""))

        hf_dataset = HFDataset.from_dict(ragas_data)
        result = ragas_evaluate(
            hf_dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextRecall(),
                ContextPrecision(),
            ],
        )
        scores = result.to_pandas()
        metrics_list: list[RagasMetrics] = []
        for _, row in scores.iterrows():
            m = RagasMetrics(
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
            )
            m.overall = round(
                (m.faithfulness + m.answer_relevancy + m.context_recall + m.context_precision) / 4, 3
            )
            metrics_list.append(m)
        return metrics_list

    async def evaluate_fallback(
        self,
        dataset: list[dict],
    ) -> list[RagasMetrics]:
        """使用 LLM-as-Judge fallback 评测。"""
        metrics_list: list[RagasMetrics] = []
        for item in dataset:
            query = item["query"]
            answer = item.get("model_answer", "")
            contexts = item.get("contexts", [])
            expected = item.get("expected_answer", "")
            context_text = "\n\n".join(contexts)

            faithfulness = await self._judge_score(
                _FAITHFULNESS_PROMPT,
                f"【检索上下文】\n{context_text}\n\n【模型回答】\n{answer}",
            )
            answer_relevancy = await self._judge_score(
                _ANSWER_RELEVANCY_PROMPT,
                f"【用户问题】\n{query}\n\n【模型回答】\n{answer}",
            )
            context_recall = await self._judge_score(
                _CONTEXT_RECALL_PROMPT,
                f"【参考答案】\n{expected}\n\n【检索上下文】\n{context_text}",
            )
            context_precision = await self._judge_score(
                _CONTEXT_PRECISION_PROMPT,
                f"【用户问题】\n{query}\n\n【检索上下文】\n{context_text}",
            )

            m = RagasMetrics(
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_recall=context_recall,
                context_precision=context_precision,
            )
            m.overall = round(
                (faithfulness + answer_relevancy + context_recall + context_precision) / 4, 3
            )
            metrics_list.append(m)
        return metrics_list

    async def evaluate(self, dataset: list[dict]) -> list[RagasMetrics]:
        if self._native:
            return await self.evaluate_native(dataset)
        return await self.evaluate_fallback(dataset)


# ─── 端到端 RAGAS 评测 ──────────────────────────────────────────────────────────

async def run_ragas_eval(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    端到端 RAGAS 风格评测。

    流程：
      1. 对每条 query 调用完整 RAG pipeline 生成 answer + contexts
      2. 使用 RAGAS（或 fallback）计算四维度指标
      3. 输出 Markdown + JSON 报告到 reports/ragas_score.md

    Args:
        dataset_path: QA 评测集路径（需含 expected_answer）
        container: RAG 容器
        max_samples: 最大评测条数
        verbose: 是否打印逐条结果

    Returns:
        综合报告字典
    """
    if container is None:
        container = RAGContainer.from_settings(get_settings())

    if dataset_path is None:
        dataset = container.domain.get_eval_dataset()
        if not dataset:
            eval_logger.warning("Domain 未提供评测集，返回空报告")
            return {}
    else:
        dataset = load_qa_dataset(dataset_path)

    if max_samples:
        dataset = dataset[:max_samples]
    total = len(dataset)

    if verbose:
        name = dataset_path.name if dataset_path else "domain_eval"
        mode = "native" if _RAGAS_AVAILABLE else "fallback"
        print(f"\n{'─'*70}")
        print(f"  RAGAS Evaluation   评测集: {name}  共 {total} 条  模式: {mode}")
        print(f"{'─'*70}\n")

    # 生成 answer + contexts
    enriched_dataset: list[dict] = []
    for i, item in enumerate(dataset, 1):
        query = item["query"]
        expected = item.get("expected_answer", "")

        t0 = time.perf_counter()
        answer = await generate_answer(query, container)
        gen_latency = (time.perf_counter() - t0) * 1000

        # 提取检索上下文（通过 retriever 直接调用获取）
        route = container.domain.classify_query(query, [])
        retrieval_result = await container.retriever.retrieve([route], top_k=5)
        contexts = [d.text for d in retrieval_result.docs]

        enriched_dataset.append({
            "query": query,
            "expected_answer": expected,
            "model_answer": answer,
            "contexts": contexts,
        })

        if verbose:
            print(f"[{i:02d}/{total}] gen_latency={gen_latency:.0f}ms  {query[:50]}...")

    # RAGAS 评分
    evaluator = RagasEvaluator(llm=container.llm)
    t1 = time.perf_counter()
    metrics_list = await evaluator.evaluate(enriched_dataset)
    eval_latency = (time.perf_counter() - t1) * 1000

    # 聚合
    faithfulness_vals = [m.faithfulness for m in metrics_list]
    relevancy_vals = [m.answer_relevancy for m in metrics_list]
    recall_vals = [m.context_recall for m in metrics_list]
    precision_vals = [m.context_precision for m in metrics_list]
    overall_vals = [m.overall for m in metrics_list]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "faithfulness": round(_mean(faithfulness_vals), 3),
        "answer_relevancy": round(_mean(relevancy_vals), 3),
        "context_recall": round(_mean(recall_vals), 3),
        "context_precision": round(_mean(precision_vals), 3),
        "overall": round(_mean(overall_vals), 3),
    }

    report = {
        "dataset": str(dataset_path) if dataset_path else "domain",
        "total_samples": total,
        "mode": "native" if _RAGAS_AVAILABLE else "fallback",
        "summary": summary,
        "details": [
            {
                "query": d["query"],
                "scores": m.to_dict(),
            }
            for d, m in zip(enriched_dataset, metrics_list)
        ],
        "eval_latency_ms": round(eval_latency, 2),
    }

    # 生成 Markdown 报告
    _generate_ragas_markdown(report)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  RAGAS 综合报告")
        print(f"    Faithfulness       = {summary['faithfulness']:.3f}")
        print(f"    Answer Relevancy   = {summary['answer_relevancy']:.3f}")
        print(f"    Context Recall     = {summary['context_recall']:.3f}")
        print(f"    Context Precision  = {summary['context_precision']:.3f}")
        print(f"    Overall            = {summary['overall']:.3f}")
        print(f"    评测耗时           = {eval_latency:.0f}ms")
        print(f"{'─'*70}\n")

    return report


# ─── 报告生成 ───────────────────────────────────────────────────────────────────

_REPORT_DIR = Path("reports")
_REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _generate_ragas_markdown(report: dict) -> Path:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    mode = report.get("mode", "unknown")
    summary = report.get("summary", {})
    details = report.get("details", [])

    lines = [
        "# RAGAS Evaluation Report\n",
        f"**生成时间**: {ts}  ",
        f"**评测模式**: {mode} ({'原生 ragas' if mode == 'native' else 'LLM-as-Judge fallback'})\n",
        f"**样本数**: {report.get('total_samples', 0)}  ",
        f"**评测耗时**: {report.get('eval_latency_ms', 0):.0f}ms\n",
        "---\n",
        "## Summary\n",
        "| 指标 | 得分 |",
        "|------|------|",
        f"| Faithfulness | {summary.get('faithfulness', 0):.3f} |",
        f"| Answer Relevancy | {summary.get('answer_relevancy', 0):.3f} |",
        f"| Context Recall | {summary.get('context_recall', 0):.3f} |",
        f"| Context Precision | {summary.get('context_precision', 0):.3f} |",
        f"| **Overall** | **{summary.get('overall', 0):.3f}** |",
        "\n---\n",
        "## Details\n",
        "| # | Query | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |",
        "|---|-------|--------------|------------------|----------------|-------------------|---------|",
    ]
    for i, d in enumerate(details, 1):
        q = d["query"][:40] + "..." if len(d["query"]) > 40 else d["query"]
        s = d["scores"]
        lines.append(
            f"| {i} | {q} | {s['faithfulness']:.2f} | {s['answer_relevancy']:.2f} | "
            f"{s['context_recall']:.2f} | {s['context_precision']:.2f} | {s['overall']:.2f} |"
        )

    lines.append("\n---\n")
    lines.append("*Report generated by rag_framework.eval.ragas_eval*\n")

    md_path = _REPORT_DIR / "ragas_score.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return md_path
