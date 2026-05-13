"""
通用端到端 QA 评测器（QA Evaluator）

评测目标：不是"有没有召回正确 chunk"，而是"LLM 最终回答是否正确"。

采用 LLM-as-Judge 模式，维度：
  - coverage（关键点覆盖）
  - hallucination（幻觉程度）
  - relevance（回答相关性）
  - conciseness（简洁性）
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TypedDict

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.core.logger import eval_logger
from rag_framework.llm.base import LLMClient


# ─── 评测集加载 ─────────────────────────────────────────────────────────────────

def load_qa_dataset(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ─── LLM-as-Judge Prompt ──────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = """\
你是 RAG 系统评测专家。你的任务是根据【参考答案】评估【模型回答】的质量。

请从以下 4 个维度打分（0~1，保留两位小数），并给出理由：

1. coverage（关键点覆盖）
   - 模型回答是否覆盖了参考答案中的核心要点？
   - 1.0 = 完全覆盖；0.5 = 部分覆盖；0.0 = 未覆盖

2. hallucination（幻觉程度）
   - 模型回答是否包含与参考答案矛盾或无法验证的信息？
   - 1.0 = 无任何幻觉；0.5 = 少量过度推断；0.0 = 严重幻觉

3. relevance（回答相关性）
   - 模型回答是否直接回应了用户问题，而非泛泛而谈？
   - 1.0 = 精准回答；0.0 = 完全跑题

4. conciseness（简洁性）
   - 模型回答是否简洁、无冗余？
   - 1.0 = 恰到好处；0.5 = 略长但可接受；0.0 = 过度冗长

输出格式（严格 JSON，不要 markdown 代码块）：
{
  "coverage": 0.85,
  "hallucination": 0.90,
  "relevance": 0.95,
  "conciseness": 0.80,
  "overall": 0.88,
  "reason": "覆盖主要要点，但遗漏了 WeakReference 细节；无幻觉；回答相关且结构清晰；略显冗长。"
}

注意：
- overall = (coverage + hallucination + relevance + conciseness) / 4
- 理由用中文，简洁客观，不超过 80 字。
"""

_JUDGE_USER_TEMPLATE = """\
【用户问题】
{query}

【参考答案】
{expected_answer}

【模型回答】
{model_answer}

请输出 JSON 评分。
"""


# ─── Judge 客户端 ─────────────────────────────────────────────────────────────

class QAJudge:
    """封装 LLM-as-Judge 的评分逻辑。"""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    async def judge(
        self,
        query: str,
        expected_answer: str,
        model_answer: str,
    ) -> dict:
        prompt = _JUDGE_USER_TEMPLATE.format(
            query=query,
            expected_answer=expected_answer,
            model_answer=model_answer,
        )
        messages = [
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = await self._llm.chat(messages, use_tools=False)
            # 提取 JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start:end + 1])
            else:
                parsed = json.loads(raw)

            scores = {
                "coverage": float(parsed.get("coverage", 0.0)),
                "hallucination": float(parsed.get("hallucination", 0.0)),
                "relevance": float(parsed.get("relevance", 0.0)),
                "conciseness": float(parsed.get("conciseness", 0.0)),
                "overall": float(parsed.get("overall", 0.0)),
                "reason": str(parsed.get("reason", "")),
            }
            computed = (scores["coverage"] + scores["hallucination"] +
                        scores["relevance"] + scores["conciseness"]) / 4.0
            if abs(scores["overall"] - computed) > 0.15:
                scores["overall"] = round(computed, 2)
            return scores
        except Exception as e:
            eval_logger.warning(f"Judge 解析失败: {e}")
            return {
                "coverage": 0.0,
                "hallucination": 0.0,
                "relevance": 0.0,
                "conciseness": 0.0,
                "overall": 0.0,
                "reason": f"Judge 解析失败: {e}",
            }


# ─── RAG 回答生成 ─────────────────────────────────────────────────────────────

async def generate_answer(query: str, container: RAGContainer) -> str:
    """调用完整 RAG pipeline 生成非流式回答。"""
    session = container.session_store.get("eval_user")
    # 清空历史，确保每条 query 独立
    session.history = []
    session.history.append({"role": "user", "content": query})

    from rag_framework.session.manager import SessionManager
    manager = SessionManager(
        store=container.session_store,
        llm=container.llm,
        retriever=container.retriever,
        domain=container.domain,
        settings=container.settings,
    )
    messages = manager.build_messages(session, query)
    answer = await container.llm.chat(messages, use_tools=False)
    return answer or ""


# ─── 单条 QA 评测 ─────────────────────────────────────────────────────────────

async def evaluate_single_qa(
    item: dict,
    judge: QAJudge,
    container: RAGContainer,
) -> dict:
    query = item["query"]
    expected = item["expected_answer"]

    t0 = time.perf_counter()
    model_answer = await generate_answer(query, container)
    gen_latency = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    scores = await judge.judge(query, expected, model_answer)
    judge_latency = (time.perf_counter() - t1) * 1000

    return {
        "query": query,
        "expected_answer": expected,
        "model_answer": model_answer,
        "scores": scores,
        "latency_ms": {
            "generation": round(gen_latency, 2),
            "judging": round(judge_latency, 2),
            "total": round(gen_latency + judge_latency, 2),
        },
    }


# ─── 批量 QA 评测 ─────────────────────────────────────────────────────────────

async def run_qa_eval(
    dataset_path: Path | None = None,
    container: RAGContainer | None = None,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    对 QA 评测集批量执行端到端评测。

    Args:
        dataset_path: QA 评测集路径
        container: RAG 容器，默认自动构建
        max_samples: 最大评测条数
        verbose: 是否打印逐条结果

    Returns:
        综合报告字典
    """
    if container is None:
        container = RAGContainer.from_settings(get_settings())

    if dataset_path is None:
        # 尝试从 domain 获取
        dataset = container.domain.get_eval_dataset()
        if not dataset:
            eval_logger.warning("Domain 未提供评测集，返回空报告")
            return {}
    else:
        dataset = load_qa_dataset(dataset_path)

    if max_samples:
        dataset = dataset[:max_samples]
    total = len(dataset)

    judge = QAJudge(llm=container.llm)

    if verbose:
        name = dataset_path.name if dataset_path else "domain_eval"
        print(f"\n{'─'*70}")
        print(f"  End-to-End QA Evaluation   评测集: {name}  共 {total} 条")
        print(f"{'─'*70}\n")

    results: list[dict] = []
    for i, item in enumerate(dataset, 1):
        res = await evaluate_single_qa(item, judge, container)
        results.append(res)

        if verbose:
            sc = res["scores"]
            print(f"[{i:02d}/{total}] overall={sc['overall']:.2f}  "
                  f"(cov={sc['coverage']:.2f} hal={sc['hallucination']:.2f} "
                  f"rel={sc['relevance']:.2f} con={sc['conciseness']:.2f})")
            print(f"  Query: {res['query'][:60]}")
            print(f"  Judge: {sc['reason'][:70]}")
            print()

    overalls = [r["scores"]["overall"] for r in results]
    coverages = [r["scores"]["coverage"] for r in results]
    hallucinations = [r["scores"]["hallucination"] for r in results]
    latencies = [r["latency_ms"]["total"] for r in results]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    report = {
        "dataset": str(dataset_path) if dataset_path else "domain",
        "total_samples": total,
        "avg_overall": round(_mean(overalls), 3),
        "avg_coverage": round(_mean(coverages), 3),
        "avg_hallucination": round(_mean(hallucinations), 3),
        "avg_latency_ms": round(_mean(latencies), 2),
        "details": results,
    }

    if verbose:
        print(f"{'─'*70}")
        print(f"  QA 综合报告")
        print(f"    平均 Overall      = {report['avg_overall']:.3f}")
        print(f"    平均 Coverage     = {report['avg_coverage']:.3f}")
        print(f"    平均 Hallucination= {report['avg_hallucination']:.3f}")
        print(f"    平均延迟           = {report['avg_latency_ms']:.0f}ms")
        print(f"{'─'*70}\n")

    return report
