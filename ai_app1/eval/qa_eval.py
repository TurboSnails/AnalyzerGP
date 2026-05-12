"""
End-to-End QA Evaluation — 端到端问答评测
==========================================

评测目标：
  不是"有没有召回正确 chunk"，而是"LLM 最终回答是否正确"。

评测维度：
  1. 关键点覆盖（Key Point Coverage）
     用 LLM-as-Judge 判断回答是否覆盖了 expected_answer 中的关键要点。
  2. 幻觉检测（Hallucination Detection）
     判断回答是否包含评测集中不存在的信息。
  3. 答案相关性（Answer Relevance）
     回答是否与问题相关，而不是答非所问。
  4. 简洁性（Conciseness）
     是否冗长或过度展开。

评测方法：
  由于无人工标注，采用 "LLM-as-Judge" 模式：
    - 使用 MiniMax-M2.7（与线上模型一致）作为裁判
    - 设计结构化 prompt，要求输出 JSON 评分
    - 每个维度 0~1 分，最终综合得分加权平均

运行方式：
    uv run python -m ai_app1.eval.qa_eval
"""
from __future__ import annotations

import json
import sys
import time
import asyncio
from pathlib import Path
from typing import TypedDict

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ai_app1.service.AiClient import AiClient
from ai_app1.service.session import build_messages, get_session, add_user_message
from ai_app1.core.config import OPENAI_API_KEY


# ─── 评测集加载 ───────────────────────────────────────────────────────────────

_QA_BENCHMARK = Path(__file__).parent / "qa_benchmark.json"


def _load_qa_dataset(path: Path = _QA_BENCHMARK) -> list[dict]:
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

    def __init__(self, ai_client: AiClient | None = None) -> None:
        self._client = ai_client or AiClient(ai_api_key=OPENAI_API_KEY)

    async def judge(
        self,
        query: str,
        expected_answer: str,
        model_answer: str,
    ) -> dict:
        """
        对单条 QA 进行评分。

        Returns:
            {
                "coverage": float,
                "hallucination": float,
                "relevance": float,
                "conciseness": float,
                "overall": float,
                "reason": str,
            }
        """
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
            raw = await self._client.chat(messages, use_tools=False)
            # 提取 JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start:end+1])
            else:
                parsed = json.loads(raw)

            # 标准化字段
            scores = {
                "coverage": float(parsed.get("coverage", 0.0)),
                "hallucination": float(parsed.get("hallucination", 0.0)),
                "relevance": float(parsed.get("relevance", 0.0)),
                "conciseness": float(parsed.get("conciseness", 0.0)),
                "overall": float(parsed.get("overall", 0.0)),
                "reason": str(parsed.get("reason", "")),
            }
            # 自动计算 overall（若缺失或明显错误）
            computed = (scores["coverage"] + scores["hallucination"] +
                        scores["relevance"] + scores["conciseness"]) / 4.0
            if abs(scores["overall"] - computed) > 0.15:
                scores["overall"] = round(computed, 2)
            return scores
        except Exception as e:
            return {
                "coverage": 0.0,
                "hallucination": 0.0,
                "relevance": 0.0,
                "conciseness": 0.0,
                "overall": 0.0,
                "reason": f"Judge 解析失败: {e}",
            }


# ─── RAG 回答生成 ─────────────────────────────────────────────────────────────

async def _generate_answer(query: str, ai_client: AiClient) -> str:
    """
    调用完整 RAG pipeline 生成回答。

    复用 session.build_messages 逻辑，构造带检索上下文的 messages，
    然后调用 AiClient.chat 获取非流式回答。
    """
    session = get_session("eval_user")
    # 清空历史，确保每条 query 独立
    session["history"] = []
    add_user_message(session, query)

    messages = build_messages(session, query)
    answer = await ai_client.chat(messages, use_tools=False)
    return answer or ""


# ─── 单条 QA 评测 ─────────────────────────────────────────────────────────────

async def evaluate_single_qa(
    item: dict,
    judge: QAJudge,
    ai_client: AiClient,
) -> dict:
    """
    评测单条 QA：生成回答 → Judge 评分。

    Returns:
        {
            "query": str,
            "expected_answer": str,
            "model_answer": str,
            "scores": dict,
            "latency_ms": float,
        }
    """
    query = item["query"]
    expected = item["expected_answer"]

    t0 = time.perf_counter()
    model_answer = await _generate_answer(query, ai_client)
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
    dataset_path: Path = _QA_BENCHMARK,
    max_samples: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    对 QA 评测集批量执行端到端评测。

    Args:
        dataset_path : QA 评测集路径
        max_samples  : 最大评测条数（用于快速验证，默认全部）
        verbose      : 是否打印逐条结果

    Returns:
        综合报告字典
    """
    dataset = _load_qa_dataset(dataset_path)
    if max_samples:
        dataset = dataset[:max_samples]
    total = len(dataset)

    ai_client = AiClient(ai_api_key=OPENAI_API_KEY)
    judge = QAJudge(ai_client=ai_client)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  End-to-End QA Evaluation   评测集: {dataset_path.name}  共 {total} 条")
        print(f"{'─'*70}\n")

    results: list[dict] = []
    for i, item in enumerate(dataset, 1):
        res = await evaluate_single_qa(item, judge, ai_client)
        results.append(res)

        if verbose:
            sc = res["scores"]
            print(f"[{i:02d}/{total}] overall={sc['overall']:.2f}  "
                  f"(cov={sc['coverage']:.2f} hal={sc['hallucination']:.2f} "
                  f"rel={sc['relevance']:.2f} con={sc['conciseness']:.2f})")
            print(f"  Query: {res['query'][:60]}")
            print(f"  Judge: {sc['reason'][:70]}")
            print()

    # 聚合指标
    overalls = [r["scores"]["overall"] for r in results]
    coverages = [r["scores"]["coverage"] for r in results]
    hallucinations = [r["scores"]["hallucination"] for r in results]
    latencies = [r["latency_ms"]["total"] for r in results]

    avg = lambda vals: sum(vals) / len(vals) if vals else 0.0

    report = {
        "dataset": str(dataset_path),
        "total_samples": total,
        "avg_overall": round(avg(overalls), 3),
        "avg_coverage": round(avg(coverages), 3),
        "avg_hallucination": round(avg(hallucinations), 3),
        "avg_latency_ms": round(avg(latencies), 2),
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


# ─── 主入口 ───────────────────────────────────────────────────────────────────

async def main():
    report = await run_qa_eval()
    # 保存报告
    out_dir = Path(__file__).parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qa_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"QA 评测报告已保存: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
