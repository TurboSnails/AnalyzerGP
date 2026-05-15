#!/usr/bin/env python3
"""
SQuAD → 评测集构建器

生成两份文件供 rag_framework 评测框架使用：

  benchmark.json    — 检索评测（ranking.py）
                      chunk_ids 为精确 passage ID，hit_judge 精确匹配
  qa_benchmark.json — 端到端 QA 评测（qa.py）
                      包含 expected_answer 字段

用法：
  cd fenxiCB
  uv run python domains/msmarco/scripts/build_benchmark.py
  uv run python domains/msmarco/scripts/build_benchmark.py --split train --max-questions 500
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "uv.lock").exists():
            return parent
    raise RuntimeError("找不到项目根目录")


_ROOT = _find_project_root()
_OUTPUT_DIR = _ROOT / "domains/msmarco/msmarco_domain/eval"


def _passage_id(title: str, context: str) -> str:
    h = hashlib.md5(context.encode()).hexdigest()[:10]
    safe_title = title.replace(" ", "_")[:40]
    return f"{safe_title}_{h}"


def _load_squad(split: str):
    try:
        from datasets import load_dataset
    except ImportError:
        print("请先安装: uv add datasets", file=sys.stderr)
        sys.exit(1)
    print(f"加载 rajpurkar/squad [{split}] ...")
    return load_dataset("rajpurkar/squad", split=split)


def _build_ranking_benchmark(ds, max_questions: int | None) -> list[dict]:
    """
    每道题对应一条评测记录：
      query     = question
      chunk_ids = [passage_id of its context]
    """
    items = []
    seen_questions: set[str] = set()

    for row in ds:
        q = row["question"].strip()
        if q in seen_questions:
            continue
        seen_questions.add(q)

        pid = _passage_id(row["title"], row["context"].strip())
        items.append({
            "query": q,
            "chunk_ids": [pid],
            "difficulty": "standard",
        })
        if max_questions and len(items) >= max_questions:
            break

    print(f"  ranking benchmark: {len(items)} 条")
    return items


def _build_qa_benchmark(ds, max_questions: int | None) -> list[dict]:
    """
    每道题对应一条 QA 记录：
      query           = question
      expected_answer = answers.text[0]
      chunk_ids       = [passage_id]（辅助字段）
    """
    items = []
    seen_questions: set[str] = set()

    for row in ds:
        q       = row["question"].strip()
        answers = row["answers"]["text"]
        if q in seen_questions or not answers:
            continue
        seen_questions.add(q)

        pid = _passage_id(row["title"], row["context"].strip())
        items.append({
            "query": q,
            "expected_answer": answers[0].strip(),
            "chunk_ids": [pid],
        })
        if max_questions and len(items) >= max_questions:
            break

    print(f"  qa benchmark:      {len(items)} 条")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 SQuAD 评测集（msmarco domain）")
    parser.add_argument("--split", default="validation",
                        choices=["validation", "train"],
                        help="数据集分割（需与 download_and_index.py 一致）")
    parser.add_argument("--max-questions", type=int, default=None, metavar="N",
                        help="最多生成 N 条评测题（默认: 全量）")
    parser.add_argument("--output-dir", type=Path, default=_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = _load_squad(args.split)

    print("\n生成评测文件...")
    ranking_items = _build_ranking_benchmark(ds, args.max_questions)
    qa_items      = _build_qa_benchmark(ds, args.max_questions)

    ranking_path = args.output_dir / "benchmark.json"
    qa_path      = args.output_dir / "qa_benchmark.json"

    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(ranking_items, f, ensure_ascii=False, indent=2)
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_items, f, ensure_ascii=False, indent=2)

    print(f"\n完成：")
    print(f"  {ranking_path}")
    print(f"  {qa_path}")
    print()
    print("运行检索评测：")
    print("  DOMAIN=msmarco RAG_BM25_INDEX_DIR=ai_app1/data/msmarco_bm25 \\")
    print("      uv run python -c \"")
    print("      import asyncio; from rag_framework.eval.ranking import run_ranking_eval")
    print("      asyncio.run(run_ranking_eval())\"")


if __name__ == "__main__":
    main()
