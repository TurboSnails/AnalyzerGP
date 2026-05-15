#!/usr/bin/env python3
"""
生成 benchmark_precise.json — 自动检索 + 人工复核真实 Chunk ID

用法：
  cd fenxiCB
  uv run python -m domains.android.scripts.generate_benchmark_precise \
      [--benchmark domains/android/android_domain/eval/benchmark.json] \
      [--output domains/android/android_domain/eval/benchmark_precise.json] \
      [--top-k 5]

工作流程：
  1. 读取 benchmark.json（标签模糊匹配版）
  2. 对每个 query 调用 HybridRetriever 执行检索
  3. 使用 hit_judge 自动判断 top-k 结果中哪条 chunk 命中 ground truth
  4. 将命中的 chunk_id 写入 benchmark_precise.json
  5. 输出 "待复核" 列表（自动匹配失败的条目）

注意：
  - 运行前必须确保索引已构建（init_vector_db_v2.py）
  - 生成后建议人工复核 "待复核" 条目中的 chunk_id
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ── 自动探测并切换工作目录（兼容 VSCode「Run」按钮）─────────────────────────────
# VSCode 的「Run Python File」默认以脚本所在目录为 cwd，会导致 uv 无法解析
# 根 pyproject.toml 中定义的 editable 包（rag-framework / android-domain）。
# 下面的代码在导入任何外部包之前，先沿 __file__ 向上找到项目根目录并切换过去。

def _find_project_root() -> Path:
    script_path = Path(__file__).resolve()
    root = Path.cwd()
    for parent in script_path.parents:
        if (parent / "pyproject.toml").exists() and (parent / "uv.lock").exists():
            root = parent
    return root

_PROJECT_ROOT = _find_project_root()
if _PROJECT_ROOT != Path.cwd():
    os.chdir(_PROJECT_ROOT)
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    if str(_PROJECT_ROOT / "rag_framework") not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT / "rag_framework"))
    if str(_PROJECT_ROOT / "domains/android") not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT / "domains/android"))

from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings
from rag_framework.eval.hit_judge import is_hit


async def generate_precise_benchmark(
    benchmark_path: Path,
    output_path: Path,
    top_k: int = 5,
) -> dict:
    """
    自动生成精确 Chunk ID 的评测集。

    Returns:
        {"precise": [...], "pending_review": [...], "stats": {...}}
    """
    with open(benchmark_path, encoding="utf-8") as f:
        dataset = json.load(f)

    container = RAGContainer.from_settings(get_settings())

    precise_items: list[dict] = []
    pending_items: list[dict] = []

    total = len(dataset)
    auto_matched = 0
    manual_needed = 0

    print(f"\n{'='*70}")
    print(f"  Generating Precise Benchmark   共 {total} 条  top_k={top_k}")
    print(f"{'='*70}\n")

    for i, item in enumerate(dataset, 1):
        query = item.get("query", "")
        if not query:
            continue

        # 执行检索
        route = container.domain.classify_query(query, [])
        result = await container.retriever.retrieve([route], top_k=top_k)

        expected = item.get("expected_chunk", "")
        raw_ev = item.get("evidence", "")
        evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

        # 自动匹配
        matched_ids: list[str] = []
        for doc in result.docs:
            hit, reason = is_hit(doc.text, expected, evidence)
            if hit:
                matched_ids.append(doc.id)

        if matched_ids:
            auto_matched += 1
            status = "✅ 自动匹配"
        else:
            manual_needed += 1
            status = "⚠️  待复核"
            # 取 top1 作为候选，人工需要确认
            matched_ids = [result.docs[0].id] if result.docs else [""]

        precise_item = {
            "query": query,
            "chunk_ids": matched_ids,
            "expected_chunk": expected,
            "evidence": item.get("evidence"),
            "difficulty": item.get("difficulty", "standard"),
            "_auto_matched": bool(matched_ids and matched_ids[0]),
            "_top_chunks": [
                {"id": d.id, "text": d.text[:80].replace(chr(10), " ")}
                for d in result.docs[:3]
            ],
        }

        if matched_ids and matched_ids[0]:
            precise_items.append(precise_item)
        else:
            pending_items.append(precise_item)

        print(f"[{i:03d}/{total}] {status}  {query[:50]}...")
        if result.docs:
            print(f"           top1: {result.docs[0].id[:40]:40s}  {result.docs[0].text[:50]!r}")
        print()

    # 合并输出（precise 在前，pending 在后）
    output_items = precise_items + pending_items

    # 统计
    stats = {
        "total": total,
        "auto_matched": auto_matched,
        "manual_needed": manual_needed,
        "auto_match_rate": round(auto_matched / total, 4) if total else 0.0,
    }

    report = {
        "_comment": "此文件由 generate_benchmark_precise.py 自动生成，pending_review 部分需人工复核",
        "stats": stats,
        "data": output_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"{'='*70}")
    print(f"  生成完成")
    print(f"    总计: {total}")
    print(f"    自动匹配: {auto_matched} ({stats['auto_match_rate']:.1%})")
    print(f"    待复核: {manual_needed}")
    print(f"  输出文件: {output_path}")
    print(f"{'='*70}\n")

    # 打印待复核列表（供人工快速浏览）
    if pending_items:
        print("\n⚠️  待复核条目（请确认 chunk_ids 是否正确）：")
        for item in pending_items:
            print(f"  Query: {item['query'][:60]}...")
            for tc in item["_top_chunks"]:
                print(f"    - {tc['id'][:40]:40s} | {tc['text'][:60]!r}")
            print()

    return report


def main():
    parser = argparse.ArgumentParser(
        description="自动生成 benchmark_precise.json（精确 Chunk ID 评测集）",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("domains/android/android_domain/eval/benchmark.json"),
        help="输入的 benchmark.json 路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("domains/android/android_domain/eval/benchmark_precise.json"),
        help="输出的 benchmark_precise.json 路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        dest="top_k",
        help="检索时取 top_k 个结果用于匹配",
    )
    args = parser.parse_args()
    asyncio.run(generate_precise_benchmark(args.benchmark, args.output, args.top_k))


if __name__ == "__main__":
    main()
