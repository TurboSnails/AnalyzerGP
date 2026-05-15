#!/usr/bin/env python3
"""
Failure Triage — BAD Case 自动/半自动分类与根因分析

工作流程：
  1. 读取 failure_cases.jsonl（由 FailureCollector 生成）
  2. 对未分类 root_cause 的条目自动推断
  3. 对置信度低的条目进入交互式半自动分类（可选）
  4. 生成结构化分析报告（Markdown + JSON）

用法：
  # 全自动分类（非交互）
  uv run python -m rag_framework.eval.failure_triage --auto

  # 半自动模式（对空 root_cause 逐项询问）
  uv run python -m rag_framework.eval.failure_triage --interactive

  # 仅生成报告（不修改数据）
  uv run python -m rag_framework.eval.failure_triage --report-only

  # 指定输入/输出路径
  uv run python -m rag_framework.eval.failure_triage \
      --input reports/failure_cases.jsonl \
      --output reports/bad_case_analysis.md
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rag_framework.eval.failure_analysis import FailureCase, FailureCollector, FailureStore


# ─── 根因标签说明（用于交互提示）────────────────────────────────────────────────

ROOT_CAUSE_LABELS: dict[str, str] = {
    "kb_gap": "知识库缺失 — 用户问了合理的技术问题，但知识库中无对应内容",
    "semantic_mismatch": "语义理解偏差 — query 与 chunk 的语义空间未对齐（如措辞差异、同义词）",
    "hallucination": "模型幻觉 — LLM 输出了上下文未提及的虚构信息",
    "bad_rewrite": "Query 扩写失真 — rewrite 后 query 偏离原意或引入噪声",
    "rerank_misorder": "精排错位 — Reranker 将正确 chunk 挤出 top1",
    "context_loss": "长上下文信息丢失 — 上下文过长导致关键信息被淹没",
    "user_noise": "用户输入噪声/歧义 — query 过短、模糊或完全无技术语义",
    "skip": "跳过 — 无法判断，留给后续人工处理",
}

# ─── 自动分类引擎 ───────────────────────────────────────────────────────────────

class TriageEngine:
    """
    BAD Case 分类引擎。

    支持全自动和半自动两种模式。
    """

    def __init__(self, store: FailureStore) -> None:
        self.store = store
        self.collector = FailureCollector(store=store)
        self._modified = False

    def load_cases(self) -> list[FailureCase]:
        """加载所有失败样本。"""
        return self.store.load_all()

    def auto_classify(self, cases: list[FailureCase] | None = None) -> list[FailureCase]:
        """
        对未分类的 case 自动推断 root_cause。

        Returns:
            被修改过的 case 列表
        """
        if cases is None:
            cases = self.load_cases()

        modified: list[FailureCase] = []
        for case in cases:
            if case.root_cause:
                continue
            inferred = self.collector.infer_root_cause(
                category=case.category,
                query=case.query,
                reason=case.reason,
                trace=case.trace,
            )
            if inferred:
                case.root_cause = inferred
                modified.append(case)
                self._modified = True
        return modified

    def interactive_classify(self, cases: list[FailureCase] | None = None) -> list[FailureCase]:
        """
        交互式分类：对未分类的 case 逐项询问用户。

        Returns:
            被修改过的 case 列表
        """
        if cases is None:
            cases = self.load_cases()

        modified: list[FailureCase] = []
        pending = [c for c in cases if not c.root_cause]
        if not pending:
            print("✅ 所有 case 均已分类，无需交互。")
            return modified

        print(f"\n{'='*60}")
        print(f"  半自动分类模式   待处理: {len(pending)} 条")
        print(f"{'='*60}")
        print("\n快捷按键说明:")
        for key, desc in ROOT_CAUSE_LABELS.items():
            print(f"  [{key}] {desc}")
        print("\n也可以直接输入原因编号或自定义字符串。\n")

        for i, case in enumerate(pending, 1):
            print(f"\n─" * 60)
            print(f"[{i}/{len(pending)}] 分类: {case.category}")
            print(f"Query : {case.query}")
            print(f"Reason: {case.reason}")
            if case.metadata:
                print(f"Meta  : {case.metadata}")

            choice = input("root_cause? > ").strip().lower()
            if choice in ("skip", "s", ""):
                print("  → 跳过")
                continue
            if choice in ROOT_CAUSE_LABELS:
                case.root_cause = choice
            else:
                # 允许自定义输入
                case.root_cause = choice
            modified.append(case)
            self._modified = True
            print(f"  → 标记为: {case.root_cause}")

        return modified

    def save(self, cases: list[FailureCase]) -> None:
        """覆盖写回 JSON Lines（仅在修改后调用）。"""
        if not self._modified:
            return
        self.store.path.write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in cases) + "\n",
            encoding="utf-8",
        )
        print(f"\n💾 已保存 {len(cases)} 条到 {self.store.path}")


# ─── 报告生成 ───────────────────────────────────────────────────────────────────

def generate_report(cases: list[FailureCase], output_path: Path) -> str:
    """
    生成 BAD Case 结构化分析报告。

    输出 Markdown 格式，包含：
      - 总体统计
      - 技术分类分布
      - 根因分布
      - Top 问题 query 列表
      - 针对性优化建议
    """
    total = len(cases)
    if total == 0:
        report = "# BAD Case 分析报告\n\n暂无失败样本。\n"
        output_path.write_text(report, encoding="utf-8")
        return report

    # 技术分类统计
    cat_counts: dict[str, int] = {}
    for c in cases:
        cat_counts[c.category] = cat_counts.get(c.category, 0) + 1

    # 根因统计
    rc_counts: dict[str, int] = {}
    for c in cases:
        if c.root_cause:
            rc_counts[c.root_cause] = rc_counts.get(c.root_cause, 0) + 1

    # 按根因分组
    by_rc: dict[str, list[FailureCase]] = {}
    for c in cases:
        rc = c.root_cause or "uncategorized"
        by_rc.setdefault(rc, []).append(c)

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# BAD Case 分析报告\n",
        f"**生成时间**: {ts}  ",
        f"**样本总数**: {total}\n",
        "---\n",
        "## 一、总体统计\n",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总失败样本 | {total} |",
        f"| 已分类根因 | {sum(rc_counts.values())} ({sum(rc_counts.values())/total:.1%}) |",
        f"| 未分类根因 | {total - sum(rc_counts.values())} |",
        "\n---\n",
        "## 二、技术分类分布\n",
        "| 分类 | 数量 | 占比 |",
        "|------|------|------|",
    ]
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {cnt} | {cnt/total:.1%} |")

    lines.extend([
        "\n---\n",
        "## 三、根因分布（人因分类）\n",
        "| 根因 | 数量 | 占比 | 说明 |",
        "|------|------|------|------|",
    ])
    for rc, cnt in sorted(rc_counts.items(), key=lambda x: -x[1]):
        desc = ROOT_CAUSE_LABELS.get(rc, "自定义")
        lines.append(f"| {rc} | {cnt} | {cnt/total:.1%} | {desc} |")

    # 未分类
    uncategorized = total - sum(rc_counts.values())
    if uncategorized:
        lines.append(f"| uncategorized | {uncategorized} | {uncategorized/total:.1%} | 尚未人工/自动分类 |")

    lines.extend([
        "\n---\n",
        "## 四、各根因典型样本\n",
    ])
    for rc, case_list in sorted(by_rc.items(), key=lambda x: -len(x[1])):
        lines.append(f"\n### {rc}（{len(case_list)} 条）\n")
        for c in case_list[:5]:
            lines.append(f"- **Query**: {c.query}")
            lines.append(f"  - 分类: {c.category} | 原因: {c.reason}")
        if len(case_list) > 5:
            lines.append(f"  - ... 等共 {len(case_list)} 条")

    lines.extend([
        "\n---\n",
        "## 五、针对性优化建议\n",
    ])

    # 根据根因分布自动生成优化建议
    suggestions: list[str] = []
    if rc_counts.get("kb_gap", 0) / total > 0.2:
        suggestions.append(
            "1. **知识库扩展（kb_gap 占比高）**: "
            "当前知识库覆盖不足，建议补充 Android 高级主题文档（如 Jetpack Compose 深层原理、"
            "性能优化进阶、安全加固等），并运行 `init_vector_db_v2.py` 重新索引。"
        )
    if rc_counts.get("semantic_mismatch", 0) / total > 0.2:
        suggestions.append(
            "2. **语义对齐优化（semantic_mismatch 占比高）**: "
            "query 与 chunk 的语义空间存在偏差。建议："
            "(a) 扩充中文术语映射表 `zh_to_en.json`；"
            "(b) 优化 HyDE prompt，生成更贴近用户表述风格的问题；"
            "(c) 尝试更大规模的 embedding 模型（如 bge-m3 → bge-large-zh-v1.5）。"
        )
    if rc_counts.get("hallucination", 0) / total > 0.1:
        suggestions.append(
            "3. **幻觉抑制（hallucination 占比高）**: "
            "LLM 在缺乏明确上下文时倾向于虚构。建议："
            "(a) 收紧 fallback 阈值，低置信度时拒绝回答而非猜测；"
            "(b) 在 system prompt 中强化『无法确认时请直接说明』的约束；"
            "(c) 引入 Self-RAG 生成时的引用标记（citation），强制 grounding。"
        )
    if rc_counts.get("bad_rewrite", 0) / total > 0.1:
        suggestions.append(
            "4. **Rewrite 策略调优（bad_rewrite 占比高）**: "
            "Query 扩写引入噪声。建议："
            "(a) 降低 LLM rewrite 的 temperature；"
            "(b) 增加 rule-based rewrite 的覆盖率，减少 LLM 介入频率；"
            "(c) 在 rewrite_eval 中加入『语义相似度』 guarding，拒绝偏离过大的 rewrite。"
        )
    if rc_counts.get("rerank_misorder", 0) / total > 0.1:
        suggestions.append(
            "5. **Reranker 优化（rerank_misorder 占比高）**: "
            "CrossEncoder 精排把正确 chunk 挤出 top1。建议："
            "(a) 检查 reranker 训练/微调数据是否与当前领域分布一致；"
            "(b) 调整 `low_confidence_threshold`，在 rerank 置信度低时回退到 RRF 排序；"
            "(c) 增加负样本多样性，微调 reranker。"
        )
    if rc_counts.get("context_loss", 0) / total > 0.1:
        suggestions.append(
            "6. **上下文压缩（context_loss 占比高）**: "
            "召回 chunk 过多导致关键信息被淹没。建议："
            "(a) 调小 `rerank_top_k`（如 3 → 2）；"
            "(b) 启用 ai_app3 的 `context_compressor` 做自适应摘要；"
            "(c) 在 Lost-in-Middle 重排基础上增加『关键句提取』前置步骤。"
        )
    if rc_counts.get("user_noise", 0) / total > 0.15:
        suggestions.append(
            "7. **输入降噪（user_noise 占比高）**: "
            "用户 query 过短或过于模糊。建议："
            "(a) 在 chat API 层增加 query 质量校验，拒绝空/极短输入；"
            "(b) 对模糊 query 主动追问（clarification），而非直接检索；"
            "(c) 在对话历史中做指代消解（anaphora resolution），补全省略主语。"
        )

    if not suggestions:
        suggestions.append(
            "当前各根因分布较为均衡，暂无显著短板。建议持续运行 `failure_triage` 监控趋势变化。"
        )

    for s in suggestions:
        lines.append(s + "\n")

    lines.append("\n---\n")
    lines.append("*Report generated by rag_framework.eval.failure_triage*\n")

    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BAD Case 自动/半自动分类与根因分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  uv run python -m rag_framework.eval.failure_triage --auto
  uv run python -m rag_framework.eval.failure_triage --interactive
  uv run python -m rag_framework.eval.failure_triage --report-only
        """,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("reports/failure_cases.jsonl"),
        help="输入的 failure_cases.jsonl 路径",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("reports/bad_case_analysis.md"),
        help="输出的分析报告路径",
    )
    parser.add_argument(
        "--auto", "-a",
        action="store_true",
        help="全自动分类（非交互，仅自动推断）",
    )
    parser.add_argument(
        "--interactive", "-I",
        action="store_true",
        help="半自动模式（对未分类条目逐项交互询问）",
    )
    parser.add_argument(
        "--report-only", "-r",
        action="store_true",
        help="仅生成报告，不修改 failure_cases.jsonl",
    )
    args = parser.parse_args()

    store = FailureStore(path=args.input)
    engine = TriageEngine(store)
    cases = engine.load_cases()

    if not cases:
        print(f"⚠️  未找到失败样本: {args.input}")
        print("请先运行 comprehensive_eval 或其他 pipeline 积累 failure cases。")
        report = generate_report(cases, args.output)
        print(f"\n📄 已生成空报告: {args.output}")
        print(report[:200] + "\n...")
        return

    print(f"\n📂 已加载 {len(cases)} 条失败样本")

    if args.report_only:
        report = generate_report(cases, args.output)
        print(f"\n📄 报告已保存: {args.output}")
        print(report[:800] + "\n...")
        return

    # 自动分类
    auto_modified = engine.auto_classify(cases)
    print(f"🤖 自动分类完成: {len(auto_modified)} 条被更新")

    # 半自动交互
    if args.interactive:
        interactive_modified = engine.interactive_classify(cases)
        print(f"👤 交互分类完成: {len(interactive_modified)} 条被更新")

    # 保存
    if not args.report_only:
        engine.save(cases)

    # 生成报告
    report = generate_report(cases, args.output)
    print(f"\n📄 分析报告已保存: {args.output}")
    print(report[:1200] + "\n...")


if __name__ == "__main__":
    main()
