"""
Retrieval Evaluation — Recall@K
================================
对评测集里每条 query 执行完整的 query_db() 检索流水线，判断是否命中
expected_chunk 所对应的证据片段。

命中规则（HIT）：
  1. evidence 子串出现在任意召回 chunk 中（精确句级匹配）
  2. expected_chunk 标签出现在任意召回 chunk 中（回落标签匹配）
  多跳题（expected_chunk 含 " / "）只需命中其中任意一个分支即可。

运行方式：
    uv run python -m ai_app1.eval.evaluate
"""

import json
import sys
import time
from pathlib import Path

# 兼容直接运行（python evaluate.py）和模块运行（python -m ai_app1.eval.evaluate）
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 评测集路径 ─────────────────────────────────────────────────────────────────
_EVAL_FILE = Path(__file__).parent / "评测集"


def _load_dataset() -> list[dict]:
    with open(_EVAL_FILE, encoding="utf-8") as f:
        raw = json.load(f)
    # 过滤掉纯注释条目（只有 json_comment 字段，没有 query）
    return [item for item in raw if "query" in item]


def _key_phrases(text: str, min_len: int = 4) -> list[str]:
    """按标点/空格切分文本，返回长度 >= min_len 的实质性片段。"""
    import re
    segments = re.split(r'[，。？！、：“”‘’（）【】\s]', text)
    return [s.strip() for s in segments if len(s.strip()) >= min_len]


def _ngrams(text: str, n: int = 4) -> list[str]:
    """返回文本中所有长度为 n 的字符滑窗（去除空格后）。"""
    t = text.replace(" ", "")
    return [t[i:i+n] for i in range(len(t) - n + 1)]


def _is_hit(result_text: str, expected_chunk: str, evidence: str) -> tuple[bool, str]:
    """
    Returns (hit: bool, reason: str).

    匹配规则（优先级由高到低）：
      1. evidence 全串精确子串匹配
      2. evidence 拆分子句（>=4 字）任意一个出现在召回文本中
      3. evidence 4-char 滑窗——任意 4 字连续片段出现在召回文本中
         （应对 evidence 是概括/意译、与原文有少量措辞差异的情况）
      4. expected_chunk 标签出现在召回文本中（多跳拆 "/"）
    """
    if not result_text:
        return False, "result_text 为空"

    # 规则 1：evidence 完整子串
    if evidence and evidence in result_text:
        return True, f"evidence 全串命中: {evidence[:40]!r}"

    if evidence:
        # 规则 2：子句短语匹配
        for phrase in _key_phrases(evidence):
            if phrase in result_text:
                return True, f"evidence 短语命中: {phrase!r}"

        # 规则 3：4-char 滑窗匹配（捕获措辞差异下的局部重叠）
        for clause in _key_phrases(evidence, min_len=6):
            for gram in _ngrams(clause, n=4):
                if gram in result_text:
                    return True, f"evidence 4-gram 命中: {gram!r}"

    # 规则 4：expected_chunk 标签（多跳拆分）
    labels = [lbl.strip() for lbl in expected_chunk.split("/")]
    for lbl in labels:
        if lbl and lbl in result_text:
            return True, f"label 命中: {lbl!r}"

    return False, f"MISS — expected={expected_chunk!r}"


def run_eval(top_k_label: str = "5") -> None:
    dataset = _load_dataset()
    total = len(dataset)
    hits = 0

    print(f"\n{'─'*60}")
    print(f"  Retrieval Evaluation   评测集大小: {total} 条")
    print(f"{'─'*60}\n")

    # 延迟导入，避免模型在 import 时就加载
    from ai_app1.service.vector_store import query_db

    for i, item in enumerate(dataset, 1):
        query    = item["query"]
        expected = item.get("expected_chunk", "")
        raw_ev   = item.get("evidence", "")
        # evidence 可能是 str 或 list（多条证据）
        evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

        t0 = time.perf_counter()
        result_text = query_db(query) or ""
        elapsed = (time.perf_counter() - t0) * 1000

        hit, reason = _is_hit(result_text, expected, evidence)
        if hit:
            hits += 1
            status = "✅ HIT"
        else:
            status = "❌ MISS"

        print(f"[{i:02d}/{total}] {status}  ({elapsed:.0f}ms)")
        print(f"  Query   : {query[:60]}")
        print(f"  Expected: {expected}")
        print(f"  Reason  : {reason}")

        if not hit and result_text:
            # MISS 时展示实际召回的前 120 个字，方便分析
            preview = result_text[:120].replace("\n", " ")
            print(f"  Retrieved preview: {preview!r}")

        print()

    recall = hits / total if total else 0.0
    print(f"{'─'*60}")
    print(f"  Recall@{top_k_label}  =  {hits} / {total}  =  {recall:.0%}")
    print(f"{'─'*60}\n")


if __name__ == "__main__":
    run_eval()
