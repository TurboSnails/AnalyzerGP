"""
通用命中判定器（Ground Truth Matcher）

将用户问题与召回结果进行匹配，判断检索是否正确命中。
支持 evidence / label / n-gram 多层级回退匹配。
"""
from __future__ import annotations

import re


# ─── 文本处理工具 ───────────────────────────────────────────────────────────────

def key_phrases(text: str, min_len: int = 4) -> list[str]:
    """按标点/空格切分文本，返回长度 >= min_len 的实质性片段。"""
    segments = re.split(r'[，。？！、：“”‘’（）【】\s]', text)
    return [s.strip() for s in segments if len(s.strip()) >= min_len]


def ngrams(text: str, n: int = 4) -> list[str]:
    """返回文本中所有长度为 n 的字符滑窗（去除空格后）。"""
    t = text.replace(" ", "")
    return [t[i:i + n] for i in range(len(t) - n + 1)]


# ─── 命中判定 ───────────────────────────────────────────────────────────────────

def is_hit(result_text: str, expected_chunk: str, evidence: str) -> tuple[bool, str]:
    """
    判断召回文本是否命中 ground truth。

    匹配规则（优先级由高到低）：
      1. evidence 全串精确子串匹配
      2. evidence 拆分子句（>=4 字）任意一个出现在召回文本中
      3. evidence 4-char 滑窗——任意 4 字连续片段出现在召回文本中
         （应对 evidence 是概括/意译、与原文有少量措辞差异的情况）
      4. expected_chunk 标签出现在召回文本中（多跳拆 "/"）

    Returns:
        (hit: bool, reason: str)
    """
    if not result_text:
        return False, "result_text 为空"

    # 规则 1：evidence 完整子串
    if evidence and evidence in result_text:
        return True, f"evidence 全串命中: {evidence[:40]!r}"

    if evidence:
        # 规则 2：子句短语匹配
        for phrase in key_phrases(evidence):
            if phrase in result_text:
                return True, f"evidence 短语命中: {phrase!r}"

        # 规则 3：4-char 滑窗匹配（捕获措辞差异下的局部重叠）
        for clause in key_phrases(evidence, min_len=6):
            for gram in ngrams(clause, n=4):
                if gram in result_text:
                    return True, f"evidence 4-gram 命中: {gram!r}"

    # 规则 4：expected_chunk 标签（多跳拆分）
    labels = [lbl.strip() for lbl in expected_chunk.split("/")]
    for lbl in labels:
        if lbl and lbl in result_text:
            return True, f"label 命中: {lbl!r}"

    return False, f"MISS — expected={expected_chunk!r}"


def ground_truth_ids(item: dict, chunks: list) -> set[str]:
    """
    根据 label / evidence 匹配，返回正确 chunk 的 id 集合。

    支持双模式 Ground Truth：
      1. 若 item 包含 "chunk_ids" 字段，直接做精确匹配（最高优先级）
      2. 否则回退到 label / evidence 模糊匹配

    多跳题（expected_chunk 含 "/"）允许命中任意一个分支。
    """
    # 模式一：精确 Chunk ID 匹配
    precise_ids = item.get("chunk_ids")
    if precise_ids:
        if isinstance(precise_ids, str):
            precise_ids = [precise_ids]
        precise_set = set(precise_ids)
        matched = {c["id"] for c in chunks if c["id"] in precise_set}
        if matched:
            return matched
        # 如果精确 ID 未命中任何 chunk，记录警告并继续回退

    # 模式二：标签 / evidence 模糊匹配
    expected = item.get("expected_chunk", "")
    raw_ev = item.get("evidence", "")
    evidence = " ".join(raw_ev) if isinstance(raw_ev, list) else str(raw_ev)

    matched = set()
    for c in chunks:
        hit, _ = is_hit(c.get("text", ""), expected, evidence)
        if hit:
            matched.add(c["id"])
    return matched
