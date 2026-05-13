"""
通用文档分块器

支持段落优先、句级切分、重叠窗口。
"""
from __future__ import annotations

import re
from typing import Iterable


def chunk_paragraphs(
    paragraphs: Iterable[str],
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    核心分块逻辑。

    优先保持段落边界，超长段落按句切分，不与其他段落混拼。
    """
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(para) > chunk_size:
            # flush 当前积累
            if current:
                chunks.append(current.strip())
                current = ""
            # 超长段落独立按句切分
            sentences = re.split(r"(?<=[。！？!?])", para)
            temp = ""
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(temp) + len(sent) <= chunk_size:
                    temp += sent
                else:
                    if temp:
                        chunks.append(temp.strip())
                    temp = sent
            if temp:
                chunks.append(temp.strip())
        elif len(current) + len(para) + 1 <= chunk_size:
            current += para + "\n"
        else:
            if current:
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if overlap > 0 and current else ""
            current = overlap_text + para + "\n"

    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def chunk_file(file_path: str, chunk_size: int, overlap: int) -> list[str]:
    """流式分块读取文件。"""
    def _iter_lines():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                para = line.strip()
                if para:
                    yield para
    return chunk_paragraphs(_iter_lines(), chunk_size, overlap)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """对文本分块。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return chunk_paragraphs(paragraphs, chunk_size, overlap)
