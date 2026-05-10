"""
轻量知识图谱 (Lightweight KG) — 基于 ChromaDB 元数据构建实体关系网络，
用于扩展检索上下文（当向量检索召回不足时，通过关系跳转补充信息）。

设计约束：
- 不引入外部图数据库，纯 Python 内存实现
- 基于现有 ChromaDB collection 的 metadata 自动构建
- 启动时懒加载，构建完成后缓存
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import chromadb

from ai_app3.core.config import CHROMA_DB_PATH, ENABLE_KNOWLEDGE_GRAPH
from ai_app3.core.logger import kg_logger

# ── 全局缓存 ──
_kg_cache: dict[str, Any] | None = None


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _extract_entities(text: str) -> list[str]:
    """
    轻量实体提取：
    1. Android 类名（大驼峰，如 Activity, ViewModel）
    2. 技术术语（全大写缩写，如 NPE, ANR, OOM）
    3. 关键 API（如 onCreate, findViewById）
    """
    entities: set[str] = set()
    # 大驼峰类名
    entities.update(re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text))
    # 全大写缩写（长度 2~6）
    entities.update(re.findall(r"\b[A-Z]{2,6}\b", text))
    # 关键小驼峰方法
    entities.update(re.findall(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", text))
    # 过滤常见噪声词
    noise = {"The", "This", "That", "Android", "Java", "Kotlin", "XML", "API", "UI", "URL", "HTTP", "JSON", "SQL"}
    entities -= noise
    return sorted(entities)


def _build_kg() -> dict[str, Any]:
    """
    从 ChromaDB 的 android_parent collection 构建轻量知识图谱。
    返回结构：
    {
        "nodes": {entity: {"docs": [doc_id], "freq": int}},
        "edges": [(entity_a, entity_b, weight)],
        "doc_entities": {doc_id: [entity]}
    }
    """
    kg_logger.info("开始构建轻量知识图谱...")
    client = _get_client()
    col = client.get_or_create_collection("android_parent")
    result = col.get(include=["documents"])
    ids = result["ids"]
    docs = result["documents"]

    nodes: dict[str, dict] = {}
    edges: list[tuple[str, str, int]] = []
    doc_entities: dict[str, list[str]] = {}

    for doc_id, text in zip(ids, docs):
        ents = _extract_entities(text)
        doc_entities[doc_id] = ents
        for e in ents:
            if e not in nodes:
                nodes[e] = {"docs": [], "freq": 0}
            nodes[e]["docs"].append(doc_id)
            nodes[e]["freq"] += 1

    # 构建共现边（同一文档中出现的实体两两连接）
    edge_weights: dict[tuple[str, str], int] = {}
    for doc_id, ents in doc_entities.items():
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                key = tuple(sorted((a, b)))
                edge_weights[key] = edge_weights.get(key, 0) + 1

    edges = [(a, b, w) for (a, b), w in edge_weights.items() if w >= 2]

    kg = {"nodes": nodes, "edges": edges, "doc_entities": doc_entities}
    kg_logger.info(
        f"知识图谱构建完成: nodes={len(nodes)}, edges={len(edges)}, docs={len(doc_entities)}"
    )
    return kg


def get_kg() -> dict[str, Any] | None:
    """获取缓存的知识图谱，若未构建则懒加载。"""
    global _kg_cache
    if not ENABLE_KNOWLEDGE_GRAPH:
        return None
    if _kg_cache is None:
        try:
            _kg_cache = _build_kg()
        except Exception as e:
            kg_logger.error(f"知识图谱构建失败: {e}")
            _kg_cache = None
    return _kg_cache


def expand_by_entities(query: str, top_k: int = 5) -> list[str]:
    """
    基于查询中的实体，在知识图谱中查找关联文档，补充检索结果。
    返回 doc_id 列表。
    """
    kg = get_kg()
    if not kg:
        return []

    query_ents = _extract_entities(query)
    if not query_ents:
        return []

    nodes = kg["nodes"]
    related_docs: dict[str, int] = {}

    # 直接命中
    for e in query_ents:
        if e in nodes:
            for doc_id in nodes[e]["docs"]:
                related_docs[doc_id] = related_docs.get(doc_id, 0) + 1

    # 通过边扩展一跳
    edges = kg["edges"]
    for e in query_ents:
        for a, b, w in edges:
            if a == e or b == e:
                neighbor = b if a == e else a
                if neighbor in nodes:
                    for doc_id in nodes[neighbor]["docs"]:
                        related_docs[doc_id] = related_docs.get(doc_id, 0) + w

    # 按关联度排序
    sorted_docs = sorted(related_docs.items(), key=lambda x: x[1], reverse=True)
    top_ids = [doc_id for doc_id, _ in sorted_docs[:top_k]]
    kg_logger.info(f"KG 扩展: query_ents={query_ents}, 补充 docs={len(top_ids)}")
    return top_ids


def fetch_docs_by_ids(doc_ids: list[str]) -> str | None:
    """批量拉取文档文本并拼接。"""
    if not doc_ids:
        return None
    client = _get_client()
    col = client.get_or_create_collection("android_parent")
    result = col.get(ids=doc_ids, include=["documents"])
    docs = result.get("documents", [])
    valid = [d for d in docs if d]
    if not valid:
        return None
    return "\n\n".join(valid)
