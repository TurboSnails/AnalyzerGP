"""
Query 自动分类器（Query Classifier）

基于规则对评测 query 进行自动分类，用于：
  1. 按类型统计 recall（发现真正的问题在哪）
  2. 为 Failure Analysis 提供维度标签

分类规则（优先级从高到低）：
  - anaphora:   含指代词（这个、上面、那、它）且长度 <20
  - typo:       含常见拼写错误（Hanlder, Acitvity, ViewModle 等）
  - long_query: 长度 >100 字符
  - multi_hop:  含 "+" 或 "和" 连接多个技术概念，或 expected_chunk 含 "/"
  - adversarial: 含错误前提反问（"是不是就不用..."、"应该没问题吧"）
  - code_switching: 中英混合（含英文术语 + 中文字符）
  - vague:      极短（<12 字）或无明确技术关键词
  - keyword:    其余（含明确技术关键词的标准问题）
"""
from __future__ import annotations

import re


# ─── 技术关键词词库 ──────────────────────────────────────────────────────────────

_ANDROID_KEYWORDS = {
    "activity", "fragment", "viewmodel", "livedata", "room", "workmanager",
    "handler", "service", "broadcast", "contentprovider", "intent",
    "anr", "oom", "memory leak", "nullpointerexception", "npe",
    "recyclerview", "listview", "constraintlayout", "linearlayout",
    "compose", "jetpack", "coroutine", "rxjava", "dagger", "hilt",
    "gradle", "proguard", "r8", "jni", "ndk", "kotlin", "java",
    "生命周期", "内存泄漏", "主线程", "子线程", "ui线程", "异步",
    "布局", "适配", "权限", "通知", "数据库", "混淆",
}

_COMMON_TYPOS = {
    "hanlder", "acitvity", "viewmodle", "fragement", "servie",
    "broadcastreciever", "intenet", "recylerview", "constaintlayout",
}

_ANAPHORA_WORDS = {"这个", "上面", "那", "它", "这种", "那样", "之前"}
_ADVERSARIAL_PATTERNS = [
    r"是不是就不用",
    r"应该没问题吧",
    r"直接.*就行",
    r"反正.*就",
    r"都说.*那",
]


# ─── 分类函数 ───────────────────────────────────────────────────────────────────

def classify_query_type(query: str, expected_chunk: str = "") -> str:
    """
    对 query 进行自动分类。

    Args:
        query: 用户查询文本
        expected_chunk: ground truth chunk 标签（用于辅助判断 multi-hop）

    Returns:
        分类标签: keyword | vague | multi_hop | typo | long_query |
                 anaphora | adversarial | code_switching
    """
    q_lower = query.lower().replace(" ", "")
    q_nospace = q_lower.replace(" ", "")
    length = len(query)

    # 1. typo
    for typo in _COMMON_TYPOS:
        if typo in q_nospace:
            return "typo"

    # 2. anaphora
    if length < 20:
        for word in _ANAPHORA_WORDS:
            if word in query:
                return "anaphora"

    # 3. long_query
    if length > 100:
        return "long_query"

    # 4. adversarial
    for pat in _ADVERSARIAL_PATTERNS:
        if re.search(pat, query):
            return "adversarial"

    # 5. code_switching
    cn_chars = len(re.findall(r"[\u4e00-\u9fff]", query))
    en_words = len(re.findall(r"[a-zA-Z]{3,}", query))
    if cn_chars > 3 and en_words > 1:
        return "code_switching"

    # 6. multi-hop
    if "/" in expected_chunk:
        return "multi_hop"
    tech_concepts = re.findall(r"[A-Z][a-zA-Z]+| Jetpack| Compose| Room| WorkManager", query)
    if len(tech_concepts) >= 3:
        return "multi_hop"
    if "+" in query or ("和" in query and len(tech_concepts) >= 2):
        return "multi_hop"

    # 7. vague
    if length < 12:
        return "vague"

    # 检查是否有明确技术关键词
    has_keyword = False
    for kw in _ANDROID_KEYWORDS:
        if kw in q_lower:
            has_keyword = True
            break
    if not has_keyword:
        return "vague"

    return "keyword"


def classify_query_type_from_item(item: dict) -> str:
    """从 benchmark/hard_cases 的 item dict 中读取 difficulty 字段或自动分类。"""
    if "difficulty" in item:
        return item["difficulty"]
    return classify_query_type(item.get("query", ""), item.get("expected_chunk", ""))


# ─── 分类统计 ───────────────────────────────────────────────────────────────────

def aggregate_by_type(results: list[dict]) -> dict[str, dict]:
    """
    按 query type 聚合评测结果。

    Args:
        results: evaluate_single_query 返回的 dict 列表，每条含 "query_type"

    Returns:
        {type: {"count": int, "recall@5": float, "hit@1": float, "mrr": float, "avg_latency_ms": float}}
    """
    groups: dict[str, list[dict]] = {}
    for r in results:
        qt = r.get("query_type", "unknown")
        groups.setdefault(qt, []).append(r)

    stats = {}
    for qt, items in groups.items():
        n = len(items)
        stats[qt] = {
            "count": n,
            "recall@5": sum(i.get("recall@5", 0.0) for i in items) / n,
            "hit@1": sum(i.get("hit@1", 0.0) for i in items) / n,
            "hit@3": sum(i.get("hit@3", 0.0) for i in items) / n,
            "hit@5": sum(i.get("hit@5", 0.0) for i in items) / n,
            "mrr": sum(i.get("rr", 0.0) for i in items) / n,
            "avg_latency_ms": sum(i.get("latency_ms", 0.0) for i in items) / n,
        }
    return stats


def format_type_stats(stats: dict[str, dict]) -> str:
    """格式化分类统计表格。"""
    lines = [
        "\n📊 Query 分类统计",
        "─" * 70,
        f"{'类型':<18} {'数量':>6} {'Recall@5':>10} {'Hit@1':>8} {'MRR':>8} {'平均延迟':>10}",
        "─" * 70,
    ]
    # 按数量排序
    for qt, s in sorted(stats.items(), key=lambda x: -x[1]["count"]):
        lines.append(
            f"{qt:<18} {s['count']:>6} {s['recall@5']:>9.1%} {s['hit@1']:>7.1%} "
            f"{s['mrr']:>7.3f} {s['avg_latency_ms']:>9.0f}ms"
        )
    lines.append("─" * 70)
    return "\n".join(lines)
