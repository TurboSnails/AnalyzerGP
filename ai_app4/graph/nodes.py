"""
ai_app4 Wealth AI Agent LangGraph 节点实现。

每个节点接收 WealthState，返回 state 的部分更新字典。
所有异步操作（模型推理、检索、LLM 调用）通过 await 处理。

阶段一骨架说明：
  - analyze_and_route_node : 先用规则+LLM 混合做简单拆解（阶段二完善中英文翻译）
  - parallel_retrieval_node : 复用 HybridRetriever 单路检索（阶段二完善多域并发）
  - evaluate_and_rerank_node : 启发式 confidence（阶段三接入真实 top_ce）
  - query_reflection_node : 简单改写（阶段三完善金融术语化反思）
  - strategy_reasoning_node : 直接调用 LLM 生成，needs_tool=False（阶段四完善工具识别）
  - execute_math_tool_node : pass through（阶段四实现真实工具调用）
  - merge_and_generate_node : 复用 LLM 生成（阶段四合并计算结果）
  - generate_final_node : 纯文本 LLM 生成
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.graph.state import WealthState


# ── 辅助：获取容器与配置 ────────────────────────────────────────────────────

def _get_container() -> WealthContainer:
    """从全局上下文获取容器（避免循环导入 main.py）。"""
    container = context.get_container()
    if container is None:
        container = WealthContainer.from_settings(WealthSettings())
        context.set_container(container)
    return container


def _get_settings() -> WealthSettings:
    """从全局上下文获取 WealthSettings；若缺失或类型不符则新建。"""
    settings = context.get_settings()
    if settings is None or not isinstance(settings, WealthSettings):
        settings = WealthSettings()
        context.set_settings(settings)
    return settings


def _add_trace(state: WealthState, node: str, detail: dict) -> list[dict]:
    """向 trace 列表追加节点执行记录。"""
    trace = list(state.get("trace", []))
    trace.append({"node": node, **detail})
    return trace


# ═════════════════════════════════════════════════════════════════════════════
# Node: analyze_and_route_node — 分析与路由
# ═════════════════════════════════════════════════════════════════════════════

# 启发式连接词：检测到这些词时尝试拆分跨域子查询
_SPLIT_MARKERS = ("和", "与", "以及", "同时", "结合", "影响", "对比", "vs", "VS")

# 时效性关键词：检测到这些词时标记查询为时间敏感，需要启用 Track B/C
_TIME_SENSITIVE_MARKERS = (
    "今天", "今日", "现在", "当前", "最新", "实时", "即时",
    "刚才", "刚刚", "最近", "近日", "本周", "本月", "今年",
    "今早", "今晚", "昨天", "昨日", "明日", "明天",
    "开盘", "收盘", "盘中", "盘后", "盘前",
    "跌了多少", "涨了多少", "多少点", "多少钱", "什么价",
    "now", "today", "current", "latest", "real-time", "live",
)

# NER 模式：用于从查询中提取金融实体
_TICKER_PATTERN = r"\b[A-Z]{1,5}\b"  # 大写股票代码（如 NVDA, TSLA）
_DATE_PATTERN = r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日)"
_MACRO_PATTERN = r"(CPI|PPI|GDP|PMI|非农|失业率|利率|国债|通胀|通缩)"


def _is_time_sensitive(text: str) -> bool:
    """检测查询是否包含时效性关键词。"""
    lower = text.lower()
    return any(marker in text or marker in lower for marker in _TIME_SENSITIVE_MARKERS)


def _extract_entities(text: str) -> list[dict]:
    """
    从查询文本中提取金融相关实体（NER）。

    返回格式: [{"type": "ticker", "value": "NVDA", "confidence": 0.95}, ...]
    """
    import re
    entities: list[dict] = []
    seen: set[str] = set()

    # 1. 股票代码（大写 1-5 字母，过滤常见非 ticker 词）
    non_tickers = {"USD", "CNY", "HKD", "ETF", "IPO", "CEO", "CFO", "CTO", "USA", "USB"}
    for match in re.finditer(_TICKER_PATTERN, text):
        ticker = match.group()
        if ticker not in non_tickers:
            key = f"ticker:{ticker}"
            if key not in seen:
                seen.add(key)
                entities.append({"type": "ticker", "value": ticker, "confidence": 0.95})

    # 2. 宏观指标
    for match in re.finditer(_MACRO_PATTERN, text):
        indicator = match.group()
        key = f"indicator:{indicator}"
        if key not in seen:
            seen.add(key)
            entities.append({"type": "macro_indicator", "value": indicator, "confidence": 0.90})

    # 3. 日期
    for match in re.finditer(_DATE_PATTERN, text):
        date_str = match.group()
        key = f"date:{date_str}"
        if key not in seen:
            seen.add(key)
            entities.append({"type": "date", "value": date_str, "confidence": 0.85})

    # 4. 已知股票中文名映射（扩展 YahooFinanceSource 的映射）
    _CN_TICKER_MAP = {
        "英伟达": "NVDA", "nvidia": "NVDA",
        "特斯拉": "TSLA", "苹果": "AAPL",
        "微软": "MSFT", "谷歌": "GOOGL",
        "亚马逊": "AMZN", "meta": "META",
        "脸书": "META", "amd": "AMD",
        "英特尔": "INTC", "台积电": "TSM",
        "阿里巴巴": "BABA", "腾讯": "TCEHY",
        "拼多多": "PDD", "小米": "XIACY",
    }
    lower_text = text.lower()
    for keyword, ticker in _CN_TICKER_MAP.items():
        if keyword in lower_text:
            key = f"ticker:{ticker}"
            if key not in seen:
                seen.add(key)
                entities.append({"type": "ticker", "value": ticker, "confidence": 0.90})

    return entities


def _translate_terms(text: str, terms: dict[str, str]) -> str:
    """基于术语映射生成英文查询变体（保留原词追加英文）。"""
    extra: list[str] = []
    for cn, en in terms.items():
        if cn in text and en not in text:
            extra.append(en)
    if extra:
        return f"{text} ({' '.join(extra[:5])})"
    return text


# ── 商业版辅助：来源标注与合规 ────────────────────────────────────────────────

def _build_source_footer(trace: list[dict]) -> str:
    """从 trace 中提取检索来源信息，生成数据来源脚注。"""
    if not trace:
        return ""

    # 查找 parallel_retrieval 节点的 tracks_used
    tracks_used: list[str] = []
    for entry in trace:
        if entry.get("node") == "parallel_retrieval" and "tracks_used" in entry:
            tracks_used = entry["tracks_used"]
            break

    if not tracks_used:
        return ""

    source_labels: dict[str, str] = {
        "track_a": "本地知识库",
        "track_b": "实时金融数据 API",
        "track_c": "网络搜索",
    }
    parts = [source_labels.get(t, t) for t in tracks_used if t in source_labels]
    if not parts:
        return ""

    return f"\n\n—\n📎 数据来源：{'、'.join(parts)}"


def _append_compliance(reply: str, settings: WealthSettings) -> str:
    """在回复末尾追加投资免责声明（如启用）。"""
    if not getattr(settings, "enable_compliance_disclaimer", True):
        return reply

    disclaimer = (
        "\n\n⚠️ 免责声明：以上内容仅供参考，不构成任何投资建议。"
        "金融市场存在风险，过往业绩不代表未来表现。"
        "投资者应独立判断并自行承担风险。"
    )
    return reply + disclaimer


async def analyze_and_route_node(state: WealthState) -> dict[str, Any]:
    """
    分析用户输入，拆解为多路子查询。

    商业级别增强：
      1. 时效性检测 — 识别"今天"、"实时"、"最新"等关键词，决定是否启用 Track B/C
      2. NER 实体提取 — 提取股票代码、宏观指标、日期等实体
      3. 调用 WealthDomainPlugin.classify_query() 识别领域类型
      4. 对 mixed/宏观/财报做子查询拆分
      5. 启用 enable_query_translation 时追加英文术语变体

    输出字段：time_sensitive, entities, sub_queries, rewritten_queries, trace
    """
    container = _get_container()
    settings = _get_settings()
    text = state["user_message"]
    trace = list(state.get("trace", []))
    start = time.monotonic()

    # ── 0. 时效性检测 + NER 实体提取 ─────────────────────────────────────────
    time_sensitive = _is_time_sensitive(text)
    entities = _extract_entities(text)

    # ── 1. 领域分类 ──────────────────────────────────────────────────────────
    domain = container.domain if container.domain else None
    route = None
    if domain is not None:
        try:
            route = domain.classify_query(text, state.get("history", []))
        except Exception as exc:
            trace.append({"node": "analyze_and_route", "classify_error": str(exc)})

    # ── 2. 子查询拆分 ────────────────────────────────────────────────────────
    sub_queries: list[dict] = []
    rewritten: list[str] = []

    if not settings.enable_query_decomposition or route is None:
        sub_queries.append({"text": text, "domain": "all", "weight": 1.0})
        rewritten.append(text)
    else:
        # 根据 route type 决定拆分策略
        if route.type == "mixed":
            # 混合问题：拆为宏观 + 财报两路
            sub_queries = [
                {"text": text, "domain": "macro_econ", "weight": 0.5},
                {"text": text, "domain": "corp_earnings", "weight": 0.5},
            ]
        elif route.type == "macro":
            sub_queries = [
                {"text": text, "domain": "macro_econ", "weight": route.weight},
            ]
        elif route.type == "corp":
            sub_queries = [
                {"text": text, "domain": "corp_earnings", "weight": route.weight},
            ]
        else:
            sub_queries = [
                {"text": text, "domain": "all", "weight": route.weight},
            ]
        rewritten = [sq["text"] for sq in sub_queries]

    # ── 3. 中英文术语翻译（追加英文关键词）───────────────────────────────────
    if settings.enable_query_translation and domain is not None:
        terms = domain.get_term_mapping()
        for sq in sub_queries:
            sq["text_en"] = _translate_terms(sq["text"], terms)

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "analyze_and_route",
        "time_sensitive": time_sensitive,
        "entities_count": len(entities),
        "entities_types": list({e["type"] for e in entities}),
        "sub_queries_count": len(sub_queries),
        "route_type": getattr(route, "type", "unknown") if route else "unknown",
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "time_sensitive": time_sensitive,
        "entities": entities,
        "sub_queries": sub_queries,
        "rewritten_queries": rewritten,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: parallel_retrieval_node — 并行检索
# ═════════════════════════════════════════════════════════════════════════════

async def parallel_retrieval_node(state: WealthState) -> dict[str, Any]:
    """
    执行检索：复用 HybridRetriever（BM25 + Dense + Rerank）。

    商业版增强（三轨检索）：
      - 当 three_track_enabled=True 且 ThreeTrackRetriever 已初始化时，
        优先使用三轨检索器（本地 RAG + 金融 API + 网络搜索）。
      - 否则回退到原生 HybridRetriever（Track A 本地检索）。

    输出字段：retrieved_context, retrieval_iterations, top_ce, trace
    """
    from rag_framework.domain.base import QueryRoute
    from rag_framework.eval.retrieval_trace import get_recent_traces

    container = _get_container()
    settings = _get_settings()
    trace = list(state.get("trace", []))
    start = time.monotonic()

    sub_queries = state.get("sub_queries", [])
    if not sub_queries:
        sub_queries = [{"text": state.get("user_message", ""), "domain": "all", "weight": 1.0}]

    context_text: str | None = None
    tracks_used: list[str] = ["track_a"]  # 默认至少使用本地 Track A

    # 构造多路 QueryRoute：每个子查询 + 英文变体（如有）
    all_routes: list[QueryRoute] = []
    for sq in sub_queries:
        text = sq.get("text", "")
        weight = sq.get("weight", 1.0)
        domain_label = sq.get("domain", "all")

        all_routes.append(
            QueryRoute(text=text, type=domain_label, weight=weight, routes=["dense", "bm25"])
        )

        text_en = sq.get("text_en")
        if text_en and text_en != text:
            all_routes.append(
                QueryRoute(
                    text=text_en,
                    type=f"{domain_label}_en",
                    weight=round(weight * 0.85, 2),
                    routes=["dense", "bm25"],
                )
            )

    top_ce = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # 商业版：三轨检索优先路径
    # ═══════════════════════════════════════════════════════════════════════
    three_track = None
    if getattr(settings, "three_track_enabled", False):
        try:
            import ai_app4.core.context as ctx
            three_track = ctx.get_three_track_retriever()
        except Exception:
            three_track = None

    if three_track is not None and all_routes:
        try:
            from rag_framework.datasource.base import FetchContext

            fetch_ctx = FetchContext(
                user_id=state.get("user_id", "anonymous"),
                session_id=state.get("session_id", ""),
                time_sensitive=state.get("time_sensitive", False),
                entities=state.get("entities", []),
                history=state.get("messages", [])[-6:] if state.get("messages") else [],
            )
            result = await three_track.retrieve(
                all_routes, top_k=settings.retriever_top_k, context=fetch_ctx
            )
            if result.docs:
                context_text = "\n\n".join(
                    d.text for d in result.docs[:settings.cross_encoder_top_k] if d.text
                )
            tracks_used = ["track_a", "track_b", "track_c"]
            # 三轨检索器内部已做 RRF 融合，trace 信息通过其他方式记录
            if result.docs:
                top_ce = max(
                    (getattr(d, "score", 0.0) or 0.0 for d in result.docs),
                    default=0.0,
                )
        except Exception as exc:
            trace.append({"node": "parallel_retrieval", "three_track_error": str(exc)})
            # 三轨失败时回退到本地检索
            three_track = None

    # ═══════════════════════════════════════════════════════════════════════
    # 标准路径：本地 HybridRetriever（回退或默认）
    # ═══════════════════════════════════════════════════════════════════════
    if three_track is None and container.retriever is not None and all_routes:
        try:
            result = await container.retriever.retrieve(all_routes, top_k=settings.retriever_top_k)
            if result.docs:
                context_text = "\n\n".join(
                    d.text for d in result.docs[:settings.cross_encoder_top_k] if d.text
                )
            # 阶段三：从全局 trace 提取真实 CrossEncoder top_ce 分数
            recent_traces = get_recent_traces(1)
            if recent_traces:
                top_ce = recent_traces[0].top_ce_score
            else:
                # legacy 路径无 trace，用 docs 最高 score 做近似 fallback
                if result.docs:
                    top_ce = max(
                        (getattr(d, "score", 0.0) or 0.0 for d in result.docs),
                        default=0.0,
                    )
        except Exception as exc:
            trace.append({"node": "parallel_retrieval", "error": str(exc)})
    elif three_track is None:
        trace.append({"node": "parallel_retrieval", "error": "retriever is None or no routes"})

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "parallel_retrieval",
        "has_context": bool(context_text),
        "context_len": len(context_text) if context_text else 0,
        "sub_queries": len(sub_queries),
        "routes": len(all_routes),
        "tracks_used": tracks_used,
        "top_ce": round(top_ce, 4),
        "latency_ms": round(latency_ms, 1),
    })

    iterations = state.get("retrieval_iterations", 0) + 1
    return {
        "retrieved_context": context_text,
        "retrieval_iterations": iterations,
        "top_ce": top_ce,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: evaluate_and_rerank_node — 检索质量评估
# ═════════════════════════════════════════════════════════════════════════════

async def evaluate_and_rerank_node(state: WealthState) -> dict[str, Any]:
    """
    评估检索质量，计算置信度与 top_ce。

    阶段一骨架：启发式 confidence（基于上下文长度）。
    阶段三完善：
      - 优先使用 parallel_retrieval_node 传来的真实 top_ce（来自 CrossEncoder sigmoid）
      - 将真实 top_ce 与启发式 confidence 加权融合
      - 接入 latency breakdown 与 retrieval trace

    输出字段：confidence, top_ce, evaluation_result, trace
    """
    settings = _get_settings()
    ctx = state.get("retrieved_context")
    trace = list(state.get("trace", []))
    start = time.monotonic()

    # 阶段三：优先使用真实的 top_ce（已由 parallel_retrieval_node 写入 state）
    real_top_ce = state.get("top_ce", 0.0)
    confidence = 0.0
    top_ce = real_top_ce

    if ctx:
        # 启发式基础 confidence
        base_conf = min(0.5 + len(ctx) / 3000, 0.9)
        iterations = state.get("retrieval_iterations", 0)
        if iterations > 1:
            base_conf *= 0.85  # 改写后仍检索，降低置信度

        if top_ce > 0:
            # CrossEncoder top_ce 为强信号，加权融合
            confidence = 0.6 * top_ce + 0.4 * base_conf
        else:
            confidence = base_conf

    # 阶段三：接入 retrieval trace 详情
    retrieval_detail: dict | None = None
    if settings.enable_trace:
        from rag_framework.eval.retrieval_trace import get_recent_traces
        recent = get_recent_traces(1)
        if recent:
            rt = recent[0]
            retrieval_detail = {
                "branches": len(rt.branches) if rt.branches else 0,
                "rerank_docs": len(rt.rerank.reranked_ids) if rt.rerank else 0,
                "latency_ms": round(rt.total_latency_ms, 1) if rt.total_latency_ms else None,
            }

    latency_ms = (time.monotonic() - start) * 1000
    eval_result = {
        "confidence": round(confidence, 3),
        "top_ce": round(top_ce, 3),
        "reflection_threshold": settings.reflection_threshold,
        "max_loop_count": settings.max_loop_count,
        "iterations": iterations if ctx else 0,
        "retrieval_detail": retrieval_detail,
        "eval_latency_ms": round(latency_ms, 1),
    }

    trace.append({
        "node": "evaluate_and_rerank",
        "confidence": round(confidence, 3),
        "top_ce": round(top_ce, 3),
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "confidence": confidence,
        "top_ce": top_ce,
        "evaluation_result": eval_result,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: query_reflection_node — 查询反思与改写
# ═════════════════════════════════════════════════════════════════════════════

_REFLECTION_SYSTEM_PROMPT = (
    "你是一位金融信息检索优化专家。"
    "用户在上一次检索中未能获得足够相关的文档。"
    "请分析原查询的问题（如术语不精确、领域模糊、缺少英文关键词），"
    "并输出一条改写后的查询，要求：\n"
    "1. 保留核心金融实体（股票代码、经济指标名称）；\n"
    "2. 将模糊中文术语替换为更精确的英文金融关键词；\n"
    "3. 添加领域限定词（如 'earnings call', 'FOMC statement', 'market outlook'）；\n"
    "4. 只输出改写后的查询文本，不要任何解释。"
)


async def query_reflection_node(state: WealthState) -> dict[str, Any]:
    """
    反思检索未命中原因，改写 Query 以提升下一轮检索质量。

    阶段一骨架：简单前缀改写。
    阶段三完善：
      - 优先尝试 LLM 驱动的深度反思（分析失败原因 + 金融术语化改写）
      - LLM 不可用时降级为 WealthDomainPlugin 术语映射规则改写

    输出字段：user_message（改写后）, sub_queries, trace
    """
    text = state.get("user_message", "")
    trace = list(state.get("trace", []))
    start = time.monotonic()

    container = _get_container()
    settings = _get_settings()
    rewritten = text
    method = "none"

    # ── 1. 优先尝试 LLM 驱动的反思改写 ───────────────────────────────────────
    if container.llm is not None:
        try:
            messages = [
                {"role": "system", "content": _REFLECTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"原查询：{text}\n\n请给出改写后的查询："},
            ]
            raw = await container.llm.chat(messages, use_tools=False)
            if raw and len(raw.strip()) > 5:
                rewritten = raw.strip()
                method = "llm_reflection"
        except Exception as exc:
            trace.append({
                "node": "query_reflection",
                "warning": f"LLM 反思失败: {exc}",
            })

    # ── 2. LLM 失败或不可用时，使用规则驱动的术语改写 ─────────────────────────
    if method == "none" and container.domain is not None:
        try:
            terms = container.domain.get_term_mapping()
            rewritten = _translate_terms(text, terms)
            # 若术语翻译未改变文本，则追加通用金融增强词
            if rewritten == text:
                rewritten = f"{text} (financial market analysis investment outlook)"
            method = "term_translation"
        except Exception:
            rewritten = f"【改写】{text}"
            method = "fallback_prefix"
    elif method == "none":
        rewritten = f"【改写】{text}"
        method = "fallback_prefix"

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "query_reflection",
        "original": text,
        "rewritten": rewritten,
        "method": method,
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "user_message": rewritten,
        "sub_queries": [{"text": rewritten, "domain": "all", "weight": 1.0}],
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: strategy_reasoning_node — 策略推演
# ═════════════════════════════════════════════════════════════════════════════

_TOOL_HINT = (
    "\n\n【可用计算工具】\n"
    "当用户问题涉及仓位计算、网格策略设计、组合回撤估算或复利收益时，"
    "你必须在分析末尾输出工具调用指令（严格 JSON 格式），格式如下：\n"
    'TOOL_CALL: {"name": "kelly_criterion_calc", "arguments": {"win_rate": 0.55, "avg_gain_pct": 8, "avg_loss_pct": 4, "current_capital": 100000}}\n'
    "可用工具：\n"
    "- kelly_criterion_calc: 凯利公式仓位（参数：win_rate, avg_gain_pct, avg_loss_pct, current_capital）\n"
    "- grid_trading_cost_estimator: 网格交易成本（参数：lower_bound, upper_bound, num_grids, total_capital, fee_rate_pct）\n"
    "- portfolio_drawdown_estimator: 组合回撤估算（参数：allocations, drawdown_scenarios, total_capital）\n"
    "- compound_growth_calculator: 复利增长（参数：principal, annual_return_pct, years, monthly_contribution）\n"
)


def _parse_tool_calls(text: str) -> list[dict]:
    """从 LLM 回复中解析 TOOL_CALL JSON 指令（支持嵌套 JSON）。"""
    import json

    calls: list[dict] = []
    idx = 0
    while True:
        start = text.find("TOOL_CALL:", idx)
        if start == -1:
            break
        brace = text.find("{", start)
        if brace == -1:
            break
        # 手动计数花括号，找到匹配的右括号
        depth = 0
        end = brace
        for i in range(brace, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if depth != 0:
            idx = brace + 1
            continue
        raw = text[brace:end + 1]
        try:
            obj = json.loads(raw)
            if "name" in obj and "arguments" in obj:
                calls.append({"name": obj["name"], "arguments": obj["arguments"]})
        except json.JSONDecodeError:
            pass
        idx = end + 1
    return calls


async def strategy_reasoning_node(state: WealthState) -> dict[str, Any]:
    """
    主模型多步推演，生成回复并判断是否需要计算工具。

    阶段四完善：
      - system prompt 注入可用工具描述
      - 解析 LLM 输出中的 TOOL_CALL 指令
      - 若识别到工具调用 → needs_tool=True

    输出字段：reply, needs_tool, tool_calls, trace
    """
    container = _get_container()
    settings = _get_settings()
    trace = list(state.get("trace", []))
    start = time.monotonic()

    # 构建 messages
    messages: list[dict] = []
    system_prompt = settings.wealth_system_prompt + _TOOL_HINT
    messages.append({"role": "system", "content": system_prompt})

    summary = state.get("summary", "")
    if summary:
        messages.append({"role": "user", "content": f"【历史摘要】{summary}"})

    for h in state.get("history", [])[-settings.max_history_per_session:]:
        messages.append(h)

    ctx = state.get("retrieved_context")
    if ctx:
        messages.append({"role": "user", "content": f"参考资料：{ctx}"})

    messages.append({"role": "user", "content": state["user_message"]})

    # 调用 LLM
    reply = ""
    try:
        reply = await container.llm.chat(messages)
    except Exception as exc:
        reply = f"抱歉，服务暂时异常，请稍后重试。（{exc}）"
        trace.append({"node": "strategy_reasoning", "error": str(exc)})

    # 阶段四：解析工具调用
    tool_calls = _parse_tool_calls(reply)
    needs_tool = bool(tool_calls) and settings.math_tool_enabled

    # 清理 reply：移除 TOOL_CALL 标记，保留自然语言分析部分
    clean_reply = reply
    if tool_calls:
        import re
        clean_reply = re.sub(r'TOOL_CALL:\s*\{.*?\}', '', reply, flags=re.DOTALL).strip()

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "strategy_reasoning",
        "reply_length": len(clean_reply),
        "needs_tool": needs_tool,
        "tool_calls_count": len(tool_calls),
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "reply": clean_reply,
        "needs_tool": needs_tool,
        "tool_calls": tool_calls,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: execute_math_tool_node — 执行数学计算工具
# ═════════════════════════════════════════════════════════════════════════════

async def execute_math_tool_node(state: WealthState) -> dict[str, Any]:
    """
    从 tool_calls 提取参数，调用 Python 硬计算函数。

    阶段四完善：
      - 遍历 tool_calls，通过 tool_registry.execute_tool 执行
      - 收集结果到 math_result

    输出字段：tool_results, math_result, trace
    """
    from rag_framework.llm.tool_registry import execute_tool

    trace = list(state.get("trace", []))
    tool_calls = state.get("tool_calls", [])
    start = time.monotonic()

    tool_results: list[dict] = []
    math_result: dict[str, Any] = {}

    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("arguments", {})
        try:
            result = execute_tool(name, args)
            tool_results.append({
                "tool": name,
                "arguments": args,
                "result": result,
                "status": "success",
            })
            math_result[name] = result
        except Exception as exc:
            tool_results.append({
                "tool": name,
                "arguments": args,
                "error": str(exc),
                "status": "error",
            })
            math_result[name] = {"error": str(exc)}

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "execute_math_tool",
        "tool_calls_count": len(tool_calls),
        "success_count": sum(1 for t in tool_results if t["status"] == "success"),
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "tool_results": tool_results,
        "math_result": math_result if math_result else None,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: merge_and_generate_node — 合并计算结果与生成
# ═════════════════════════════════════════════════════════════════════════════

async def merge_and_generate_node(state: WealthState) -> dict[str, Any]:
    """
    合并数学计算结果与 LLM 话术，生成严谨金融报告。

    阶段四完善：
      - 将 math_result 格式化为结构化文本摘要
      - 若 LLM 可用，将计算结果注入 prompt 重新润色生成最终回复
      - 若 LLM 不可用，直接格式化追加到原 reply

    输出字段：reply, trace
    """
    import json

    container = _get_container()
    settings = _get_settings()
    trace = list(state.get("trace", []))
    reply = state.get("reply", "")
    math_result = state.get("math_result")
    start = time.monotonic()

    # 格式化 math_result 为可读文本
    math_text = ""
    if math_result and isinstance(math_result, dict):
        parts: list[str] = []
        for tool_name, result in math_result.items():
            if isinstance(result, dict) and "error" not in result:
                parts.append(f"■ {tool_name}:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
            elif isinstance(result, dict):
                parts.append(f"■ {tool_name}: 计算失败 — {result.get('error', 'unknown')}")
            else:
                parts.append(f"■ {tool_name}: {result}")
        if parts:
            math_text = "\n\n【计算结果摘要】\n" + "\n\n".join(parts)

    final_reply = reply

    # 尝试让 LLM 基于计算结果重新润色
    if container.llm is not None and math_text:
        try:
            merge_prompt = (
                "你是一位严谨的金融报告撰写专家。"
                "请根据以下原始分析和精确计算结果，生成一段简洁、专业、数据准确的最终回复。"
                "要求：直接给出结论和建议，不要重复解释公式。\n\n"
                f"【原始分析】\n{reply}\n\n"
                f"{math_text}\n\n"
                "请输出最终回复："
            )
            merged = await container.llm.chat(
                [{"role": "user", "content": merge_prompt}],
                use_tools=False,
            )
            if merged and len(merged.strip()) > 10:
                final_reply = merged.strip()
        except Exception as exc:
            trace.append({"node": "merge_and_generate", "warning": f"LLM 润色失败: {exc}"})

    # LLM 失败或未启用时，直接追加格式化结果
    if final_reply == reply and math_text:
        final_reply = reply + math_text

    # 商业版：来源标注 + 合规免责声明
    if getattr(settings, "enable_source_attribution", True):
        final_reply += _build_source_footer(state.get("trace", []))
    final_reply = _append_compliance(final_reply, settings)

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "merge_and_generate",
        "has_math_result": bool(math_result),
        "reply_length": len(final_reply),
        "latency_ms": round(latency_ms, 1),
    })

    return {
        "reply": final_reply,
        "trace": trace,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Node: generate_final_node — 纯文本最终回复
# ═════════════════════════════════════════════════════════════════════════════

async def generate_final_node(state: WealthState) -> dict[str, Any]:
    """
    纯文本最终回复（无需工具时）。

    直接复用 strategy_reasoning 已生成的 reply，追加 trace。
    商业版增强：追加数据来源标注与合规免责声明。
    """
    settings = _get_settings()
    trace = list(state.get("trace", []))
    reply = state.get("reply", "")

    # 商业版：来源标注 + 合规免责声明
    if getattr(settings, "enable_source_attribution", True):
        reply += _build_source_footer(state.get("trace", []))
    reply = _append_compliance(reply, settings)

    trace.append({
        "node": "generate_final",
        "reply_length": len(reply),
    })

    return {
        "reply": reply,
        "trace": trace,
    }
