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


def _translate_terms(text: str, terms: dict[str, str]) -> str:
    """基于术语映射生成英文查询变体（保留原词追加英文）。"""
    extra: list[str] = []
    for cn, en in terms.items():
        if cn in text and en not in text:
            extra.append(en)
    if extra:
        return f"{text} ({' '.join(extra[:5])})"
    return text


async def analyze_and_route_node(state: WealthState) -> dict[str, Any]:
    """
    分析用户输入，拆解为多路子查询。

    阶段二实现：
      1. 调用 WealthDomainPlugin.classify_query() 识别领域类型
      2. 对 mixed/宏观/财报做子查询拆分
      3. 启用 enable_query_translation 时追加英文术语变体
      4. 返回带 domain 权重的 sub_queries

    输出字段：sub_queries, rewritten_queries, trace
    """
    container = _get_container()
    settings = _get_settings()
    text = state["user_message"]
    trace = list(state.get("trace", []))
    start = time.monotonic()

    sub_queries: list[dict] = []
    rewritten: list[str] = []

    # ── 1. 领域分类 ──────────────────────────────────────────────────────────
    domain = container.domain if container.domain else None
    route = None
    if domain is not None:
        try:
            route = domain.classify_query(text, state.get("history", []))
        except Exception as exc:
            trace.append({"node": "analyze_and_route", "classify_error": str(exc)})

    # ── 2. 子查询拆分 ────────────────────────────────────────────────────────
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
        "sub_queries_count": len(sub_queries),
        "route_type": getattr(route, "type", "unknown") if route else "unknown",
        "latency_ms": round(latency_ms, 1),
    })

    return {
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

    阶段二实现：
      1. 将每个 sub_query（含中英文变体）包装为 QueryRoute
      2. 多路 QueryRoute 传入 HybridRetriever，由其内部 asyncio.gather 并发召回
      3. HybridRetriever 的 Weighted RRF 自动融合各路结果
      4. 截 top_k 拼接为 retrieved_context

    阶段三增强：
      - 从 HybridRetriever 的全局 trace 中提取真实 top_ce 分数

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
    if container.retriever is not None and all_routes:
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
    else:
        trace.append({"node": "parallel_retrieval", "error": "retriever is None or no routes"})

    latency_ms = (time.monotonic() - start) * 1000
    trace.append({
        "node": "parallel_retrieval",
        "has_context": bool(context_text),
        "context_len": len(context_text) if context_text else 0,
        "sub_queries": len(sub_queries),
        "routes": len(all_routes),
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
    """
    trace = list(state.get("trace", []))
    reply = state.get("reply", "")

    trace.append({
        "node": "generate_final",
        "reply_length": len(reply),
    })

    return {
        "reply": reply,
        "trace": trace,
    }
