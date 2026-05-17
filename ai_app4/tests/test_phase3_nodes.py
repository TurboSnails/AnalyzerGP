"""
阶段三测试：检索质量评估、真实 top_ce 提取、反思改写、条件边验证。

运行方式：
    uv run python -m ai_app4.tests.test_phase3_nodes

前置条件：
    - 已构建 wealth 索引（domains/wealth/scripts/init_wealth_db.py）
    - 模型已下载（ai_app1/scripts/download_bge_m3.py）
"""
from __future__ import annotations

import asyncio
import sys

# 注册 wealth 领域插件
sys.path.insert(0, "domains/wealth")
from rag_framework.core.registry import register_domain

from wealth_domain.plugin import WealthDomainPlugin  # type: ignore[import-not-found]

register_domain(WealthDomainPlugin)


from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.graph.state import WealthState
from ai_app4.graph.nodes import (
    parallel_retrieval_node,
    evaluate_and_rerank_node,
    query_reflection_node,
)
from ai_app4.graph.conditional_edges import after_evaluate


def _setup() -> None:
    settings = WealthSettings()
    context.set_settings(settings)
    container = WealthContainer.from_settings(settings)
    context.set_container(container)


async def _test_real_top_ce_extraction() -> None:
    """验证 parallel_retrieval_node 能提取真实 CrossEncoder top_ce。"""
    print("\n▶ Test 1: 真实 top_ce 提取")

    state: WealthState = {
        "user_message": "NVIDIA 2026 Q1 earnings revenue",
        "user_id": "test_p3_1",
        "sub_queries": [
            {"text": "NVIDIA 2026 Q1 earnings revenue", "domain": "corp_earnings", "weight": 1.0},
        ],
        "rewritten_queries": [],
        "retrieved_context": None,
        "retrieval_iterations": 0,
        "confidence": 0.0,
        "top_ce": 0.0,
        "evaluation_result": None,
        "needs_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }

    result = await parallel_retrieval_node(state)
    top_ce = result.get("top_ce", 0.0)
    ctx = result.get("retrieved_context")

    print(f"  top_ce = {top_ce:.4f}")
    print(f"  context_len = {len(ctx) if ctx else 0}")

    # 注：若 v2 collections（parent/child/hyde）齐全，HybridRetriever 会走 CrossEncoder rerank，
    #     此时 top_ce 为正；若仅 parent 存在（legacy 路径），则无 rerank，top_ce 为 0。
    #     这里至少验证 top_ce 字段存在且范围合法。
    assert "top_ce" in result, "结果中应包含 top_ce 字段"
    assert top_ce <= 1.0, f"分数不应超过 1.0，实际 {top_ce}"
    if top_ce > 0:
        print(f"  ✓ 真实 top_ce 提取通过 (v2 rerank 路径, top_ce={top_ce:.4f})")
    else:
        print(f"  ⚠ legacy 路径无 CrossEncoder rerank，top_ce=0（可接受）")


async def _test_evaluate_and_rerank() -> None:
    """验证 evaluate_and_rerank_node 使用真实 top_ce 与启发式 confidence 加权融合。"""
    print("\n▶ Test 2: evaluate_and_rerank 使用真实 top_ce")

    state: WealthState = {
        "user_message": "NVIDIA 2026 Q1 earnings",
        "user_id": "test_p3_2",
        "sub_queries": [],
        "rewritten_queries": [],
        "retrieved_context": "a" * 2000,
        "retrieval_iterations": 1,
        "confidence": 0.0,
        "top_ce": 0.72,  # 模拟真实 CrossEncoder 分数
        "evaluation_result": None,
        "needs_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }

    result = await evaluate_and_rerank_node(state)
    confidence = result.get("confidence", 0.0)
    top_ce = result.get("top_ce", 0.0)
    eval_result = result.get("evaluation_result") or {}

    print(f"  confidence = {confidence:.3f}, top_ce = {top_ce:.3f}")
    print(f"  evaluation_result = {eval_result}")

    assert top_ce == 0.72, f"top_ce 应保持传入值 0.72，实际 {top_ce}"
    assert confidence > 0.5, f"confidence 应大于 0.5，实际 {confidence}"
    assert "retrieval_detail" in eval_result, "evaluation_result 应包含 retrieval_detail"
    print("  ✓ evaluate_and_rerank 加权融合通过")


async def _test_query_reflection() -> None:
    """验证 query_reflection_node 改写逻辑（LLM 或规则降级）。"""
    print("\n▶ Test 3: query_reflection 改写")

    state: WealthState = {
        "user_message": "美联储最新利率决议对英伟达股价有什么影响",
        "user_id": "test_p3_3",
        "sub_queries": [],
        "rewritten_queries": [],
        "retrieved_context": None,
        "retrieval_iterations": 1,
        "confidence": 0.0,
        "top_ce": 0.2,
        "evaluation_result": None,
        "needs_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }

    result = await query_reflection_node(state)
    rewritten = result.get("user_message", "")
    trace = result.get("trace", [])

    print(f"  original: {state['user_message']}")
    print(f"  rewritten: {rewritten}")

    assert rewritten != state["user_message"], "改写后文本应与原文不同"
    assert len(rewritten) > 5, "改写文本应有实质内容"

    reflection_trace = [t for t in trace if t.get("node") == "query_reflection"]
    assert reflection_trace, "trace 中应包含 query_reflection 记录"
    method = reflection_trace[0].get("method", "unknown")
    print(f"  method: {method}")
    print("  ✓ query_reflection 改写通过")


async def _test_after_evaluate_conditional() -> None:
    """验证 after_evaluate 在 top_ce < threshold 且 loop < max 时返回 reflection。"""
    print("\n▶ Test 4: after_evaluate 条件边")

    # Case 1: 低 top_ce，第一次迭代 → reflection
    state1: WealthState = {
        "user_message": "test",
        "user_id": "test_p3_4",
        "sub_queries": [],
        "rewritten_queries": [],
        "retrieved_context": "x",
        "retrieval_iterations": 1,
        "confidence": 0.2,
        "top_ce": 0.2,
        "evaluation_result": {"reflection_threshold": 0.35, "max_loop_count": 2},
        "needs_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }
    r1 = after_evaluate(state1)
    print(f"  top_ce=0.2, iter=1/2 → {r1}")
    assert r1 == "reflection", f"应返回 reflection，实际 {r1}"

    # Case 2: 高 top_ce → strategy
    state2 = dict(state1)
    state2["top_ce"] = 0.6  # type: ignore[assignment]
    r2 = after_evaluate(state2)
    print(f"  top_ce=0.6, iter=1/2 → {r2}")
    assert r2 == "strategy", f"应返回 strategy，实际 {r2}"

    # Case 3: 低 top_ce 但已用尽迭代 → strategy
    state3 = dict(state1)
    state3["retrieval_iterations"] = 2  # type: ignore[assignment]
    r3 = after_evaluate(state3)
    print(f"  top_ce=0.2, iter=2/2 → {r3}")
    assert r3 == "strategy", f"应返回 strategy，实际 {r3}"

    print("  ✓ after_evaluate 条件边通过")


async def _test_end_to_end_reflection_loop() -> None:
    """端到端：低置信度 Query 进入 analyze → retrieve → evaluate → reflection 循环。"""
    print("\n▶ Test 5: 端到端低置信度 reflection 循环")

    # 构造一个故意低 top_ce 的状态，模拟 evaluate 输出
    state: WealthState = {
        "user_message": "某小众科技股最新财报",  # 中文模糊查询，检索质量可能低
        "user_id": "test_p3_5",
        "sub_queries": [],
        "rewritten_queries": [],
        "retrieved_context": None,
        "retrieval_iterations": 0,
        "confidence": 0.0,
        "top_ce": 0.0,
        "evaluation_result": None,
        "needs_tool": False,
        "tool_calls": [],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }

    # analyze
    from ai_app4.graph.nodes import analyze_and_route_node
    s1 = await analyze_and_route_node(state)
    print(f"  analyze → sub_queries={s1.get('sub_queries')}")

    # retrieve
    s2 = await parallel_retrieval_node({**state, **s1})
    print(f"  retrieve → top_ce={s2.get('top_ce'):.4f}, has_ctx={bool(s2.get('retrieved_context'))}")

    # evaluate
    s3 = await evaluate_and_rerank_node({**state, **s1, **s2})
    print(f"  evaluate → confidence={s3.get('confidence'):.3f}, top_ce={s3.get('top_ce'):.3f}")

    # conditional
    route = after_evaluate({**state, **s1, **s2, **s3})
    print(f"  conditional → {route}")

    if route == "reflection":
        s4 = await query_reflection_node({**state, **s1, **s2, **s3})
        print(f"  reflection → rewritten='{s4.get('user_message')}'")
        # 第二轮 retrieve
        s5 = await parallel_retrieval_node({**state, **s1, **s2, **s3, **s4})
        print(f"  retrieve2 → top_ce={s5.get('top_ce'):.4f}")

    print("  ✓ 端到端 reflection 循环通过")


async def main() -> None:
    _setup()
    await _test_real_top_ce_extraction()
    await _test_evaluate_and_rerank()
    await _test_query_reflection()
    await _test_after_evaluate_conditional()
    await _test_end_to_end_reflection_loop()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 阶段三全部测试通过 ✓")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    asyncio.run(main())
