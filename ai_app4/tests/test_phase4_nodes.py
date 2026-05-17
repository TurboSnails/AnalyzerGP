"""
阶段四测试：数学工具函数、工具注册、strategy/execute/merge 节点、端到端链路。

运行方式：
    uv run python -m ai_app4.tests.test_phase4_nodes
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "domains/wealth")
from rag_framework.core.registry import register_domain
from wealth_domain.plugin import WealthDomainPlugin  # type: ignore[import-not-found]

register_domain(WealthDomainPlugin)

from ai_app4.core.config import WealthSettings
from ai_app4.core.container import WealthContainer
from ai_app4.core import context
from ai_app4.graph.state import WealthState
from ai_app4.graph.nodes import (
    strategy_reasoning_node,
    execute_math_tool_node,
    merge_and_generate_node,
    _parse_tool_calls,
)
from ai_app4.graph.conditional_edges import after_strategy
from ai_app4.service.math_tools import (
    kelly_criterion_calc,
    grid_trading_cost_estimator,
    portfolio_drawdown_estimator,
    compound_growth_calculator,
)
from ai_app4.service.tools import register_wealth_tools
from rag_framework.llm.tool_registry import list_tools, execute_tool


def _setup() -> None:
    settings = WealthSettings()
    context.set_settings(settings)
    container = WealthContainer.from_settings(settings)
    context.set_container(container)
    register_wealth_tools()


async def _test_kelly_criterion() -> None:
    print("\n▶ Test 1: 凯利公式计算")
    result = kelly_criterion_calc(
        win_rate=0.55, avg_gain_pct=8.0, avg_loss_pct=4.0, current_capital=100_000
    )
    print(f"  {result}")
    assert result["optimal_fraction"] > 0, "最优比例应为正"
    assert result["recommended_position"] > 0, "建议金额应为正"
    assert result["half_kelly_position"] == result["recommended_position"] * 0.5, "半凯利应为一半"
    assert result["warning"] is None, f"不应有警告: {result['warning']}"
    print("  ✓ 凯利公式通过")


async def _test_grid_trading() -> None:
    print("\n▶ Test 2: 网格交易成本估算")
    result = grid_trading_cost_estimator(
        lower_bound=100.0,
        upper_bound=150.0,
        num_grids=5,
        total_capital=50_000,
        fee_rate_pct=0.1,
    )
    print(f"  grid_interval={result['grid_interval']}")
    assert len(result["price_levels"]) == 5, "应有 5 个价格级别"
    assert result["round_trip_fee_pct"] == 0.2, "往返手续费应为 0.2%"
    assert result["warning"] is None
    print("  ✓ 网格交易通过")


async def _test_portfolio_drawdown() -> None:
    print("\n▶ Test 3: 组合回撤估算")
    result = portfolio_drawdown_estimator(
        allocations={"tech_stocks": 0.6, "bonds": 0.3, "cash": 0.1},
        drawdown_scenarios={"tech_stocks": -30, "bonds": -5, "cash": 0},
        total_capital=200_000,
    )
    print(f"  weighted_drawdown={result['weighted_drawdown_pct']}%")
    expected_dd = 0.6 * -30 + 0.3 * -5 + 0.1 * 0  # -19.5
    assert result["weighted_drawdown_pct"] == expected_dd, f"期望 {expected_dd}"
    print("  ✓ 组合回撤通过")


async def _test_compound_growth() -> None:
    print("\n▶ Test 4: 复利增长计算")
    result = compound_growth_calculator(
        principal=100_000, annual_return_pct=8, years=10, monthly_contribution=2_000
    )
    print(f"  final_value={result['final_value']:,}")
    assert result["final_value"] > 100_000, "最终价值应大于本金"
    assert result["return_multiplier"] > 1.0
    print("  ✓ 复利增长通过")


async def _test_tool_registration() -> None:
    print("\n▶ Test 5: 工具注册")
    tools = list_tools()
    print(f"  已注册工具: {tools}")
    assert "kelly_criterion_calc" in tools
    assert "grid_trading_cost_estimator" in tools
    assert "portfolio_drawdown_estimator" in tools
    assert "compound_growth_calculator" in tools
    print("  ✓ 工具注册通过")


async def _test_execute_tool_directly() -> None:
    print("\n▶ Test 6: 直接执行工具")
    result = execute_tool("kelly_criterion_calc", {
        "win_rate": 0.6, "avg_gain_pct": 10, "avg_loss_pct": 5, "current_capital": 50_000
    })
    print(f"  {result}")
    assert isinstance(result, dict)
    assert result.get("optimal_fraction", 0) > 0
    print("  ✓ 直接执行工具通过")


async def _test_parse_tool_calls() -> None:
    print("\n▶ Test 7: TOOL_CALL 解析")
    text = (
        "建议进行凯利公式计算。"
        'TOOL_CALL: {"name": "kelly_criterion_calc", "arguments": {"win_rate": 0.55, "avg_gain_pct": 8, "avg_loss_pct": 4}}\n'
        "同时估算网格成本。"
        'TOOL_CALL: {"name": "grid_trading_cost_estimator", "arguments": {"lower_bound": 100, "upper_bound": 150, "num_grids": 5, "total_capital": 50000}}'
    )
    calls = _parse_tool_calls(text)
    print(f"  解析到 {len(calls)} 个调用")
    assert len(calls) == 2
    assert calls[0]["name"] == "kelly_criterion_calc"
    assert calls[1]["name"] == "grid_trading_cost_estimator"
    print("  ✓ TOOL_CALL 解析通过")


async def _test_execute_math_tool_node() -> None:
    print("\n▶ Test 8: execute_math_tool_node")
    state: WealthState = {
        "user_message": "test",
        "user_id": "test_p4_8",
        "sub_queries": [],
        "rewritten_queries": [],
        "retrieved_context": None,
        "retrieval_iterations": 0,
        "confidence": 0.0,
        "top_ce": 0.0,
        "evaluation_result": None,
        "needs_tool": False,
        "tool_calls": [
            {"name": "kelly_criterion_calc", "arguments": {"win_rate": 0.55, "avg_gain_pct": 8, "avg_loss_pct": 4, "current_capital": 100000}},
        ],
        "tool_results": [],
        "math_result": None,
        "reply": "",
        "history": [],
        "trace": [],
    }
    result = await execute_math_tool_node(state)
    math_result = result.get("math_result")
    tool_results = result.get("tool_results", [])
    print(f"  math_result keys: {list(math_result.keys()) if math_result else 'None'}")
    assert math_result is not None
    assert "kelly_criterion_calc" in math_result
    assert tool_results[0]["status"] == "success"
    print("  ✓ execute_math_tool_node 通过")


async def _test_merge_and_generate_node() -> None:
    print("\n▶ Test 9: merge_and_generate_node")
    state: WealthState = {
        "user_message": "test",
        "user_id": "test_p4_9",
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
        "math_result": {
            "kelly_criterion_calc": {
                "optimal_fraction": 0.25,
                "recommended_position": 25000,
                "warning": None,
            }
        },
        "reply": "根据历史回测，该策略胜率约 55%。",
        "history": [],
        "trace": [],
    }
    result = await merge_and_generate_node(state)
    reply = result.get("reply", "")
    print(f"  reply preview: {reply[:120]}...")
    assert "kelly_criterion_calc" in reply or "25000" in reply or "optimal_fraction" in reply
    print("  ✓ merge_and_generate_node 通过")


async def _test_after_strategy_conditional() -> None:
    print("\n▶ Test 10: after_strategy 条件边")
    s1: WealthState = {
        "user_message": "test", "user_id": "test_p4_10",
        "sub_queries": [], "rewritten_queries": [],
        "retrieved_context": None, "retrieval_iterations": 0,
        "confidence": 0.0, "top_ce": 0.0, "evaluation_result": None,
        "needs_tool": True, "tool_calls": [], "tool_results": [],
        "math_result": None, "reply": "", "history": [], "trace": [],
    }
    assert after_strategy(s1) == "tool"

    s2 = dict(s1)
    s2["needs_tool"] = False  # type: ignore[assignment]
    assert after_strategy(s2) == "final"
    print("  ✓ after_strategy 条件边通过")


async def main() -> None:
    _setup()
    await _test_kelly_criterion()
    await _test_grid_trading()
    await _test_portfolio_drawdown()
    await _test_compound_growth()
    await _test_tool_registration()
    await _test_execute_tool_directly()
    await _test_parse_tool_calls()
    await _test_execute_math_tool_node()
    await _test_merge_and_generate_node()
    await _test_after_strategy_conditional()
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" 阶段四全部测试通过 ✓")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    asyncio.run(main())
