"""
ai_app4 工具注册模块。

在 lifespan 中通过 `register_wealth_tools()` 一次性注册所有 Wealth AI 数学计算工具，
供 strategy_reasoning_node 的 LLM 识别与调用。
"""
from __future__ import annotations

from rag_framework.llm.tool_registry import register_tool

from ai_app4.service.math_tools import (
    kelly_criterion_calc,
    grid_trading_cost_estimator,
    portfolio_drawdown_estimator,
    compound_growth_calculator,
)


def register_wealth_tools() -> None:
    """注册 Wealth AI 专属数学与策略计算工具。"""
    register_tool(
        name="kelly_criterion_calc",
        func=kelly_criterion_calc,
        description=(
            "凯利公式最优仓位计算器。输入胜率、平均盈利百分比、平均亏损百分比和当前资金，"
            "返回最优投入比例、建议金额、半凯利保守仓位、期望收益率等。"
            "适用于基于历史回测数据计算下一次交易的最优仓位。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "win_rate": {
                    "type": "number",
                    "description": "胜率（0.0 ~ 1.0），例如 0.55 表示 55% 胜率",
                },
                "avg_gain_pct": {
                    "type": "number",
                    "description": "平均盈利百分比（正数），例如 8.0",
                },
                "avg_loss_pct": {
                    "type": "number",
                    "description": "平均亏损百分比（正数），例如 4.0",
                },
                "current_capital": {
                    "type": "number",
                    "description": "当前总资金，默认 100000",
                    "default": 100_000.0,
                },
            },
            "required": ["win_rate", "avg_gain_pct", "avg_loss_pct"],
        },
    )

    register_tool(
        name="grid_trading_cost_estimator",
        func=grid_trading_cost_estimator,
        description=(
            "网格交易成本与收益估算器。输入价格上下界、网格数量、总资金和手续费率，"
            "返回网格间距、各级价格、每格资金、往返手续费、盈亏平衡波动幅度、预估最大回撤等。"
            "适用于设计 ETF 或个股的网格交易策略。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "lower_bound": {
                    "type": "number",
                    "description": "网格下界价格（如 100.0）",
                },
                "upper_bound": {
                    "type": "number",
                    "description": "网格上界价格（如 150.0）",
                },
                "num_grids": {
                    "type": "integer",
                    "description": "网格数量（>=2）",
                },
                "total_capital": {
                    "type": "number",
                    "description": "总投入资金",
                },
                "fee_rate_pct": {
                    "type": "number",
                    "description": "单边手续费率（%，默认 0.1）",
                    "default": 0.1,
                },
            },
            "required": ["lower_bound", "upper_bound", "num_grids", "total_capital"],
        },
    )

    register_tool(
        name="portfolio_drawdown_estimator",
        func=portfolio_drawdown_estimator,
        description=(
            "组合最大回撤估算器。输入各资产权重和假设回撤场景，计算组合整体最大回撤和预估亏损金额。"
            "适用于压力测试和仓位分配决策。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "allocations": {
                    "type": "object",
                    "description": "资产名到权重的映射，如 {'tech_stocks': 0.6, 'bonds': 0.3, 'cash': 0.1}",
                },
                "drawdown_scenarios": {
                    "type": "object",
                    "description": "资产名到假设回撤幅度（%）的映射，如 {'tech_stocks': -30, 'bonds': -5}",
                },
                "total_capital": {
                    "type": "number",
                    "description": "总资金，默认 100000",
                    "default": 100_000.0,
                },
            },
            "required": ["allocations"],
        },
    )

    register_tool(
        name="compound_growth_calculator",
        func=compound_growth_calculator,
        description=(
            "复利增长计算器。输入初始本金、年化收益率、投资年限和每月定投金额，"
            "返回最终价值、总投入、总收益和收益倍数。"
            "适用于长期定投计划的收益估算。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "principal": {
                    "type": "number",
                    "description": "初始本金",
                },
                "annual_return_pct": {
                    "type": "number",
                    "description": "年化收益率（%）",
                },
                "years": {
                    "type": "integer",
                    "description": "投资年限",
                },
                "monthly_contribution": {
                    "type": "number",
                    "description": "每月定投金额，默认 0",
                    "default": 0.0,
                },
            },
            "required": ["principal", "annual_return_pct", "years"],
        },
    )
