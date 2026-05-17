"""
ai_app4 数学与策略计算工具箱。

纯 Python 函数，无外部依赖，可被 LLM tool calling 直接调用。
所有函数返回结构化 dict，便于 merge_and_generate_node 解析。
"""
from __future__ import annotations

import math
from typing import Any


def kelly_criterion_calc(
    win_rate: float,
    avg_gain_pct: float,
    avg_loss_pct: float,
    current_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    凯利公式最优仓位计算器。

    公式：f* = (p * b - q) / b
      - p = 胜率（win_rate）
      - q = 1 - p
      - b = 盈亏比 = avg_gain_pct / avg_loss_pct

    Args:
        win_rate: 胜率（0.0 ~ 1.0），例如 0.55
        avg_gain_pct: 平均盈利百分比（正数），例如 8.0
        avg_loss_pct: 平均亏损百分比（正数），例如 4.0
        current_capital: 当前总资金（默认 10万）

    Returns:
        {
            "optimal_fraction": float,      # 凯利最优资金比例（如 0.25）
            "recommended_position": float,  # 建议投入金额
            "half_kelly_position": float,   # 半凯利保守仓位金额
            "quarter_kelly_position": float,# 四分之一凯利安全仓位金额
            "expected_return": float,       # 单次期望收益率（%）
            "risk_unit": float,             # 每单位资金承担的风险
            "warning": str | None,          # 若 win_rate 或盈亏比不合理则给出警告
        }
    """
    result: dict[str, Any] = {
        "optimal_fraction": 0.0,
        "recommended_position": 0.0,
        "half_kelly_position": 0.0,
        "quarter_kelly_position": 0.0,
        "expected_return": 0.0,
        "risk_unit": 0.0,
        "warning": None,
    }

    # 校验
    if not (0 < win_rate < 1):
        result["warning"] = f"胜率应在 (0,1) 区间，当前 {win_rate}"
        return result
    if avg_gain_pct <= 0 or avg_loss_pct <= 0:
        result["warning"] = "平均盈利/亏损百分比必须为正数"
        return result

    q = 1.0 - win_rate
    b = avg_gain_pct / avg_loss_pct  # 盈亏比

    if b == 0:
        result["warning"] = "盈亏比为 0，无法计算"
        return result

    kelly_f = (win_rate * b - q) / b

    if kelly_f <= 0:
        result["warning"] = (
            f"凯利最优比例为 {kelly_f:.2%}，该策略期望收益为负或接近零，"
            "建议不持仓或重新评估策略"
        )
        result["optimal_fraction"] = kelly_f
        return result

    # 限制最大仓位不超过 100%（防止极端参数）
    kelly_f = min(kelly_f, 1.0)

    expected_return = win_rate * avg_gain_pct - q * avg_loss_pct

    result["optimal_fraction"] = round(kelly_f, 4)
    result["recommended_position"] = round(current_capital * kelly_f, 2)
    result["half_kelly_position"] = round(current_capital * kelly_f * 0.5, 2)
    result["quarter_kelly_position"] = round(current_capital * kelly_f * 0.25, 2)
    result["expected_return"] = round(expected_return, 2)
    result["risk_unit"] = round(avg_loss_pct * kelly_f, 2)

    return result


def grid_trading_cost_estimator(
    lower_bound: float,
    upper_bound: float,
    num_grids: int,
    total_capital: float,
    fee_rate_pct: float = 0.1,
    price_format: str = "decimal",
) -> dict[str, Any]:
    """
    网格交易成本与收益估算器。

    等差网格：在 [lower_bound, upper_bound] 之间均匀放置 num_grids 条网格线。

    Args:
        lower_bound: 网格下界价格
        upper_bound: 网格上界价格
        num_grids: 网格数量（>=2）
        total_capital: 总投入资金
        fee_rate_pct: 单边手续费率（%，默认 0.1%）
        price_format: 价格显示格式，"decimal" | "percent"

    Returns:
        {
            "grid_interval": float,          # 每条网格间距
            "price_levels": list[float],     # 各级价格
            "capital_per_grid": float,       # 每格分配资金
            "round_trip_fee_pct": float,     # 往返一次手续费（%）
            "fee_drag_per_cycle": float,     # 每轮完整穿越网格的手续费损耗金额
            "breakeven_move_pct": float,     # 覆盖手续费需要的最小价格波动（%）
            "max_drawdown_estimate": float,  # 假设跌穿下界时的预估最大回撤金额
            "warning": str | None,
        }
    """
    result: dict[str, Any] = {
        "grid_interval": 0.0,
        "price_levels": [],
        "capital_per_grid": 0.0,
        "round_trip_fee_pct": 0.0,
        "fee_drag_per_cycle": 0.0,
        "breakeven_move_pct": 0.0,
        "max_drawdown_estimate": 0.0,
        "warning": None,
    }

    if num_grids < 2:
        result["warning"] = "网格数量必须 >= 2"
        return result
    if lower_bound >= upper_bound or lower_bound <= 0 or total_capital <= 0:
        result["warning"] = "价格区间或资金参数不合法"
        return result

    interval = (upper_bound - lower_bound) / (num_grids - 1)
    levels = [round(lower_bound + i * interval, 4) for i in range(num_grids)]
    capital_per_grid = total_capital / num_grids

    round_trip_fee = fee_rate_pct * 2  # 买入 + 卖出
    fee_drag = total_capital * (round_trip_fee / 100)

    # 覆盖手续费需要的最小价格波动（以中间价为基准）
    mid_price = (lower_bound + upper_bound) / 2
    breakeven_move = (round_trip_fee / 100) * mid_price
    breakeven_move_pct = (breakeven_move / mid_price) * 100 if mid_price > 0 else 0

    # 若价格跌穿下界，假设全部持仓按 lower_bound 清仓的回撤估算
    max_drawdown = total_capital * (1 - lower_bound / mid_price) if mid_price > 0 else 0

    result["grid_interval"] = round(interval, 4)
    result["price_levels"] = levels
    result["capital_per_grid"] = round(capital_per_grid, 2)
    result["round_trip_fee_pct"] = round(round_trip_fee, 3)
    result["fee_drag_per_cycle"] = round(fee_drag, 2)
    result["breakeven_move_pct"] = round(breakeven_move_pct, 4)
    result["max_drawdown_estimate"] = round(max_drawdown, 2)

    return result


def portfolio_drawdown_estimator(
    allocations: dict[str, float],
    drawdown_scenarios: dict[str, float] | None = None,
    total_capital: float = 100_000.0,
) -> dict[str, Any]:
    """
    组合最大回撤估算器。

    给定各资产权重和假设回撤幅度，计算组合整体回撤。

    Args:
        allocations: 资产名 → 权重（0.0~1.0，总和建议 = 1.0）
        drawdown_scenarios: 资产名 → 假设回撤幅度（%，负数），默认科技股 -30%、债券 -5%、现金 0%
        total_capital: 总资金

    Returns:
        {
            "weighted_drawdown_pct": float,  # 组合加权回撤百分比
            "estimated_loss": float,         # 预估亏损金额
            "residual_value": float,         # 剩余资金
            "scenario_details": list[dict],  # 每项资产明细
        }
    """
    if drawdown_scenarios is None:
        drawdown_scenarios = {
            "tech_stocks": -30.0,
            "bonds": -5.0,
            "cash": 0.0,
        }

    details: list[dict] = []
    weighted_dd = 0.0

    for asset, weight in allocations.items():
        dd = drawdown_scenarios.get(asset, -10.0)
        contribution = weight * dd
        weighted_dd += contribution
        details.append({
            "asset": asset,
            "weight": round(weight, 4),
            "assumed_drawdown_pct": dd,
            "contribution_pct": round(contribution, 2),
        })

    estimated_loss = total_capital * (abs(weighted_dd) / 100)
    residual = total_capital - estimated_loss

    return {
        "weighted_drawdown_pct": round(weighted_dd, 2),
        "estimated_loss": round(estimated_loss, 2),
        "residual_value": round(residual, 2),
        "scenario_details": details,
    }


def compound_growth_calculator(
    principal: float,
    annual_return_pct: float,
    years: int,
    monthly_contribution: float = 0.0,
) -> dict[str, Any]:
    """
    复利增长计算器。

    Args:
        principal: 初始本金
        annual_return_pct: 年化收益率（%）
        years: 投资年限
        monthly_contribution: 每月定投金额

    Returns:
        {
            "final_value": float,
            "total_contributed": float,
            "total_return": float,
            "return_multiplier": float,
        }
    """
    r = annual_return_pct / 100
    n = years

    # 本金复利
    final_principal = principal * ((1 + r) ** n)

    # 定投复利（月末投入）
    final_contrib = 0.0
    if monthly_contribution > 0 and r > 0:
        monthly_rate = r / 12
        months = n * 12
        final_contrib = monthly_contribution * (
            ((1 + monthly_rate) ** months - 1) / monthly_rate
        )
    elif monthly_contribution > 0:
        final_contrib = monthly_contribution * n * 12

    total = final_principal + final_contrib
    contributed = principal + monthly_contribution * n * 12

    return {
        "final_value": round(total, 2),
        "total_contributed": round(contributed, 2),
        "total_return": round(total - contributed, 2),
        "return_multiplier": round(total / contributed, 2) if contributed > 0 else 0.0,
    }
