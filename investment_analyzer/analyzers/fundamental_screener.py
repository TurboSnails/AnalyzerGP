"""
Layer 2: 基本面双轨筛选器

A类 (价值/红利): ROE + 负债率 + 股息率 + 毛利率 + 扣非利润增速
B类 (第二曲线): Rule of 40 + 毛利率趋势 + 营收加速 + 现金流改善

B类优先：成长型公司先判断B类，通过则推荐B类；否则判断A类。
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from config import LAYER2_CONFIG


class FundamentalScreener:

    def __init__(self):
        self.cfg = LAYER2_CONFIG

    def screen(
        self,
        financial_data: Dict,
        valuation: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        双轨筛选，B类优先。

        Returns:
            type_a / type_b: 各类筛选结果
            recommended_type: "A" | "B" | None
            passed: 是否通过任一类型
        """
        result = {
            "type_a": self._screen_type_a(financial_data, valuation),
            "type_b": self._screen_type_b(financial_data, valuation),
            "recommended_type": None,
            "passed": False,
            "summary": "",
        }

        if result["type_b"].get("passed"):
            result["recommended_type"] = "B"
            result["passed"] = True
        elif result["type_a"].get("passed"):
            result["recommended_type"] = "A"
            result["passed"] = True

        result["summary"] = self._build_summary(result)
        return result

    # ─── Type A ───────────────────────────────────────────────

    def _screen_type_a(self, financial_data: Dict, valuation: Optional[Dict]) -> Dict:
        cfg    = self.cfg["type_a"]
        ind    = financial_data.get("indicators", pd.DataFrame())
        bal    = financial_data.get("balance",    pd.DataFrame())
        checks = {}
        score  = 0

        # 1. ROE > 15%
        roe_raw = self._latest(ind, ["净资产收益率", "ROE", "roe"])
        if roe_raw is not None:
            roe = roe_raw / 100 if roe_raw > 2 else roe_raw
            ok  = roe >= cfg["min_roe"]
            checks["roe"] = self._check_item(
                "ROE", f"{roe:.1%}", f">{cfg['min_roe']:.0%}", ok
            )
            score += ok
        else:
            checks["roe"] = self._pending("ROE")

        # 2. 资产负债率 < 60%
        debt = self._debt_ratio(bal, ind)
        if debt is not None:
            ok = debt <= cfg["max_debt_ratio"]
            checks["debt_ratio"] = self._check_item(
                "资产负债率", f"{debt:.1%}", f"<{cfg['max_debt_ratio']:.0%}", ok
            )
            score += ok
        else:
            checks["debt_ratio"] = self._pending("资产负债率")

        # 3. 股息率 > 2%
        div = self._dividend_yield(valuation, ind)
        if div is not None:
            ok = div >= cfg["min_dividend_yield"]
            checks["dividend_yield"] = self._check_item(
                "股息率(TTM)", f"{div:.2%}", f">{cfg['min_dividend_yield']:.0%}", ok
            )
            score += ok
        else:
            checks["dividend_yield"] = self._pending("股息率")

        # 4. 毛利率 > 20%
        gm = self._latest_pct(ind, ["销售毛利率", "毛利率"])
        if gm is not None:
            ok = gm >= cfg["min_gross_margin"]
            checks["gross_margin"] = self._check_item(
                "毛利率", f"{gm:.1%}", f">{cfg['min_gross_margin']:.0%}", ok
            )
            score += ok
        else:
            checks["gross_margin"] = self._pending("毛利率")

        # 5. 扣非净利润增速 > 10%
        pg = self._latest_pct(
            ind, ["扣非净利润增速", "归母净利润增速", "净利润同比增长率", "净利润增速"]
        )
        if pg is not None:
            ok = pg >= cfg["min_profit_growth"]
            checks["profit_growth"] = self._check_item(
                "扣非净利润增速", f"{pg:.1%}", f">{cfg['min_profit_growth']:.0%}", ok
            )
            score += ok
        else:
            checks["profit_growth"] = self._pending("扣非净利润增速")

        threshold = cfg["pass_threshold"]
        return {
            "type":      "A",
            "label":     "价值/红利型",
            "passed":    score >= threshold,
            "score":     score,
            "total":     5,
            "threshold": threshold,
            "checks":    checks,
        }

    # ─── Type B ───────────────────────────────────────────────

    def _screen_type_b(self, financial_data: Dict, valuation: Optional[Dict]) -> Dict:
        cfg    = self.cfg["type_b"]
        ind    = financial_data.get("indicators", pd.DataFrame())
        checks = {}
        score  = 0

        # 1. Rule of 40 >= 40
        r40 = self._rule_of_40(ind)
        if r40 is not None:
            ok = r40 >= cfg["min_rule_of_40"]
            checks["rule_of_40"] = {
                **self._check_item(
                    "Rule of 40", f"{r40:.1f}", f">={cfg['min_rule_of_40']}", ok
                ),
                "detail": "营收增速(%) + 净利润率(%)",
            }
            score += ok
        else:
            checks["rule_of_40"] = self._pending("Rule of 40 (营收增速+净利润率)")

        # 2. 毛利率 > 40% 且趋势稳定/上升
        gm_check = self._gross_margin_trend(ind, cfg["min_gross_margin_tech"])
        checks["gross_margin_trend"] = gm_check
        score += gm_check.get("status") == "positive"

        # 3. 营收增速连续加速 3 季度
        accel_check = self._revenue_acceleration(ind, cfg["revenue_accel_quarters"])
        checks["revenue_acceleration"] = accel_check
        score += accel_check.get("status") == "positive"

        # 4. 经营现金流由负转正或增速 > 20%
        cf_check = self._cashflow_improvement(ind, financial_data.get("cashflow"))
        checks["cashflow"] = cf_check
        score += cf_check.get("status") == "positive"

        threshold = cfg["pass_threshold"]
        return {
            "type":      "B",
            "label":     "第二曲线成长型",
            "passed":    score >= threshold,
            "score":     score,
            "total":     4,
            "threshold": threshold,
            "checks":    checks,
        }

    # ─── Metric Helpers ───────────────────────────────────────

    def _latest(self, df: pd.DataFrame, cols: List[str]) -> Optional[float]:
        """从 DataFrame 按列名列表获取最新非空值"""
        if df is None or df.empty:
            return None
        for col in cols:
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    try:
                        return float(s.iloc[0])
                    except (ValueError, TypeError):
                        continue
        return None

    def _latest_pct(self, df: pd.DataFrame, cols: List[str]) -> Optional[float]:
        """获取百分比指标，自动处理 >1 的百分数格式"""
        val = self._latest(df, cols)
        if val is None:
            return None
        return val / 100 if abs(val) > 2 else val

    def _debt_ratio(self, balance: pd.DataFrame, indicators: pd.DataFrame) -> Optional[float]:
        # 先尝试从指标表直接取
        ratio = self._latest_pct(indicators, ["资产负债率"])
        if ratio is not None:
            return ratio

        if balance is None or balance.empty:
            return None

        # 尝试美股格式（yfinance）：列为日期，行为指标
        if "Total Assets" in balance.index and "Total Liabilities Net Minority Interest" in balance.index:
            try:
                assets = float(balance.loc["Total Assets"].iloc[0])
                liab   = float(balance.loc["Total Liabilities Net Minority Interest"].iloc[0])
                return liab / assets if assets > 0 else None
            except Exception:
                pass

        # A股格式：行为期数，列为指标
        assets = self._latest(balance, ["资产总计", "总资产"])
        liab   = self._latest(balance, ["负债合计", "总负债"])
        if assets and liab and assets > 0:
            return liab / assets
        return None

    def _dividend_yield(self, valuation: Optional[Dict], indicators: pd.DataFrame) -> Optional[float]:
        if valuation:
            for k in ("dv_ttm", "dividend_yield", "dividendYield"):
                v = valuation.get(k)
                if v is not None:
                    try:
                        fv = float(v)
                        return fv / 100 if fv > 1 else fv
                    except (ValueError, TypeError):
                        pass
        return self._latest_pct(indicators, ["股息率", "股息率(TTM)"])

    def _rule_of_40(self, indicators: pd.DataFrame) -> Optional[float]:
        rg = self._latest(indicators, ["营业总收入增速", "营收增速", "营业收入增速"])
        nm = self._latest(indicators, ["净利率", "净利润率", "销售净利率"])
        if rg is None or nm is None:
            return None
        rg_pct = rg if abs(rg) > 2 else rg * 100
        nm_pct = nm if abs(nm) > 2 else nm * 100
        return rg_pct + nm_pct

    def _gross_margin_trend(self, indicators: pd.DataFrame, threshold: float) -> Dict:
        gm_col = next(
            (c for c in ["销售毛利率", "毛利率"] if c in (indicators.columns if indicators is not None and not indicators.empty else [])),
            None,
        )
        if gm_col is None:
            return self._pending("毛利率趋势 (>40%)")

        series = indicators[gm_col].dropna().head(4)
        if series.empty:
            return self._pending("毛利率趋势 (>40%)")

        latest = float(series.iloc[0])
        gm     = latest / 100 if abs(latest) > 2 else latest

        if gm < threshold:
            return {
                "label":  "毛利率趋势",
                "status": "negative",
                "value":  f"{gm:.1%}",
                "detail": f"毛利率 {gm:.1%} 低于门槛 {threshold:.0%}",
            }

        # 趋势判断：最新值 >= 前期值（允许 2% 误差）
        vals = [float(v) for v in series.values[:3]]
        trending_up = len(vals) >= 2 and vals[0] >= vals[1] * 0.98
        trend = "上升" if trending_up else "稳定"

        return {
            "label":  "毛利率趋势",
            "status": "positive",
            "value":  f"{gm:.1%}",
            "trend":  trend,
            "detail": f"毛利率 {gm:.1%}，趋势{trend}",
        }

    def _revenue_acceleration(self, indicators: pd.DataFrame, quarters: int) -> Dict:
        rev_col = next(
            (
                c for c in ["营业总收入增速", "营收增速", "营业收入增速"]
                if c in (indicators.columns if indicators is not None and not indicators.empty else [])
            ),
            None,
        )
        if rev_col is None:
            return self._pending(f"营收加速度（需连续{quarters}季度）")

        series = indicators[rev_col].dropna().head(quarters + 1)
        if len(series) < quarters:
            return self._pending(f"营收加速度（数据不足{quarters}期）")

        vals = [float(v) for v in series.values[:quarters]]
        # 最新值最大 = 加速
        accelerating = all(vals[i] >= vals[i + 1] * 0.95 for i in range(len(vals) - 1))
        trend_str = " → ".join(reversed([f"{v:.1f}%" for v in vals]))

        return {
            "label":  "营收加速度",
            "status": "positive" if accelerating else "negative",
            "trend":  trend_str,
            "detail": f"连续{quarters}季度{'加速' if accelerating else '未加速'}: {trend_str}",
        }

    def _cashflow_improvement(self, indicators: pd.DataFrame, cashflow_df=None) -> Dict:
        # 尝试增速指标
        growth = self._latest_pct(
            indicators, ["经营现金流增速", "经营活动现金流净额增速"]
        )
        if growth is not None:
            ok = growth >= 0.20
            return {
                "label":  "经营现金流",
                "status": "positive" if ok else "negative",
                "value":  f"{growth:.1%}",
                "detail": f"经营现金流增速 {growth:.1%}",
            }

        # 尝试绝对值（A股）
        abs_val = self._latest(
            indicators, ["经营活动产生的现金流量净额", "经营现金流净额"]
        )
        if abs_val is not None:
            ok = abs_val > 0
            return {
                "label":  "经营现金流",
                "status": "positive" if ok else "negative",
                "detail": f"经营现金流净额: {abs_val:,.0f} ({'正' if ok else '负'})",
            }

        # 尝试美股 cashflow DataFrame
        if cashflow_df is not None and not (
            isinstance(cashflow_df, pd.DataFrame) and cashflow_df.empty
        ):
            try:
                cf = cashflow_df
                row = next(
                    (r for r in ["Operating Cash Flow", "Total Cash From Operating Activities"] if r in cf.index),
                    None,
                )
                if row:
                    val = float(cf.loc[row].iloc[0])
                    ok  = val > 0
                    return {
                        "label":  "经营现金流",
                        "status": "positive" if ok else "negative",
                        "detail": f"Operating CF: {val:,.0f} ({'正' if ok else '负'})",
                    }
            except Exception:
                pass

        return self._pending("经营现金流")

    # ─── Formatting Helpers ───────────────────────────────────

    @staticmethod
    def _check_item(label: str, value: str, threshold: str, passed: bool) -> Dict:
        return {
            "label":     label,
            "status":    "positive" if passed else "negative",
            "value":     value,
            "threshold": threshold,
            "detail":    f"{label}: {value} (阈值 {threshold})",
        }

    @staticmethod
    def _pending(label: str) -> Dict:
        return {"label": label, "status": "pending", "detail": f"{label}: 数据不足"}

    def _build_summary(self, result: Dict) -> str:
        rec = result.get("recommended_type")
        if rec == "B":
            b = result["type_b"]
            return f"B类成长型通过 ({b['score']}/{b['total']} 项满足)"
        if rec == "A":
            a = result["type_a"]
            return f"A类价值型通过 ({a['score']}/{a['total']} 项满足)"
        a, b = result["type_a"], result["type_b"]
        return (
            f"未通过基本面筛选 "
            f"(A类 {a.get('score', 0)}/{a.get('total', 5)}, "
            f"B类 {b.get('score', 0)}/{b.get('total', 4)})"
        )
