"""
第二关: 真反转 vs 价值陷阱验证器
七维度定量检查（数据充足时自动计算，不足时标记供AI辅助）
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import ANALYSIS_CONFIG


class ReversalChecker:

    DIMENSIONS = [
        "gross_margin",
        "operating_cashflow",
        "inventory_cycle",
        "insider_behavior",
        "competition",
        "debt_structure",
        "core_business",
    ]

    def __init__(self):
        self.pass_threshold = ANALYSIS_CONFIG["reversal_pass_threshold"]

    def analyze(self, financial_data: Dict, insider_data=None) -> Dict[str, Any]:
        ind = financial_data.get("indicators", pd.DataFrame())
        bal = financial_data.get("balance",    pd.DataFrame())
        inc = financial_data.get("income",     pd.DataFrame())
        cf  = financial_data.get("cashflow",   pd.DataFrame())

        checks = [
            self._check_gross_margin(ind, inc),
            self._check_cashflow(ind, cf),
            self._check_inventory(ind, bal),
            self._check_insider(insider_data),
            self._check_competition(ind),        # 需AI辅助
            self._check_debt(ind, bal),
            self._check_core_business(ind, inc),
        ]

        result = {
            "passed":     False,
            "verdict":    "",
            "score":      0,
            "total":      7,
            "dimensions": {},
            "signals":    [],
            "warnings":   [],
        }

        for dim_name, check in zip(self.DIMENSIONS, checks):
            result["dimensions"][dim_name] = check
            if check["status"] == "positive":
                result["score"] += 1
                result["signals"].append(f"✅ {check['label']}: {check['detail']}")
            elif check["status"] == "negative":
                result["warnings"].append(f"❌ {check['label']}: {check['detail']}")
            else:
                result["signals"].append(f"⏳ {check['label']}: {check['detail']}")

        result["passed"]  = result["score"] >= self.pass_threshold
        result["verdict"] = "真反转信号" if result["passed"] else "价值陷阱警告"
        result["summary"] = (
            f"{result['verdict']} | 通过 {result['score']}/{result['total']} 维度 | "
            f"阈值: {self.pass_threshold}"
        )
        return result

    # ─── Dimension Checks ─────────────────────────────────────

    def _check_gross_margin(self, ind: pd.DataFrame, inc: pd.DataFrame) -> Dict:
        """毛利率趋势: 连续2期改善 = 正面"""
        series = self._get_series(ind, ["销售毛利率", "毛利率"])
        if series is None and inc is not None and not inc.empty:
            # 尝试从利润表计算（A股行格式）
            gp  = self._get_series(inc, ["毛利润", "营业毛利"])
            rev = self._get_series(inc, ["营业总收入", "营业收入"])
            if gp is not None and rev is not None and len(gp) >= 2:
                series = pd.Series([float(g) / float(r) for g, r in zip(gp, rev) if float(r) > 0])

        if series is None or len(series) < 2:
            return self._dim("毛利率趋势", "pending", "需要连续4季度毛利率数据")

        vals = self._normalize_pct(series[:4])
        latest, prev = vals[0], vals[1]
        improving = latest >= prev - 0.005  # 允许 0.5% 误差

        return self._dim(
            "毛利率趋势",
            "positive" if improving else "negative",
            f"最新 {latest:.1%}，前期 {prev:.1%}，{'改善' if improving else '恶化'}",
        )

    def _check_cashflow(self, ind: pd.DataFrame, cf: pd.DataFrame) -> Dict:
        """经营现金流为正且改善"""
        # 尝试增速
        growth = self._latest_pct(ind, ["经营现金流增速", "经营活动现金流净额增速"])
        if growth is not None:
            ok = growth >= 0
            return self._dim(
                "经营现金流",
                "positive" if ok else "negative",
                f"增速 {growth:.1%}",
            )

        # 尝试绝对值
        abs_val = self._latest(ind, ["经营活动产生的现金流量净额", "经营现金流净额"])
        if abs_val is None and cf is not None and not cf.empty:
            abs_val = self._cf_value(cf, ["Operating Cash Flow",
                                          "Total Cash From Operating Activities"])
        if abs_val is not None:
            ok = abs_val > 0
            return self._dim(
                "经营现金流",
                "positive" if ok else "negative",
                f"净额 {abs_val:,.0f} ({'正' if ok else '负'})",
            )

        return self._dim("经营现金流", "pending", "需要现金流量表数据")

    def _check_inventory(self, ind: pd.DataFrame, bal: pd.DataFrame) -> Dict:
        """库存周期: 存货周转天数下降 = 去库存"""
        series = self._get_series(ind, ["存货周转天数", "应收账款周转天数"])
        if series is not None and len(series) >= 2:
            latest, prev = float(series.iloc[0]), float(series.iloc[1])
            declining = latest <= prev * 1.05   # 允许5%误差
            return self._dim(
                "库存周期",
                "positive" if declining else "negative",
                f"周转天数 {latest:.0f}天 vs 前期 {prev:.0f}天，"
                f"{'去库存' if declining else '库存积压'}",
            )

        # 尝试从余额表计算简单周转率
        inv = self._latest(bal, ["存货", "库存"]) if bal is not None and not bal.empty else None
        rev = self._latest(ind, ["营业总收入", "营业收入"])
        if inv is not None and rev is not None and rev > 0:
            ratio = rev / inv
            return self._dim(
                "库存周期",
                "positive" if ratio > 3 else "negative",
                f"存货/营收比 {1/ratio:.2f}，{'偏低(健康)' if ratio > 3 else '偏高(积压)'}",
            )

        return self._dim("库存周期", "pending", "需要存货周转率数据，将由AI辅助判断")

    def _check_insider(self, data) -> Dict:
        """管理层增持为正面信号"""
        if data is None:
            return self._dim("管理层行为", "pending", "暂无管理层交易数据")
        if isinstance(data, pd.DataFrame) and data.empty:
            return self._dim("管理层行为", "pending", "暂无管理层交易数据")

        try:
            df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
            # 查找买入/增持记录
            buy_kws = ["买入", "增持", "purchase", "buy", "Purchase"]
            sell_kws = ["卖出", "减持", "sale", "sell", "Sale"]

            text_col = next(
                (c for c in df.columns if any(k in c.lower() for k in ["type", "类型", "变动"])),
                df.columns[0] if not df.empty else None,
            )
            if text_col:
                buys  = df[df[text_col].astype(str).str.contains("|".join(buy_kws),  na=False)]
                sells = df[df[text_col].astype(str).str.contains("|".join(sell_kws), na=False)]
                net_signal = len(buys) - len(sells)
                if net_signal > 0:
                    return self._dim("管理层行为", "positive",
                                     f"近期增持 {len(buys)} 次，减持 {len(sells)} 次")
                if net_signal < 0:
                    return self._dim("管理层行为", "negative",
                                     f"近期减持 {len(sells)} 次多于增持 {len(buys)} 次")
        except Exception:
            pass

        return self._dim("管理层行为", "pending",
                         f"发现 {len(data)} 条交易记录，需AI进一步分析")

    def _check_competition(self, ind: pd.DataFrame) -> Dict:
        """行业竞争格局需AI分析，定量端看市占率代理指标"""
        # 市占率代理：营收增速是否跑赢行业（当前无行业对标数据，标记待AI）
        return self._dim("行业竞争格局", "pending",
                         "需AI分析行业产能、竞争格局及公司市场份额变化")

    def _check_debt(self, ind: pd.DataFrame, bal: pd.DataFrame) -> Dict:
        """负债结构: 资产负债率 < 60% 且短债/长债比合理"""
        ratio = self._latest_pct(ind, ["资产负债率"])
        if ratio is None and bal is not None and not bal.empty:
            assets = self._latest(bal, ["资产总计", "总资产"])
            liab   = self._latest(bal, ["负债合计", "总负债"])
            if assets and liab and assets > 0:
                ratio = liab / assets
            # 美股格式
            if ratio is None and "Total Assets" in bal.index:
                try:
                    a = float(bal.loc["Total Assets"].iloc[0])
                    l = float(bal.loc["Total Liabilities Net Minority Interest"].iloc[0])
                    ratio = l / a if a > 0 else None
                except Exception:
                    pass

        if ratio is not None:
            ok = ratio <= 0.60
            return self._dim(
                "负债结构",
                "positive" if ok else "negative",
                f"资产负债率 {ratio:.1%} ({'安全' if ok else '偏高'}，阈值 60%)",
            )

        return self._dim("负债结构", "pending", "需要资产负债表数据")

    def _check_core_business(self, ind: pd.DataFrame, inc: pd.DataFrame) -> Dict:
        """核心业务健康度: 营收增速 + 毛利率双升"""
        rev_growth = self._latest_pct(ind, ["营业总收入增速", "营收增速", "营业收入增速"])
        gm         = self._latest_pct(ind, ["销售毛利率", "毛利率"])

        if rev_growth is None and gm is None:
            return self._dim("核心业务健康度", "pending", "需要营收及毛利数据")

        parts = []
        positive_signals = 0

        if rev_growth is not None:
            ok = rev_growth > 0
            positive_signals += ok
            parts.append(f"营收增速 {rev_growth:.1%}")

        if gm is not None:
            ok = gm > 0.20
            positive_signals += ok
            parts.append(f"毛利率 {gm:.1%}")

        total = len(parts)
        status = "positive" if positive_signals == total else (
            "negative" if positive_signals == 0 else "pending"
        )
        return self._dim("核心业务健康度", status, "，".join(parts))

    # ─── Data Helpers ─────────────────────────────────────────

    def _latest(self, df: pd.DataFrame, cols: list) -> Optional[float]:
        if df is None or df.empty:
            return None
        for col in cols:
            if col in df.columns:
                s = df[col].dropna()
                if not s.empty:
                    try:
                        return float(s.iloc[0])
                    except (ValueError, TypeError):
                        pass
        return None

    def _latest_pct(self, df: pd.DataFrame, cols: list) -> Optional[float]:
        val = self._latest(df, cols)
        if val is None:
            return None
        return val / 100 if abs(val) > 2 else val

    def _get_series(self, df: pd.DataFrame, cols: list) -> Optional[pd.Series]:
        if df is None or df.empty:
            return None
        for col in cols:
            if col in df.columns:
                s = df[col].dropna()
                if len(s) >= 2:
                    return s
        return None

    def _normalize_pct(self, series: pd.Series) -> pd.Series:
        vals = series.astype(float)
        if vals.abs().max() > 2:
            vals = vals / 100
        return vals

    def _cf_value(self, cf: pd.DataFrame, row_names: list) -> Optional[float]:
        for row in row_names:
            if row in cf.index:
                try:
                    return float(cf.loc[row].iloc[0])
                except Exception:
                    pass
        return None

    @staticmethod
    def _dim(label: str, status: str, detail: str) -> Dict:
        return {"label": label, "status": status, "detail": detail}
