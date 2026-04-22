"""
Layer 0: 宏观定轨分析器
三合一信号: 宽基指数200MA + PE历史分位 + 10年期国债收益率方向

四种市场状态:
  bear_bottom    - 熊市底部，积累科技/宽基ETF
  recession_bear - 衰退型熊市，转红利ETF
  rate_hike_bear - 加息型熊市，转货币基金
  ambiguous      - 信号矛盾，维持现状
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import LAYER0_CONFIG


class MacroChecker:

    REGIME_LABELS = {
        "bear_bottom":    "熊市底部 → 定投科技/宽基ETF",
        "recession_bear": "衰退型熊市 → 转红利ETF",
        "rate_hike_bear": "加息型熊市 → 转货币基金/短债",
        "ambiguous":      "信号矛盾 → 维持现状，季末重新检查",
    }

    REGIME_ACTIONS = {
        "bear_bottom":    "定投科技/宽基ETF，积累筹码",
        "recession_bear": "逐步转仓至红利ETF（单次≤20%），维持现金流防御",
        "rate_hike_bear": "转仓至货币基金/短债ETF，等待利率见顶",
        "ambiguous":      "维持上一状态，季度末重新检查信号",
    }

    def __init__(self):
        self.cfg = LAYER0_CONFIG

    def analyze(
        self,
        index_data: Optional[pd.DataFrame] = None,
        pe_history: Optional[pd.DataFrame] = None,
        yield_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        result = {
            "regime": "ambiguous",
            "regime_label": "",
            "action": "",
            "signals": {},
            "layer1_allocation": {},
            "layer2_cap": 0.25,
            "summary": "",
        }

        result["signals"]["ma200"]       = self._check_200ma(index_data)
        result["signals"]["pe_percentile"] = self._check_pe_percentile(pe_history)
        result["signals"]["bond_yield"]  = self._check_yield_trend(yield_data)

        regime = self._determine_regime(
            result["signals"]["ma200"],
            result["signals"]["pe_percentile"],
            result["signals"]["bond_yield"],
        )
        result["regime"]       = regime
        result["regime_label"] = self.REGIME_LABELS[regime]
        result["action"]       = self.REGIME_ACTIONS[regime]
        result["layer1_allocation"] = self._get_layer1_allocation(regime)
        result["layer2_cap"]   = self._get_layer2_cap(regime)
        result["summary"]      = self._build_summary(result)
        return result

    # ─── Signal Checks ────────────────────────────────────────

    def _check_200ma(self, index_data: Optional[pd.DataFrame]) -> Dict:
        if index_data is None or index_data.empty:
            return {"status": "unknown", "detail": "指数数据不可用"}

        close_col = next(
            (c for c in ["收盘", "close", "Close"] if c in index_data.columns), None
        )
        if close_col is None or len(index_data) < self.cfg["ma_period"]:
            return {"status": "unknown", "detail": f"数据不足{self.cfg['ma_period']}日"}

        prices = index_data[close_col].astype(float).values
        ma200   = float(np.mean(prices[-self.cfg["ma_period"]:]))
        current = float(prices[-1])
        pct_diff = (current - ma200) / ma200

        return {
            "status":   "below" if current < ma200 else "above",
            "current":  round(current, 2),
            "ma200":    round(ma200, 2),
            "pct_diff": round(pct_diff, 4),
            "detail":   f"当前 {current:.0f} vs MA200 {ma200:.0f} ({pct_diff:+.1%})",
        }

    def _check_pe_percentile(self, pe_history: Optional[pd.DataFrame]) -> Dict:
        if pe_history is None or pe_history.empty:
            return {"status": "unknown", "detail": "PE历史数据不可用"}

        pe_col = next(
            (c for c in ["pe", "PE", "市盈率", "static_pe", "pe_ttm"] if c in pe_history.columns),
            None,
        )
        if pe_col is None:
            return {"status": "unknown", "detail": "未找到PE列"}

        series = pe_history[pe_col].dropna().astype(float)
        if series.empty:
            return {"status": "unknown", "detail": "PE数据为空"}

        current_pe  = float(series.iloc[-1])
        percentile  = float((series < current_pe).sum() / len(series))

        if percentile <= self.cfg["pe_bear_percentile"]:
            status = "low"
        elif percentile >= self.cfg["pe_bull_percentile"]:
            status = "high"
        else:
            status = "neutral"

        return {
            "status":      status,
            "current_pe":  round(current_pe, 2),
            "percentile":  round(percentile, 4),
            "detail":      f"PE={current_pe:.1f}, 历史分位={percentile:.1%}",
        }

    def _check_yield_trend(self, yield_data: Optional[pd.DataFrame]) -> Dict:
        if yield_data is None or yield_data.empty:
            return {"status": "unknown", "detail": "国债收益率数据不可用"}

        yield_col = next(
            (
                c for c in yield_data.columns
                if any(k in c for k in ["10年", "10y", "10Y", "yield", "收益率"])
            ),
            yield_data.columns[-1] if len(yield_data.columns) else None,
        )
        if yield_col is None:
            return {"status": "unknown", "detail": "未找到收益率列"}

        series   = yield_data[yield_col].dropna().astype(float)
        lookback = min(self.cfg["yield_trend_days"], len(series) - 1)
        if lookback < 10:
            return {"status": "unknown", "detail": "数据期数不足"}

        recent = float(series.iloc[-1])
        past   = float(series.iloc[-lookback])
        change = recent - past
        threshold = self.cfg["yield_threshold_bp"]

        if change > threshold:
            status = "rising"
        elif change < -threshold:
            status = "falling"
        else:
            status = "flat"

        return {
            "status":    status,
            "current":   round(recent, 4),
            "change_60d": round(change, 4),
            "detail":    f"10Y国债={recent:.2f}%, {lookback}日变化={change:+.2f}%",
        }

    # ─── Regime Logic ─────────────────────────────────────────

    def _determine_regime(self, ma_sig: Dict, pe_sig: Dict, yield_sig: Dict) -> str:
        ma     = ma_sig.get("status", "unknown")
        pe     = pe_sig.get("status", "unknown")
        y      = yield_sig.get("status", "unknown")

        # 熊市底部: 指数<200MA + PE低分位（真便宜）
        if ma == "below" and pe == "low":
            return "bear_bottom"

        # 加息型熊市: 指数<200MA + 收益率快速上行
        if ma == "below" and y == "rising":
            return "rate_hike_bear"

        # 衰退型熊市: 指数<200MA + 收益率下行或平稳
        if ma == "below" and y in ("falling", "flat"):
            return "recession_bear"

        # 信号矛盾（含牛市中）
        return "ambiguous"

    def _get_layer1_allocation(self, regime: str) -> Dict:
        alloc = {
            "bear_bottom":    {"科技/宽基ETF": "65%", "红利ETF": "0%",  "货币基金": "0%"},
            "recession_bear": {"科技/宽基ETF": "0%",  "红利ETF": "65%", "货币基金": "0%"},
            "rate_hike_bear": {"科技/宽基ETF": "0%",  "红利ETF": "0%",  "货币基金": "65%"},
            "ambiguous":      {"维持现状": "不操作，季末复查"},
        }
        return alloc.get(regime, {})

    def _get_layer2_cap(self, regime: str) -> float:
        """熊市早期（PE仍高）压缩Alpha上限"""
        if regime in ("rate_hike_bear", "recession_bear"):
            return 0.12
        return 0.25

    def _build_summary(self, result: Dict) -> str:
        parts = [v.get("detail", "") for v in result["signals"].values() if v.get("detail")]
        return f"{result['regime_label']} | {' | '.join(parts)}"
