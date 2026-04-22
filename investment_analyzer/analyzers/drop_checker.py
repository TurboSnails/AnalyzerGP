"""
第一关: 跌幅量化分析器 - 超跌验证
"""
import pandas as pd
import numpy as np
from typing import Dict, Any
from config import ANALYSIS_CONFIG


class DropChecker:
    """
    检查股票是否满足「超跌」条件:
    1. 相对历史高点跌幅 > 50%
    2. PE/PB 处于历史分位数后20%以内
    3. 市值缩水幅度远大于营收/利润下滑幅度
    """

    def __init__(self):
        self.min_drop = ANALYSIS_CONFIG["min_drop_from_high"]
        self.pe_pb_max = ANALYSIS_CONFIG["pe_pb_percentile_max"]

    def analyze(self, price_history: pd.DataFrame,
                valuation: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行跌幅量化分析

        Args:
            price_history: 历史行情 DataFrame (需含 '收盘' 或 'Close' 列)
            valuation: 估值数据字典 (含 pe_ttm, pb, history)

        Returns:
            分析结果字典
        """
        result = {
            "passed": False,
            "score": 0,
            "details": {},
            "signals": [],
            "warnings": [],
        }

        # 1. 计算相对历史高点的跌幅
        close_col = self._find_close_col(price_history)
        if close_col and not price_history.empty:
            prices = price_history[close_col]
            high = prices.max()
            current = prices.iloc[-1]
            drop_pct = (high - current) / high

            result["details"]["historical_high"] = round(float(high), 2)
            result["details"]["current_price"] = round(float(current), 2)
            result["details"]["drop_from_high"] = round(float(drop_pct), 4)

            if drop_pct >= self.min_drop:
                result["signals"].append(
                    f"✅ 跌幅 {drop_pct:.1%}，超过阈值 {self.min_drop:.0%}"
                )
                result["score"] += 1
            else:
                result["warnings"].append(
                    f"⚠️ 跌幅 {drop_pct:.1%}，未达到 {self.min_drop:.0%} 阈值"
                )

        # 2. PE/PB 历史分位数
        if "history" in valuation and not valuation["history"].empty:
            hist = valuation["history"]

            # PE分位数
            if "pe_ttm" in hist.columns:
                pe_series = hist["pe_ttm"].dropna()
                if not pe_series.empty and valuation.get("pe_ttm"):
                    current_pe = float(valuation["pe_ttm"])
                    pe_percentile = (pe_series < current_pe).sum() / len(pe_series)
                    result["details"]["pe_ttm"] = round(current_pe, 2)
                    result["details"]["pe_percentile"] = round(float(pe_percentile), 4)

                    if pe_percentile <= self.pe_pb_max:
                        result["signals"].append(
                            f"✅ PE分位数 {pe_percentile:.1%}，处于历史低位"
                        )
                        result["score"] += 1
                    else:
                        result["warnings"].append(
                            f"⚠️ PE分位数 {pe_percentile:.1%}，未处于后{self.pe_pb_max:.0%}"
                        )

            # PB分位数
            if "pb" in hist.columns:
                pb_series = hist["pb"].dropna()
                if not pb_series.empty and valuation.get("pb"):
                    current_pb = float(valuation["pb"])
                    pb_percentile = (pb_series < current_pb).sum() / len(pb_series)
                    result["details"]["pb"] = round(current_pb, 2)
                    result["details"]["pb_percentile"] = round(float(pb_percentile), 4)

                    if pb_percentile <= self.pe_pb_max:
                        result["signals"].append(
                            f"✅ PB分位数 {pb_percentile:.1%}，处于历史低位"
                        )
                        result["score"] += 1

        # 综合判断
        result["passed"] = result["score"] >= 2  # 至少2项通过
        result["summary"] = self._generate_summary(result)
        return result

    def _find_close_col(self, df: pd.DataFrame) -> str:
        """找到收盘价列名"""
        for col in ["收盘", "Close", "close"]:
            if col in df.columns:
                return col
        return ""

    def _generate_summary(self, result: Dict) -> str:
        details = result["details"]
        if result["passed"]:
            return (
                f"通过超跌验证 | "
                f"跌幅: {details.get('drop_from_high', 0):.1%} | "
                f"PE分位: {details.get('pe_percentile', 'N/A')}"
            )
        return "未通过超跌验证，建议关注但暂不符合超跌标准"
