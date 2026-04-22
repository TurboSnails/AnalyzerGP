"""
历史回测模块 - 验证超跌买入信号在历史数据中的表现
策略: 当股价较滚动高点跌幅超过阈值时记录买入点，统计后续 6/12/18 个月收益
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class Backtester:
    """
    基于「超跌买入」信号的历史回测器

    回测逻辑:
    1. 以滚动窗口内高点为基准计算实时跌幅
    2. 跌幅首次触及阈值时记录入场点
    3. 每个入场点设 60 日冷静期，防止同一下跌趋势重复计数
    4. 统计各持有期（6/12/18 个月）的胜率与收益分布
    """

    def __init__(
        self,
        drop_threshold: float = 0.50,
        lookback_days: int = 504,   # ~2 年
        cooldown_days: int = 60,
    ):
        self.drop_threshold = drop_threshold
        self.lookback_days  = lookback_days
        self.cooldown_days  = cooldown_days

    # ── 主入口 ────────────────────────────────────────────────

    def run(self, price_history: pd.DataFrame) -> Dict[str, Any]:
        df = self._normalize(price_history)
        if df.empty or len(df) < 120:
            return {
                "error": "数据不足（至少需要 120 个交易日）",
                "signal_count": 0,
                "stats": {},
                "entries": [],
                "summary": "数据不足，无法回测",
            }

        df = df.sort_values("date").reset_index(drop=True)
        df["high_roll"] = df["close"].rolling(self.lookback_days, min_periods=60).max()
        df["drop_pct"]  = df["close"] / df["high_roll"] - 1

        entry_indices = self._find_entries(df)

        if not entry_indices:
            return {
                "signal_count": 0,
                "stats": {},
                "entries": [],
                "summary": f"回测期间未触发超跌信号（阈值: 跌幅 >{self.drop_threshold:.0%}）",
            }

        records = self._build_records(df, entry_indices)
        stats   = self._compute_stats(records)
        summary = self._build_summary(entry_indices, stats)

        return {
            "signal_count": len(entry_indices),
            "stats": stats,
            "entries": records,
            "summary": summary,
        }

    # ── 入场点检测 ────────────────────────────────────────────

    def _find_entries(self, df: pd.DataFrame) -> List[int]:
        entries   = []
        cooldown  = 0
        triggered = False

        for i, row in df.iterrows():
            if cooldown > 0:
                cooldown -= 1
                continue

            drop = row["drop_pct"]
            if pd.isna(drop):
                continue

            if not triggered and drop <= -self.drop_threshold:
                entries.append(i)
                triggered = True
                cooldown  = self.cooldown_days
            elif triggered and drop > -self.drop_threshold * 0.7:
                # 价格回升超过阈值 70% 时重置，允许再次触发
                triggered = False

        return entries

    # ── 收益计算 ──────────────────────────────────────────────

    def _build_records(self, df: pd.DataFrame, entry_indices: List[int]) -> List[Dict]:
        records = []
        horizons = [(6, "6m"), (12, "12m"), (18, "18m")]

        for idx in entry_indices:
            entry_row   = df.loc[idx]
            entry_date  = entry_row["date"]
            entry_price = entry_row["close"]
            drop_at_entry = entry_row["drop_pct"]

            row = {"entry_date": entry_date.strftime("%Y-%m-%d"),
                   "entry_price": round(float(entry_price), 2),
                   "drop_at_entry": f"{drop_at_entry*100:.1f}%"}

            for months, label in horizons:
                target_dt = entry_date + pd.Timedelta(days=int(months * 30.5))
                future = df[df["date"] >= target_dt]
                if not future.empty:
                    exit_price = float(future.iloc[0]["close"])
                    ret = (exit_price / entry_price - 1) * 100
                    row[f"return_{label}"]     = round(ret, 1)
                    row[f"exit_price_{label}"] = round(exit_price, 2)
                else:
                    row[f"return_{label}"]     = None
                    row[f"exit_price_{label}"] = None

            # Max drawdown after entry (18-month window)
            end_dt = entry_date + pd.Timedelta(days=548)
            window = df[(df["date"] > entry_date) & (df["date"] <= end_dt)]["close"]
            if not window.empty:
                peak = entry_price
                mdd  = 0.0
                for p in window.values:
                    peak = max(peak, p)
                    mdd  = min(mdd, (p - peak) / peak * 100)
                row["max_drawdown_after"] = round(mdd, 1)
            else:
                row["max_drawdown_after"] = None

            records.append(row)

        return records

    # ── 统计汇总 ──────────────────────────────────────────────

    def _compute_stats(self, records: List[Dict]) -> Dict[str, Any]:
        stats = {}
        for label in ["6m", "12m", "18m"]:
            key  = f"return_{label}"
            vals = [r[key] for r in records if r.get(key) is not None]
            if not vals:
                continue
            arr = np.array(vals)
            stats[label] = {
                "count":          len(arr),
                "win_rate":       f"{(arr > 0).mean() * 100:.0f}%",
                "avg_return":     f"{arr.mean():.1f}%",
                "median_return":  f"{np.median(arr):.1f}%",
                "best":           f"{arr.max():.1f}%",
                "worst":          f"{arr.min():.1f}%",
                "std":            f"{arr.std():.1f}%",
            }

        mdd_vals = [r["max_drawdown_after"] for r in records if r.get("max_drawdown_after") is not None]
        if mdd_vals:
            stats["drawdown"] = {
                "avg_max_drawdown":    f"{np.mean(mdd_vals):.1f}%",
                "worst_max_drawdown":  f"{np.min(mdd_vals):.1f}%",
            }
        return stats

    def _build_summary(self, entries: List[int], stats: Dict) -> str:
        s12 = stats.get("12m", {})
        return (
            f"共触发 {len(entries)} 次超跌信号 | "
            f"12个月胜率 {s12.get('win_rate', 'N/A')} | "
            f"平均收益 {s12.get('avg_return', 'N/A')} | "
            f"中位收益 {s12.get('median_return', 'N/A')}"
        )

    # ── 数据标准化 ────────────────────────────────────────────

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()

        date_col = next((c for c in ["日期", "Date", "date"] if c in df.columns), None)
        if date_col:
            df["date"] = pd.to_datetime(df[date_col], errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["date"] = df.index
        else:
            return pd.DataFrame()

        close_col = next(
            (c for c in ["收盘", "Close", "close", "Adj Close"] if c in df.columns), None
        )
        if not close_col:
            return pd.DataFrame()
        df["close"] = pd.to_numeric(df[close_col], errors="coerce")

        return df[["date", "close"]].dropna().drop_duplicates("date")

    # ── 格式化报告文本 ─────────────────────────────────────────

    def format_report(self, result: Dict) -> str:
        if "error" in result:
            return f"**回测失败**: {result['error']}\n"

        lines = [
            f"**触发信号次数**: {result['signal_count']}",
            f"**回测摘要**: {result.get('summary', '')}",
            "",
            "| 持有期 | 样本数 | 胜率 | 均值收益 | 中位收益 | 最优 | 最差 |",
            "|--------|--------|------|----------|----------|------|------|",
        ]
        for label in ["6m", "12m", "18m"]:
            s = result["stats"].get(label, {})
            if not s:
                continue
            lines.append(
                f"| {label} | {s['count']} | {s['win_rate']} | {s['avg_return']} "
                f"| {s['median_return']} | {s['best']} | {s['worst']} |"
            )

        dd = result["stats"].get("drawdown", {})
        if dd:
            lines += [
                "",
                f"**持有期平均最大回撤**: {dd.get('avg_max_drawdown', 'N/A')}  "
                f"  **最大最大回撤**: {dd.get('worst_max_drawdown', 'N/A')}",
            ]

        entries = result.get("entries", [])
        if entries:
            lines += [
                "",
                "**历次入场记录 (12个月收益)**:",
                "",
                "| 入场日期 | 入场价 | 跌幅 | 12M收益 |",
                "|----------|--------|------|---------|",
            ]
            for e in entries:
                r12 = e.get("return_12m")
                r12_str = f"{r12:.1f}%" if r12 is not None else "N/A"
                lines.append(
                    f"| {e['entry_date']} | {e['entry_price']} "
                    f"| {e['drop_at_entry']} | {r12_str} |"
                )

        return "\n".join(lines)
