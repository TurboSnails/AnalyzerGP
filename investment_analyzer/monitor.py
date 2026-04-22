#!/usr/bin/env python3
"""
投资监控工具 v1.0 - 观察名单管理 + 自动信号检测

用法:
    python monitor.py                        # 检查所有观察标的并输出信号摘要
    python monitor.py --add 600519           # 添加到观察名单
    python monitor.py --add AAPL --note "待确认反转" # 添加并附注
    python monitor.py --remove 600519        # 从观察名单移除
    python monitor.py --list                 # 显示观察名单状态
    python monitor.py --symbol 600519        # 只检查单只标的
"""
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_OUTPUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
_WATCHLIST_FILE = os.path.join(_OUTPUT_DIR, "watchlist.json")
_MONITOR_LOG    = os.path.join(_OUTPUT_DIR, "monitor_log.jsonl")


# ──────────────────────────────────────────────
# Watchlist persistence
# ──────────────────────────────────────────────

def _load() -> List[Dict]:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(_WATCHLIST_FILE):
        return []
    with open(_WATCHLIST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(watchlist: List[Dict]):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)


def cmd_add(symbol: str, note: str = ""):
    wl = _load()
    if any(w["symbol"] == symbol for w in wl):
        print(f"  {symbol} 已在观察名单中")
        return
    wl.append({
        "symbol":       symbol,
        "note":         note,
        "added_at":     datetime.now().strftime("%Y-%m-%d"),
        "last_check":   None,
        "last_signal":  None,
        "alert_count":  0,
    })
    _save(wl)
    print(f"  ✅ 已添加: {symbol}" + (f"  ({note})" if note else ""))


def cmd_remove(symbol: str):
    wl = _load()
    new_wl = [w for w in wl if w["symbol"] != symbol]
    if len(new_wl) == len(wl):
        print(f"  {symbol} 不在观察名单中")
        return
    _save(new_wl)
    print(f"  ✅ 已移除: {symbol}")


def cmd_list():
    wl = _load()
    if not wl:
        print("  观察名单为空。使用 --add <代码> 添加标的。")
        return
    print(f"  观察名单（{len(wl)} 只）:\n")
    print(f"  {'代码':<14} {'加入日期':<12} {'最后检查':<20} {'最后信号':<18} {'备注'}")
    print("  " + "─" * 78)
    for w in wl:
        print(
            f"  {w['symbol']:<14} {w['added_at']:<12} "
            f"{w.get('last_check', '从未'):<20} "
            f"{w.get('last_signal', '—'):<18} "
            f"{w.get('note', '')}"
        )


# ──────────────────────────────────────────────
# Single-stock quick check
# ──────────────────────────────────────────────

def check_stock(symbol: str) -> Dict:
    """对单只股票运行快速量化信号检测（跌幅 + 宏观状态）"""
    from data.market_router import MarketRouter, Market

    result: Dict = {
        "symbol":    symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "signal":    "—",
    }

    try:
        stock_info = MarketRouter.identify(symbol)
        market     = stock_info.market

        if market == Market.A_SHARE:
            from data.a_share import AShareData
            src           = AShareData()
            price_history = src.get_price_history(symbol)
            valuation     = src.get_valuation(symbol)
        else:
            from data.us_share import YFinanceData
            src    = YFinanceData()
            mkt    = "HK" if market == Market.HK_SHARE else "US"
            price_history = src.get_price_history(symbol, mkt)
            valuation     = {}

        if price_history is not None and not price_history.empty:
            from analyzers.drop_checker import DropChecker
            drop = DropChecker().analyze(price_history, valuation)
            result["drop_check"] = {
                "passed":         drop.get("passed", False),
                "drop_from_high": drop.get("details", {}).get("drop_from_high", "N/A"),
                "pe_percentile":  drop.get("details", {}).get("pe_percentile", "N/A"),
                "summary":        drop.get("summary", ""),
            }
            if drop.get("passed"):
                result["signal"] = "⚡ 超跌信号"
            else:
                drop_val = drop.get("details", {}).get("drop_from_high", 0)
                if isinstance(drop_val, float) and drop_val >= 0.30:
                    result["signal"] = f"⚠ 关注 ({drop_val:.0%} 跌幅)"
                else:
                    result["signal"] = "✅ 正常"
        else:
            result["signal"] = "❓ 数据获取失败"

    except Exception as e:
        result["signal"] = f"❌ 错误: {str(e)[:60]}"

    return result


# ──────────────────────────────────────────────
# Monitor all watchlist stocks
# ──────────────────────────────────────────────

def _append_log(entry: Dict):
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_MONITOR_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def cmd_monitor(only_symbol: Optional[str] = None):
    wl = _load()
    if not wl:
        print("  观察名单为空。使用 --add <代码> 添加标的后再运行。")
        return

    targets = [w for w in wl if only_symbol is None or w["symbol"] == only_symbol]
    if not targets:
        print(f"  {only_symbol} 不在观察名单中")
        return

    print(f"  检查 {len(targets)} 只标的...\n")

    alerts: List[Dict] = []
    for item in targets:
        sym = item["symbol"]
        print(f"  [{sym}]", end=" ", flush=True)
        r = check_stock(sym)
        print(r["signal"])

        prev_signal = item.get("last_signal")
        new_signal  = r["signal"]

        # 新触发超跌信号才记为 alert
        if "超跌" in new_signal and "超跌" not in (prev_signal or ""):
            alerts.append({"symbol": sym, "signal": new_signal,
                           "detail": r.get("drop_check", {})})
            item["alert_count"] = item.get("alert_count", 0) + 1

        item["last_check"]  = r["timestamp"]
        item["last_signal"] = new_signal
        _append_log(r)

    _save(wl)

    # ── 摘要 ──
    print()
    print("  " + "═" * 50)
    if alerts:
        print(f"  ⚡ {len(alerts)} 个新超跌信号:\n")
        for a in alerts:
            dc = a.get("detail", {})
            drop_str = f"跌幅 {float(dc['drop_from_high']):.1%}" if isinstance(dc.get("drop_from_high"), float) else ""
            pe_str   = f"PE分位 {float(dc['pe_percentile']):.1%}" if isinstance(dc.get("pe_percentile"), float) else ""
            meta     = " | ".join(filter(None, [drop_str, pe_str]))
            print(f"     {a['symbol']:12}  {a['signal']}  {meta}")
    else:
        print("  ✅ 无新超跌信号")
    print("  " + "═" * 50)

    # ── 全表状态 ──
    if only_symbol is None and len(targets) > 1:
        print()
        cmd_list()


# ──────────────────────────────────────────────
# Report: last N log entries per symbol
# ──────────────────────────────────────────────

def cmd_report(days: int = 30):
    if not os.path.exists(_MONITOR_LOG):
        print("  尚无监控日志，先运行 python monitor.py 生成记录。")
        return

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    import pandas as pd

    rows = []
    with open(_MONITOR_LOG, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line.strip()))
            except Exception:
                continue

    if not rows:
        print("  日志为空")
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    recent = df[df["timestamp"] >= cutoff].sort_values("timestamp", ascending=False)

    print(f"\n  最近 {days} 天监控记录（{len(recent)} 条）:\n")
    print(f"  {'代码':<14} {'时间':<20} {'信号'}")
    print("  " + "─" * 55)
    for _, row in recent.iterrows():
        print(f"  {row.get('symbol',''):<14} {str(row.get('timestamp',''))[:16]:<20} {row.get('signal','—')}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="投资监控工具 - 观察名单管理与信号检测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--add",    metavar="SYMBOL", help="添加标的到观察名单")
    group.add_argument("--remove", metavar="SYMBOL", help="从观察名单移除")
    group.add_argument("--list",   action="store_true", help="显示观察名单")
    group.add_argument("--symbol", metavar="SYMBOL", help="只检查单只标的")
    group.add_argument("--report", action="store_true", help="显示最近 30 天日志")
    parser.add_argument("--note",  default="", help="添加标的时附加备注")
    parser.add_argument("--days",  type=int, default=30, help="--report 显示天数")
    args = parser.parse_args()

    print("═" * 52)
    print("  投资监控工具 v1.0")
    print("═" * 52)
    print()

    if args.add:
        cmd_add(args.add, args.note)
    elif args.remove:
        cmd_remove(args.remove)
    elif args.list:
        cmd_list()
    elif args.report:
        cmd_report(args.days)
    else:
        cmd_monitor(only_symbol=args.symbol)


if __name__ == "__main__":
    main()
