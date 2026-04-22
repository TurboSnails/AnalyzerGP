"""
图表可视化模块 - 生成价格走势图、估值历史图、财务趋势图
依赖: matplotlib (可选, 未安装时跳过图表生成)
"""
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

_BG_DARK  = "#1a1a2e"
_BG_PANEL = "#16213e"
_TEXT     = "#e0e0e0"
_GRID     = "#2a2a4a"
_BLUE     = "#4fc3f7"
_ORANGE   = "#ff9800"
_GREEN    = "#26a69a"
_RED      = "#ef5350"
_GREY     = "#78909c"


def _apply_dark_style(ax):
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.yaxis.label.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.4, alpha=0.6)


def _get_close_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """从行情 DataFrame 提取带日期索引的收盘价序列"""
    if df is None or df.empty:
        return None
    df = df.copy()
    date_col  = next((c for c in ["日期", "Date", "date"] if c in df.columns), None)
    close_col = next((c for c in ["收盘", "Close", "close", "Adj Close"] if c in df.columns), None)
    if not close_col:
        return None
    if date_col:
        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df["_d"] = df.index
    else:
        return None
    df["_c"] = pd.to_numeric(df[close_col], errors="coerce")
    s = df.dropna(subset=["_d", "_c"]).sort_values("_d").set_index("_d")["_c"]
    return s if not s.empty else None


def _get_volume_series(df: pd.DataFrame) -> Optional[pd.Series]:
    vol_col = next((c for c in ["成交量", "Volume", "volume"] if c in df.columns), None)
    if not vol_col or df.empty:
        return None
    df = df.copy()
    date_col = next((c for c in ["日期", "Date", "date"] if c in df.columns), None)
    if date_col:
        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
    elif isinstance(df.index, pd.DatetimeIndex):
        df["_d"] = df.index
    else:
        return None
    df["_v"] = pd.to_numeric(df[vol_col], errors="coerce")
    return df.dropna(subset=["_d", "_v"]).sort_values("_d").set_index("_d")["_v"]


# ──────────────────────────────────────────────
# Chart 1: Price history + 200MA + drop-from-high
# ──────────────────────────────────────────────

def generate_price_chart(
    price_history: pd.DataFrame,
    symbol: str,
    name: str = "",
    output_dir: str = ".",
) -> str:
    if not HAS_MATPLOTLIB:
        return ""
    close = _get_close_series(price_history)
    if close is None or len(close) < 20:
        return ""

    vol   = _get_volume_series(price_history)
    ma200 = close.rolling(200, min_periods=1).mean()
    window = min(504, len(close))
    high_roll = close.rolling(window, min_periods=1).max()
    drop_pct  = (close / high_roll - 1) * 100

    nrows = 3 if vol is not None else 2
    ratios = [3, 1, 1] if nrows == 3 else [3, 1]

    fig = plt.figure(figsize=(14, 10 if nrows == 3 else 7), facecolor=_BG_DARK)
    gs  = GridSpec(nrows, 1, figure=fig, height_ratios=ratios, hspace=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(nrows)]

    title = f"{name} ({symbol})" if name else symbol
    fig.suptitle(f"{title} - Price Analysis", color=_TEXT, fontsize=13, fontweight="bold", y=0.99)

    ax_price = axes[0]
    _apply_dark_style(ax_price)
    ax_price.plot(close.index, close.values, color=_BLUE, linewidth=1.2, label="Close", zorder=2)
    ax_price.plot(ma200.index, ma200.values, color=_ORANGE, linewidth=1.4, linestyle="--", label="MA200", zorder=3)
    ax_price.fill_between(close.index, close.values, ma200.values,
                          where=(close.values >= ma200.values), alpha=0.12, color=_BLUE)
    ax_price.fill_between(close.index, close.values, ma200.values,
                          where=(close.values < ma200.values), alpha=0.12, color=_RED)
    ax_price.set_ylabel("Price", color=_TEXT, fontsize=9)
    ax_price.legend(loc="upper left", facecolor=_BG_DARK, labelcolor=_TEXT, fontsize=8, framealpha=0.7)

    idx = 1
    if vol is not None:
        vol_aligned = vol.reindex(close.index).fillna(0)
        prev_close  = close.shift(1).fillna(close)
        bar_colors  = [_RED if c < p else _GREEN for c, p in zip(close.values, prev_close.values)]
        ax_vol = axes[idx]
        _apply_dark_style(ax_vol)
        ax_vol.bar(vol_aligned.index, vol_aligned.values, color=bar_colors, alpha=0.65, width=1.2)
        ax_vol.set_ylabel("Volume", color=_TEXT, fontsize=9)
        ax_vol.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x/1e8:.1f}亿" if x >= 1e8 else f"{x/1e6:.0f}M")
        )
        idx += 1

    ax_drop = axes[idx]
    _apply_dark_style(ax_drop)
    ax_drop.fill_between(drop_pct.index, drop_pct.values, 0,
                         where=(drop_pct.values <= -50), color=_RED, alpha=0.55, label="Drop >50%")
    ax_drop.fill_between(drop_pct.index, drop_pct.values, 0,
                         where=(drop_pct.values > -50), color=_GREY, alpha=0.25)
    ax_drop.axhline(y=-50, color=_RED, linewidth=0.8, linestyle="--", alpha=0.8)
    ax_drop.set_ylabel("Drop %", color=_TEXT, fontsize=9)
    ax_drop.set_ylim(min(drop_pct.min() * 1.1, -65), 5)
    ax_drop.legend(loc="lower left", facecolor=_BG_DARK, labelcolor=_TEXT, fontsize=8, framealpha=0.7)
    ax_drop.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_drop.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax_drop.xaxis.get_majorticklabels(), rotation=25, ha="right", color=_TEXT)

    for ax in axes[:-1]:
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_price_{datetime.now().strftime('%Y%m%d')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# Chart 2: PE/PB valuation percentile history
# ──────────────────────────────────────────────

def generate_valuation_chart(
    valuation: Dict[str, Any],
    symbol: str,
    name: str = "",
    output_dir: str = ".",
) -> str:
    if not HAS_MATPLOTLIB:
        return ""
    hist = valuation.get("history")
    if hist is None or hist.empty:
        return ""

    hist = hist.copy()
    date_col = next((c for c in ["trade_date", "date", "Date", "日期"] if c in hist.columns), None)
    if date_col:
        hist["_d"] = pd.to_datetime(hist[date_col], errors="coerce")
    elif isinstance(hist.index, pd.DatetimeIndex):
        hist["_d"] = hist.index
    else:
        return ""

    hist = hist.dropna(subset=["_d"]).sort_values("_d").set_index("_d")

    pe_col = next((c for c in ["pe_ttm", "pe", "PE_TTM", "PE"] if c in hist.columns), None)
    pb_col = next((c for c in ["pb", "PB"] if c in hist.columns), None)

    if pe_col is None and pb_col is None:
        return ""

    series_to_plot = []
    if pe_col:
        pe = pd.to_numeric(hist[pe_col], errors="coerce").dropna()
        pe = pe[(pe > 0) & (pe < 300)]
        if not pe.empty:
            series_to_plot.append(("PE TTM", pe, _BLUE))
    if pb_col:
        pb = pd.to_numeric(hist[pb_col], errors="coerce").dropna()
        pb = pb[(pb > 0) & (pb < 50)]
        if not pb.empty:
            series_to_plot.append(("PB", pb, _ORANGE))

    if not series_to_plot:
        return ""

    fig, axes = plt.subplots(len(series_to_plot), 1, figsize=(14, 4 * len(series_to_plot)),
                              facecolor=_BG_DARK, sharex=True)
    if len(series_to_plot) == 1:
        axes = [axes]

    title = f"{name} ({symbol})" if name else symbol
    fig.suptitle(f"{title} - Valuation History", color=_TEXT, fontsize=13, fontweight="bold", y=0.99)

    for ax, (label, series, color) in zip(axes, series_to_plot):
        _apply_dark_style(ax)
        p20 = series.quantile(0.20)
        p80 = series.quantile(0.80)
        ax.plot(series.index, series.values, color=color, linewidth=1.0, label=label)
        ax.axhline(p20, color=_GREEN, linewidth=0.9, linestyle="--", alpha=0.8, label=f"20th pct ({p20:.1f})")
        ax.axhline(p80, color=_RED,   linewidth=0.9, linestyle="--", alpha=0.8, label=f"80th pct ({p80:.1f})")
        ax.fill_between(series.index, series.values, p20,
                        where=(series.values <= p20), color=_GREEN, alpha=0.18)
        ax.fill_between(series.index, series.values, p80,
                        where=(series.values >= p80), color=_RED,   alpha=0.18)
        ax.set_ylabel(label, color=_TEXT, fontsize=9)
        ax.legend(loc="upper left", facecolor=_BG_DARK, labelcolor=_TEXT, fontsize=8, framealpha=0.7)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=25, ha="right", color=_TEXT)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_valuation_{datetime.now().strftime('%Y%m%d')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# Chart 3: Financial trends (revenue growth + gross margin)
# ──────────────────────────────────────────────

def _extract_metric_series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
    """从财务指标表中提取指定指标的时间序列"""
    if df is None or df.empty:
        return None
    col = next((c for c in candidates if c in df.columns), None)
    if col is None:
        return None
    date_col = next((c for c in ["报告期", "date", "Date", "REPORT_DATE"] if c in df.columns), None)
    if not date_col:
        return None
    df = df.copy()
    df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_v"] = pd.to_numeric(df[col], errors="coerce")
    s = df.dropna(subset=["_d", "_v"]).sort_values("_d").set_index("_d")["_v"]
    # Normalize if raw percentage (values > 2 assume they're already in percent, else *100)
    if not s.empty and s.abs().median() > 2:
        pass  # already in percent
    else:
        s = s * 100
    return s if not s.empty else None


def generate_financial_chart(
    financials: Dict[str, pd.DataFrame],
    symbol: str,
    name: str = "",
    output_dir: str = ".",
) -> str:
    if not HAS_MATPLOTLIB:
        return ""
    if not financials:
        return ""

    indicators = financials.get("indicators", pd.DataFrame())

    revenue_growth = _extract_metric_series(
        indicators,
        ["营业收入增长率", "营收增速", "Revenue Growth", "revenue_growth",
         "营业总收入同比增长率", "营业收入同比增长率"]
    )
    gross_margin = _extract_metric_series(
        indicators,
        ["毛利率", "销售毛利率", "Gross Margin", "gross_margin"]
    )
    net_margin = _extract_metric_series(
        indicators,
        ["净利率", "销售净利率", "净利润率", "Net Margin", "net_margin"]
    )

    series_map = {
        "Revenue Growth %": (revenue_growth, _BLUE,   "bar"),
        "Gross Margin %":   (gross_margin,   _ORANGE, "line"),
        "Net Margin %":     (net_margin,     _GREEN,  "line"),
    }
    valid = {k: v for k, v in series_map.items() if v[0] is not None}
    if not valid:
        return ""

    nrows = len(valid)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 3.5 * nrows),
                              facecolor=_BG_DARK, sharex=True)
    if nrows == 1:
        axes = [axes]

    title = f"{name} ({symbol})" if name else symbol
    fig.suptitle(f"{title} - Financial Trends", color=_TEXT, fontsize=13, fontweight="bold", y=0.99)

    for ax, (label, (series, color, style)) in zip(axes, valid.items()):
        _apply_dark_style(ax)
        recent = series.tail(20)  # last 20 periods for readability
        if style == "bar":
            bar_colors = [_GREEN if v >= 0 else _RED for v in recent.values]
            ax.bar(range(len(recent)), recent.values, color=bar_colors, alpha=0.75, width=0.65)
            ax.set_xticks(range(len(recent)))
            ax.set_xticklabels(
                [d.strftime("%Y-Q%q") if hasattr(d, "strftime") else str(d) for d in recent.index],
                rotation=40, ha="right", color=_TEXT, fontsize=7
            )
            ax.axhline(0, color=_TEXT, linewidth=0.5, alpha=0.5)
        else:
            ax.plot(range(len(recent)), recent.values, color=color, linewidth=1.4, marker="o",
                    markersize=3, label=label)
            ax.set_xticks(range(len(recent)))
            ax.set_xticklabels(
                [d.strftime("%Y-Q%q") if hasattr(d, "strftime") else str(d) for d in recent.index],
                rotation=40, ha="right", color=_TEXT, fontsize=7
            )
            ax.fill_between(range(len(recent)), recent.values, alpha=0.12, color=color)
        ax.set_ylabel(label, color=_TEXT, fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{symbol}_financials_{datetime.now().strftime('%Y%m%d')}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor=_BG_DARK)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# Convenience: generate all charts at once
# ──────────────────────────────────────────────

def generate_all_charts(
    symbol: str,
    name: str = "",
    price_history: pd.DataFrame = None,
    valuation: Dict[str, Any] = None,
    financials: Dict[str, pd.DataFrame] = None,
    output_dir: str = ".",
) -> Dict[str, str]:
    """生成所有图表，返回 {chart_type: file_path} 字典（路径为空字符串表示生成失败）"""
    charts = {}
    charts_dir = os.path.join(output_dir, "charts")

    if price_history is not None:
        charts["price"] = generate_price_chart(price_history, symbol, name, charts_dir)

    if valuation is not None:
        charts["valuation"] = generate_valuation_chart(valuation, symbol, name, charts_dir)

    if financials is not None:
        charts["financials"] = generate_financial_chart(financials, symbol, name, charts_dir)

    return {k: v for k, v in charts.items() if v}
