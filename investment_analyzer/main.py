#!/usr/bin/env python3
"""
投资分析工作流 - 主入口

用法:
    python main.py 600519                    # 分析贵州茅台
    python main.py AAPL                      # 分析苹果
    python main.py 00700.HK                  # 分析腾讯
    python main.py 600519 --depth quick      # 快速分析（仅量化）
    python main.py 600519 --no-ai            # 仅量化分析，不调用AI
    python main.py 600519 --charts           # 生成价格/估值/财务图表
    python main.py 600519 --backtest         # 附加历史回测报告
    python main.py --batch 600519,AAPL,00700.HK        # 批量分析
    python main.py --batch-file stocks.txt   # 从文件批量分析（每行一个代码）
"""
import sys
import os
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.market_router import MarketRouter, Market
from report.generator import ReportGenerator


def print_banner():
    print("=" * 60)
    print("  投资分析工作流 v2.0")
    print("  逻辑驱动型框架: 宏观定轨 + 基本面双轨筛选 + 仓位管理")
    print("=" * 60)
    print()


def fetch_data(stock_info):
    market = stock_info.market
    symbol = stock_info.symbol
    data   = {}

    if market == Market.A_SHARE:
        from data.a_share import AShareData
        source = AShareData()

        print(f"  获取A股数据: {symbol}")

        if not symbol.isdigit():
            found = source.search_by_name(symbol)
            if found:
                print(f"  找到代码: {found}")
                symbol = found
                stock_info.symbol = found
            else:
                print(f"  未找到股票: {symbol}")
                return data

        print("    → 基本信息...", end="", flush=True)
        data["basic_info"] = source.get_stock_info(symbol)
        print(" ✓")

        print("    → 历史行情(5年)...", end="", flush=True)
        data["price_history"] = source.get_price_history(symbol)
        print(f" ✓ ({len(data['price_history'])} 条记录)")

        print("    → 估值数据...", end="", flush=True)
        data["valuation"] = source.get_valuation(symbol)
        print(" ✓")

        print("    → 财务数据...", end="", flush=True)
        data["financials"] = source.get_financial_data(symbol)
        print(" ✓")

        print("    → 管理层持股变动...", end="", flush=True)
        data["insider"] = source.get_holder_changes(symbol)
        print(" ✓")

    elif market in (Market.US_SHARE, Market.HK_SHARE):
        from data.us_share import YFinanceData
        source = YFinanceData()
        mkt = "HK" if market == Market.HK_SHARE else "US"

        print(f"  获取{'港股' if mkt == 'HK' else '美股'}数据: {symbol}")

        print("    → 基本信息...", end="", flush=True)
        data["basic_info"] = source.get_stock_info(symbol, mkt)
        print(" ✓")

        print("    → 历史行情(5年)...", end="", flush=True)
        data["price_history"] = source.get_price_history(symbol, mkt)
        print(f" ✓ ({len(data['price_history'])} 条记录)")

        print("    → 财务数据...", end="", flush=True)
        data["financials"] = source.get_financials(symbol, mkt)
        print(" ✓")

        print("    → 内部人交易...", end="", flush=True)
        data["insider"] = source.get_insider_trades(symbol, mkt)
        print(" ✓")

    return data


def fetch_macro_data(market: Market):
    """获取宏观数据（Layer 0 所需）"""
    macro_data = {}

    if market != Market.A_SHARE:
        return macro_data

    try:
        from data.a_share import AShareMacro
        source = AShareMacro()

        print("    → 宽基指数(沪深300)行情...", end="", flush=True)
        macro_data["index_history"] = source.get_index_history("sh000300")
        print(" ✓")

        print("    → 沪深300 PE历史...", end="", flush=True)
        macro_data["pe_history"] = source.get_index_pe_history("沪深300")
        print(" ✓")

        print("    → 10年期国债收益率...", end="", flush=True)
        macro_data["yield_data"] = source.get_bond_yield()
        print(" ✓")

    except Exception as e:
        print(f" ⚠ 宏观数据获取失败: {e}")

    return macro_data


def run_macro_analysis(macro_data: dict) -> dict:
    """Layer 0: 宏观定轨"""
    from analyzers.macro_checker import MacroChecker
    checker = MacroChecker()
    return checker.analyze(
        index_data=macro_data.get("index_history"),
        pe_history=macro_data.get("pe_history"),
        yield_data=macro_data.get("yield_data"),
    )


def run_quant_analysis(data: dict, macro_regime: str = "ambiguous") -> dict:
    """量化分析: 跌幅检查 + 真反转验证 + 基本面双轨筛选 + 仓位建议"""
    analysis = {}

    # 第一关: 跌幅量化
    if "price_history" in data and "valuation" in data:
        from analyzers.drop_checker import DropChecker
        print("\n  第一关: 跌幅量化分析...", end="", flush=True)
        result = DropChecker().analyze(data["price_history"], data.get("valuation", {}))
        analysis["drop_check"] = result
        print(f" {'通过 ✅' if result['passed'] else '未通过 ⚠'}")

    # 第二关: 真反转验证
    if "financials" in data:
        from analyzers.reversal_checker import ReversalChecker
        print("  第二关: 真反转验证...", end="", flush=True)
        result = ReversalChecker().analyze(data["financials"], data.get("insider"))
        analysis["reversal_check"] = result
        print(f" {result['verdict']}")

    # 第三关: 护城河验证（六维度）
    if "financials" in data:
        from analyzers.moat_checker import MoatChecker
        print("  第三关: 护城河验证（六维度）...", end="", flush=True)
        result = MoatChecker().analyze(data["financials"], data.get("basic_info"))
        analysis["moat"] = result
        print(f" {'通过 ✅' if result['passed'] else '待AI补充 ⚠'} ({result['score']}/6)")

    # 基本面双轨筛选 (Layer 2 Alpha 决策层)
    if "financials" in data:
        from analyzers.fundamental_screener import FundamentalScreener
        print("  Layer 2:  基本面双轨筛选 (A/B类)...", end="", flush=True)
        result = FundamentalScreener().screen(
            data["financials"], data.get("valuation")
        )
        analysis["fundamental_screen"] = result
        rec = result.get("recommended_type")
        label = f"通过 ({rec}类) ✅" if rec else "未通过 ⚠"
        print(f" {label}")

    # 仓位建议
    from analyzers.position_sizer import PositionSizer
    analysis["position"] = PositionSizer().calculate(analysis, macro_regime)

    analysis["basic_info"] = data.get("basic_info", {})
    return analysis


def run_ai_analysis(stock_info, data: dict, analysis: dict) -> dict:
    """AI 辅助分析: 多头 + 空头 + 综合决策"""
    from agents.base_agent import BullAgent, BearAgent, JudgeAgent

    company = stock_info.name or stock_info.symbol

    context_parts = []
    if "basic_info" in data:
        context_parts.append(f"基本信息: {data['basic_info']}")
    if "drop_check" in analysis:
        context_parts.append(f"跌幅分析: {analysis['drop_check'].get('summary', '')}")
    if "reversal_check" in analysis:
        context_parts.append(f"反转验证: {analysis['reversal_check'].get('summary', '')}")
    if "moat" in analysis:
        context_parts.append(f"护城河: {analysis['moat'].get('summary', '')}")
    if "fundamental_screen" in analysis:
        context_parts.append(f"基本面筛选: {analysis['fundamental_screen'].get('summary', '')}")
    if "macro" in analysis:
        context_parts.append(f"宏观状态: {analysis['macro'].get('summary', '')}")

    data_context = "\n".join(context_parts)

    # Phase 3: 第一性原理分析
    print("\n  第一性原理分析中...", flush=True)
    from agents.base_agent import FirstPrinciplesAgent
    analysis["first_principles"] = FirstPrinciplesAgent().analyze(company, data_context)

    # Phase 4: 多空辩论
    print("  多头分析师工作中...", flush=True)
    analysis["bull_case"] = BullAgent().analyze(company, data_context)

    print("  空头分析师工作中...", flush=True)
    analysis["bear_case"] = BearAgent().analyze(
        company, analysis["bull_case"], data_context
    )

    print("  裁判综合评估中...", flush=True)
    position_ctx = analysis.get("position", {}).get("summary", "仓位数据待计算")
    macro_ctx    = analysis.get("macro",    {}).get("summary", "宏观数据待获取")
    analysis["decision"] = JudgeAgent().decide(
        company=company,
        bull_case=analysis["bull_case"],
        bear_case=analysis["bear_case"],
        quant_data=data_context,
        macro_context=macro_ctx,
        position_context=position_ctx,
    )
    return analysis


def run_backtest(price_history, symbol: str) -> str:
    """运行历史回测并返回格式化报告文本"""
    from analyzers.backtester import Backtester
    print("  历史回测运行中...", end="", flush=True)
    result = Backtester().run(price_history)
    print(f" {result.get('summary', 'done')}")
    return Backtester().format_report(result)


def generate_charts(stock_info, data: dict) -> dict:
    """生成所有图表，返回 {type: path} 字典"""
    from report.charts import generate_all_charts
    from config import REPORT_CONFIG
    print("  生成图表...", end="", flush=True)
    charts = generate_all_charts(
        symbol       = stock_info.symbol,
        name         = stock_info.name or "",
        price_history= data.get("price_history"),
        valuation    = data.get("valuation"),
        financials   = data.get("financials"),
        output_dir   = REPORT_CONFIG["output_dir"],
    )
    generated = [k for k, v in charts.items() if v]
    print(f" {len(generated)} 张 ({', '.join(generated)})" if generated else " ⚠ matplotlib 未安装")
    return charts


def analyze_one(symbol: str, args) -> dict:
    """对单只股票执行完整分析流程，返回摘要信息"""
    market_names = {Market.A_SHARE: "A股(沪深)", Market.HK_SHARE: "港股", Market.US_SHARE: "美股"}

    # Step 1: 识别市场
    stock_info = MarketRouter.identify(symbol)
    print(f"\n  识别结果: {stock_info.symbol} → {market_names[stock_info.market]}")

    # Step 2: 个股数据
    print("获取个股数据...")
    data = fetch_data(stock_info)
    if not data:
        print(f"  ⚠ [{symbol}] 数据获取失败，跳过")
        return {"symbol": symbol, "error": "数据获取失败"}

    if "basic_info" in data:
        info = data["basic_info"]
        stock_info.name = info.get("股票简称", info.get("name", stock_info.symbol))
    print(f"\n  公司名称: {stock_info.name}")

    # Step 3: 宏观数据 (仅A股)
    macro_result = {}
    macro_regime = "ambiguous"
    if stock_info.market == Market.A_SHARE:
        print("\n获取宏观数据 (Layer 0)...")
        macro_data = fetch_macro_data(stock_info.market)
        if macro_data:
            print("\n  Layer 0: 宏观定轨...", end="", flush=True)
            macro_result = run_macro_analysis(macro_data)
            macro_regime = macro_result.get("regime", "ambiguous")
            print(f" {macro_result.get('regime_label', macro_regime)}")

    # Step 4: 量化分析
    print("\n量化分析...")
    analysis = run_quant_analysis(data, macro_regime)
    if macro_result:
        analysis["macro"] = macro_result

    # Step 5: 回测（可选）
    backtest_text = ""
    if getattr(args, "backtest", False) and "price_history" in data:
        print("\n历史回测...")
        backtest_text = run_backtest(data["price_history"], stock_info.symbol)

    # Step 6: AI 分析（可选）
    if not args.no_ai and args.depth == "full":
        print("\nAI 辅助分析...")
        try:
            analysis = run_ai_analysis(stock_info, data, analysis)
        except Exception as e:
            print(f"  ⚠ AI分析跳过 (原因: {e})")
            print("  提示: 设置环境变量 LLM_API_KEY 和 LLM_MODEL 以启用AI分析")

    # Step 7: 图表（可选）
    charts = {}
    if getattr(args, "charts", False):
        print("\n生成图表...")
        charts = generate_charts(stock_info, data)

    # Step 8: 生成报告
    print("\n生成报告...")
    generator = ReportGenerator()
    report_content = generator.generate(
        stock_info   = {"name": stock_info.name, "symbol": stock_info.symbol},
        analysis     = analysis,
        backtest_text= backtest_text,
        charts       = charts,
    )
    filepath = generator.save(report_content, stock_info.name, stock_info.symbol)
    print(f"\n  报告已保存: {filepath}")

    # 摘要
    summary = {
        "symbol":   stock_info.symbol,
        "name":     stock_info.name,
        "filepath": filepath,
    }
    if macro_result:
        summary["macro"] = macro_result.get("regime_label", "N/A")
    if "drop_check" in analysis:
        summary["drop"] = "✅" if analysis["drop_check"].get("passed") else "❌"
    if "reversal_check" in analysis:
        summary["reversal"] = analysis["reversal_check"].get("verdict", "N/A")
    if "moat" in analysis:
        summary["moat"] = f"{analysis['moat'].get('score', 0)}/6"
    if "position" in analysis:
        summary["position"] = analysis["position"].get("summary", "N/A")
    return summary


def print_batch_summary(summaries: list):
    print("\n" + "=" * 70)
    print("  批量分析摘要")
    print("=" * 70)
    print(f"  {'代码':<12} {'名称':<16} {'超跌':<6} {'反转':<10} {'护城河':<8} {'仓位建议'}")
    print("  " + "─" * 66)
    for s in summaries:
        if "error" in s:
            print(f"  {s['symbol']:<12} ❌ {s['error']}")
            continue
        print(
            f"  {s.get('symbol',''):<12} {s.get('name',''):<16} "
            f"{s.get('drop','—'):<6} {s.get('reversal','—'):<10} "
            f"{s.get('moat','—'):<8} {s.get('position','—')}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="投资分析工作流 v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stock", nargs="?", help="股票代码或名称 (如: 600519, AAPL, 00700.HK)")
    parser.add_argument("--depth",      choices=["quick", "full"], default="full")
    parser.add_argument("--no-ai",      action="store_true", help="跳过AI分析")
    parser.add_argument("--charts",     action="store_true", help="生成价格/估值/财务图表")
    parser.add_argument("--backtest",   action="store_true", help="附加历史回测分析")
    parser.add_argument("--batch",      metavar="SYMBOLS",
                        help="批量分析（逗号分隔，如: 600519,AAPL,00700.HK）")
    parser.add_argument("--batch-file", metavar="FILE",
                        help="从文件批量分析（每行一个股票代码）")
    args = parser.parse_args()

    print_banner()

    # ── 批量模式 ──────────────────────────────────────────────
    symbols = []
    if args.batch:
        symbols = [s.strip() for s in args.batch.split(",") if s.strip()]
    elif args.batch_file:
        if not os.path.exists(args.batch_file):
            print(f"文件不存在: {args.batch_file}")
            sys.exit(1)
        with open(args.batch_file, "r", encoding="utf-8") as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if symbols:
        print(f"  批量模式: {len(symbols)} 只标的\n")
        summaries = []
        for i, sym in enumerate(symbols, 1):
            print(f"\n{'─'*60}")
            print(f"  [{i}/{len(symbols)}] {sym}")
            print("─" * 60)
            summaries.append(analyze_one(sym, args))
        print_batch_summary(summaries)
        return

    # ── 单只模式 ──────────────────────────────────────────────
    if not args.stock:
        parser.print_help()
        sys.exit(1)

    summary = analyze_one(args.stock, args)

    print("\n" + "=" * 60)
    print("  分析摘要")
    print("=" * 60)
    for k, v in summary.items():
        if k not in ("filepath", "symbol", "error"):
            print(f"  {k:<14}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
