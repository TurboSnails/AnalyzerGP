"""
Markdown 报告生成器 v2.0
融合三层框架: Layer 0 宏观定轨 + Layer 2 基本面双轨筛选 + 仓位管理
"""
from datetime import datetime
from typing import Dict, Any
import os
from config import REPORT_CONFIG


class ReportGenerator:

    def generate(
        self,
        stock_info: Dict,
        analysis: Dict,
        backtest_text: str = "",
        charts: Dict[str, str] = None,
    ) -> str:
        now    = datetime.now().strftime("%Y-%m-%d %H:%M")
        name   = stock_info.get("name", stock_info.get("symbol", "未知"))
        symbol = stock_info.get("symbol", "")
        charts = charts or {}

        sections = []

        # ── 标题 ──────────────────────────────────────────────
        sections.append(f"# 投资分析报告: {name} ({symbol})")
        sections.append(f"> 生成时间: {now} | 分析框架版本: 2.0")
        sections.append(f"> ⚠️ 本报告仅供个人投资研究参考，不构成专业投资建议\n")

        # ── 一、认知偏误自检 ──────────────────────────────────
        sections.append("## 一、认知偏误自检\n")
        sections.append(
            "阅读报告前，请先确认自己没有以下偏误:\n"
            "- **确认偏误**: 是否已有预设立场，只在寻找支持证据？\n"
            "- **锚定偏误**: 是否被某个历史价格牢牢锁定？\n"
            "- **近因偏误**: 是否过度依赖近期消息而忽视长期逻辑？\n"
        )

        # ── 二、Layer 0 宏观定轨 ──────────────────────────────
        sections.append("## 二、Layer 0: 宏观定轨\n")
        macro = analysis.get("macro", {})
        if macro and macro.get("regime"):
            sections.append(f"**市场状态**: {macro.get('regime_label', 'N/A')}")
            sections.append(f"**建议操作**: {macro.get('action', 'N/A')}\n")

            sigs = macro.get("signals", {})
            if sigs:
                sections.append("| 信号 | 状态 | 数据 |")
                sections.append("|------|------|------|")
                sig_labels = {
                    "ma200":         "宽基指数 vs 200MA",
                    "pe_percentile": "PE历史分位",
                    "bond_yield":    "10Y国债收益率方向",
                }
                status_icons = {
                    "below": "📉 低于MA200", "above": "📈 高于MA200",
                    "low":   "✅ 低分位(低估)", "high": "⚠️ 高分位(高估)", "neutral": "➡️ 中性",
                    "rising": "⬆️ 上行", "falling": "⬇️ 下行", "flat": "➡️ 平稳",
                    "unknown": "❓ 数据不足",
                }
                for key, label in sig_labels.items():
                    sig  = sigs.get(key, {})
                    icon = status_icons.get(sig.get("status", "unknown"), sig.get("status", ""))
                    detail = sig.get("detail", "N/A")
                    sections.append(f"| {label} | {icon} | {detail} |")
                sections.append("")

            alloc = macro.get("layer1_allocation", {})
            if alloc:
                sections.append("**Layer 1 建议配置**:")
                for asset, pct in alloc.items():
                    sections.append(f"- {asset}: {pct}")
                sections.append("")
        else:
            sections.append("*宏观数据待获取（非A股市场或数据源不可用）*\n")

        # ── 三、个股量化分析 ──────────────────────────────────
        sections.append("## 三、个股量化分析\n")

        # 第一关
        sections.append("### 第一关: 超跌验证\n")
        drop = analysis.get("drop_check", {})
        if drop:
            status = "✅ 通过" if drop.get("passed") else "❌ 未通过"
            sections.append(f"**结果**: {status}\n")
            details = drop.get("details", {})
            if details:
                sections.append("| 指标 | 数值 |")
                sections.append("|------|------|")
                if "drop_from_high" in details:
                    sections.append(f"| 历史高点跌幅 | {details['drop_from_high']:.1%} |")
                if "pe_ttm" in details:
                    sections.append(f"| PE(TTM) | {details['pe_ttm']} |")
                if "pe_percentile" in details:
                    sections.append(f"| PE历史分位 | {details['pe_percentile']:.1%} |")
                if "pb_percentile" in details:
                    sections.append(f"| PB历史分位 | {details['pb_percentile']:.1%} |")
                sections.append("")
            for s in drop.get("signals", []):
                sections.append(f"- {s}")
            for w in drop.get("warnings", []):
                sections.append(f"- {w}")
            sections.append("")
        else:
            sections.append("*待分析*\n")

        # 第二关
        sections.append("### 第二关: 真反转 vs 价值陷阱（七维度）\n")
        reversal = analysis.get("reversal_check", {})
        if reversal:
            verdict = reversal.get("verdict", "待判断")
            score   = reversal.get("score", 0)
            total   = reversal.get("total", 7)
            sections.append(f"**判断**: {verdict} ({score}/{total} 维度通过)\n")
            sections.append("| 维度 | 状态 | 说明 |")
            sections.append("|------|------|------|")
            icons = {"positive": "✅", "negative": "❌", "pending": "⏳"}
            for dim_name, dim in reversal.get("dimensions", {}).items():
                sections.append(
                    f"| {dim['label']} | {icons.get(dim['status'], '❓')} | {dim['detail']} |"
                )
            sections.append("")
        else:
            sections.append("*待分析*\n")

        # 第三关: 护城河验证
        sections.append("### 第三关: 护城河验证（六维度）\n")
        moat = analysis.get("moat", {})
        if moat:
            score  = moat.get("score", 0)
            passed = moat.get("passed", False)
            sections.append(
                f"**结果**: {'✅ 具备护城河' if passed else '⚠️ 护城河薄弱'} "
                f"({score}/6 维度通过)\n"
            )
            sections.append(f"**护城河来源**: {moat.get('moat_type', 'N/A')}\n")
            sections.append("| 维度 | 状态 | 说明 |")
            sections.append("|------|------|------|")
            icons = {"positive": "✅", "negative": "❌", "pending": "⏳"}
            for dim in moat.get("dimensions", {}).values():
                sections.append(
                    f"| {dim['label']} | {icons.get(dim['status'], '❓')} | {dim['detail']} |"
                )
            sections.append("")
        else:
            sections.append("*待分析*\n")

        # Layer 2: 基本面双轨筛选
        sections.append("### Layer 2: 基本面双轨筛选（Alpha 入场条件）\n")
        fs = analysis.get("fundamental_screen", {})
        if fs:
            rec = fs.get("recommended_type")
            passed_icon = "✅" if fs.get("passed") else "❌"
            sections.append(
                f"**结果**: {passed_icon} {fs.get('summary', '')}\n"
            )

            for type_key, type_label in [("type_a", "A类: 价值/红利型"), ("type_b", "B类: 第二曲线成长型")]:
                t = fs.get(type_key, {})
                if not t:
                    continue
                t_icon = "✅" if t.get("passed") else "❌"
                marker = " ◀ 推荐" if rec == t.get("type") else ""
                sections.append(
                    f"#### {type_label} {t_icon}{marker} "
                    f"({t.get('score', 0)}/{t.get('total', 0)} 项通过，阈值 {t.get('threshold', 0)})\n"
                )
                sections.append("| 指标 | 结果 | 数据 | 阈值 |")
                sections.append("|------|------|------|------|")
                icons = {"positive": "✅", "negative": "❌", "pending": "⏳"}
                for ck in t.get("checks", {}).values():
                    sections.append(
                        f"| {ck.get('label', '')} "
                        f"| {icons.get(ck.get('status'), '❓')} "
                        f"| {ck.get('value', ck.get('detail', 'N/A'))} "
                        f"| {ck.get('threshold', '-')} |"
                    )
                sections.append("")
        else:
            sections.append("*待分析*\n")

        # ── 四、仓位管理建议 ──────────────────────────────────
        sections.append("## 四、仓位管理建议\n")
        pos = analysis.get("position", {})
        if pos:
            sections.append(f"**综合确信度**: {pos.get('confidence', 0):.0%} → {pos.get('position_tier', 'N/A')}")
            sections.append(f"**建议首批仓位**: {pos.get('suggested_position', 0):.0%}")
            sections.append(f"**单标的上限**: {pos.get('max_single', 0):.0%}")
            sections.append(f"**Alpha层总上限**: {pos.get('total_alpha_cap', 0):.0%}\n")

            rationale = pos.get("rationale", [])
            if rationale:
                sections.append("**仓位依据**:")
                for r in rationale:
                    sections.append(f"- {r}")
                sections.append("")

            constraints = pos.get("constraints", [])
            if constraints:
                sections.append("**执行纪律**:")
                for c in constraints:
                    sections.append(f"- {c}")
                sections.append("")

            exits = pos.get("exit_triggers", [])
            if exits:
                sections.append("**逻辑退出触发条件** (满足任一立即清仓):")
                for e in exits:
                    sections.append(f"- {e}")
                sections.append("")

            ts = pos.get("time_stop", {})
            if ts:
                sections.append(
                    f"**时间止损**: "
                    f"{ts.get('reassess_months', 18)}个月逻辑成立但无反应→重新评估；"
                    f"{ts.get('reduce_months', 24)}个月→减仓至1/2\n"
                )
        else:
            sections.append("*待计算*\n")

        # ── 五、第一性原理分析 ─────────────────────────────────
        sections.append("## 五、第一性原理分析（Phase 3）\n")
        fp = analysis.get("first_principles", "")
        if fp:
            sections.append(fp)
            sections.append("")
        else:
            sections.append("*需AI辅助分析（设置 LLM_API_KEY 后启用）*\n")

        # ── 六、多空辩论 ──────────────────────────────────────
        sections.append("## 六、多空辩论（Phase 4）\n")
        bull = analysis.get("bull_case", "")
        bear = analysis.get("bear_case", "")

        if bull:
            sections.append("### 多头视角\n")
            sections.append(bull)
            sections.append("")
        else:
            sections.append("*需AI辅助分析（设置 LLM_API_KEY 后启用）*\n")

        if bear:
            sections.append("### 空头视角（反对清单）\n")
            sections.append(bear)
            sections.append("")

        # ── 七、综合决策 ──────────────────────────────────────
        sections.append("## 七、综合决策（Phase 5）\n")
        decision = analysis.get("decision", "")
        sections.append(decision if decision else "*待AI综合分析后生成*\n")

        # ── 八、图表 ──────────────────────────────────────────
        if charts:
            sections.append("## 八、图表\n")
            labels = {"price": "价格走势 & 200日均线", "valuation": "估值历史 (PE/PB分位)",
                      "financials": "财务趋势 (营收增速 & 毛利率)"}
            for key, path in charts.items():
                if path:
                    rel = os.path.relpath(path, REPORT_CONFIG["output_dir"])
                    sections.append(f"### {labels.get(key, key)}\n")
                    sections.append(f"![{labels.get(key, key)}]({rel})\n")

        # ── 九、历史回测 ──────────────────────────────────────
        if backtest_text:
            sections.append("## 九、历史回测（超跌信号验证）\n")
            sections.append("> 回测说明: 统计历史上满足「跌幅 >50%」条件时买入的后续收益分布，验证信号有效性\n")
            sections.append(backtest_text)
            sections.append("")

        # ── 附录 ──────────────────────────────────────────────
        sections.append("## 附录: 基础数据\n")

        basic = analysis.get("basic_info", {})
        if basic:
            for k, v in basic.items():
                if k != "error":
                    sections.append(f"- **{k}**: {v}")
            sections.append("")

        sections.append("---")
        sections.append(
            "*本报告由投资分析工作流 v2.0 自动生成，"
            "融合逻辑驱动型三层框架（宏观定轨 + ETF轮动 + Alpha挖掘），仅供个人参考。*"
        )

        return "\n".join(sections)

    def save(self, content: str, stock_name: str, symbol: str) -> str:
        output_dir = REPORT_CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        date_str  = datetime.now().strftime("%Y%m%d")
        filename  = f"{symbol}_{stock_name}_{date_str}.md"
        filepath  = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath
