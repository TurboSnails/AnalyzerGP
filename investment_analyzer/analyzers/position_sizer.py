"""
仓位管理器: 基于框架固定规则计算建议仓位

规则:
  - 单标的硬性上限: 10%
  - Alpha层正常总上限: 25%（熊市压缩至 12%）
  - 确信度决定靠近上限还是下限
  - 退出触发器: 逻辑失效（非价格跌幅）
  - 时间止损: 18/24个月重评/减仓

不使用凯利公式计算具体数值（输入参数无法客观量化）。
"""
from typing import Dict, Any
from config import POSITION_CONFIG, LAYER2_CONFIG


class PositionSizer:

    TIME_STOP = {
        "reassess_months": LAYER2_CONFIG["time_stop_reassess_months"],
        "reduce_months":   LAYER2_CONFIG["time_stop_reduce_months"],
    }

    EXIT_TRIGGERS = [
        "营收增速连续2季度减速",
        "毛利率持续下滑（连续2季度）",
        "核心研发/业务负责人离职",
        "Rule of 40 跌破 30",
        "逻辑假设被证伪（如技术路线被替代）",
    ]

    def __init__(self):
        self.cfg = POSITION_CONFIG

    def calculate(
        self,
        analysis: Dict[str, Any],
        macro_regime: str = "ambiguous",
    ) -> Dict[str, Any]:
        """
        Args:
            analysis: 综合分析结果 (含 drop_check, reversal_check, fundamental_screen)
            macro_regime: Layer 0 输出的市场状态

        Returns:
            suggested_position: 建议首批仓位占总资产比例
            max_single:         单标的硬性上限
            total_alpha_cap:    当前宏观环境下 Alpha 层总上限
            position_tier:      高/中/低确信 or 观望
            constraints:        执行纪律列表
            exit_triggers:      逻辑退出条件
        """
        # Alpha 总上限随宏观收缩
        total_cap = (
            self.cfg["alpha_max_total_bear"]
            if macro_regime in ("rate_hike_bear", "recession_bear")
            else self.cfg["alpha_max_total_normal"]
        )

        confidence   = self._calc_confidence(analysis)
        position, tier = self._position_from_confidence(confidence)
        position     = min(position, self.cfg["alpha_max_single"])

        rationale = self._build_rationale(macro_regime, confidence, tier, total_cap)

        return {
            "suggested_position": round(position, 3),
            "max_single":         self.cfg["alpha_max_single"],
            "total_alpha_cap":    total_cap,
            "position_tier":      tier,
            "confidence":         round(confidence, 2),
            "rationale":          rationale,
            "constraints":        self._constraints(macro_regime),
            "exit_triggers":      self.EXIT_TRIGGERS,
            "time_stop":          self.TIME_STOP,
            "summary":            self._summary(position, tier, total_cap, macro_regime),
        }

    # ─── Confidence Scoring ───────────────────────────────────

    def _calc_confidence(self, analysis: Dict) -> float:
        scores = []

        drop = analysis.get("drop_check", {})
        if drop:
            if drop.get("passed"):
                scores.append(1.0)
            else:
                drop_score = drop.get("score", 0) / max(drop.get("total", 3), 1)
                scores.append(drop_score * 0.5)

        reversal = analysis.get("reversal_check", {})
        if reversal:
            ratio = reversal.get("score", 0) / max(reversal.get("total", 7), 1)
            scores.append(ratio)

        fundamental = analysis.get("fundamental_screen", {})
        if fundamental:
            if fundamental.get("passed"):
                scores.append(0.9 if fundamental.get("recommended_type") == "B" else 0.7)
            else:
                scores.append(0.15)

        return sum(scores) / len(scores) if scores else 0.30

    def _position_from_confidence(self, confidence: float):
        cfg = self.cfg
        if confidence >= cfg["confidence_high"]:
            return cfg["alpha_max_single"],       "高确信"
        if confidence >= cfg["confidence_mid"]:
            return cfg["alpha_max_single"] * 0.7, "中确信"
        if confidence >= cfg["confidence_low"]:
            return cfg["alpha_max_single"] * 0.5, "低确信"
        return 0.0, "观望"

    # ─── Formatting ───────────────────────────────────────────

    def _build_rationale(
        self, regime: str, confidence: float, tier: str, total_cap: float
    ) -> list:
        lines = [
            f"综合确信度: {confidence:.0%} → {tier}档",
            f"Alpha层总上限: {total_cap:.0%}"
            + (" (宏观压缩)" if total_cap < self.cfg["alpha_max_total_normal"] else ""),
        ]
        if regime == "bear_bottom":
            lines.append("熊市底部宏观：好公司此时更便宜，Alpha仓位正常操作")
        elif regime in ("rate_hike_bear", "recession_bear"):
            lines.append("熊市中段：暂停新开Alpha仓位或减半上限，先完成ETF轮动")
        return lines

    def _constraints(self, macro_regime: str) -> list:
        base = [
            f"单标的硬性上限: {self.cfg['alpha_max_single']:.0%}（含浮盈后需再平衡）",
            "分批建仓：首批≤50%目标仓位，逻辑持续验证后追加",
            "退出依据：逻辑失效立即清仓，不以价格涨跌作为主要退出信号",
            f"时间止损：{self.TIME_STOP['reassess_months']}个月逻辑成立但市场无反应→重评；"
            f"{self.TIME_STOP['reduce_months']}个月→减至1/2仓",
        ]
        if macro_regime in ("rate_hike_bear", "recession_bear"):
            base.insert(0, "当前宏观：先完成Layer1 ETF轮动，再考虑Alpha仓位")
        return base

    def _summary(
        self, position: float, tier: str, total_cap: float, macro_regime: str
    ) -> str:
        if position == 0:
            return "建议观望，确信度不足，不开新仓"
        macro_note = {
            "bear_bottom":    "（熊市底部，可正常操作）",
            "recession_bear": "（衰退熊市，Alpha上限压缩）",
            "rate_hike_bear": "（加息熊市，Alpha上限压缩）",
            "ambiguous":      "",
        }.get(macro_regime, "")
        return (
            f"建议首批仓位: {position:.0%} ({tier}){macro_note} | "
            f"Alpha层总上限: {total_cap:.0%}"
        )
