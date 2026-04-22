"""
Phase 2 第三关: 护城河验证
六维度评分，量化维度自动计算，定性维度标记供AI辅助

六维度:
  1. 定价权       - 毛利率稳定/上升（定量）
  2. 成本优势     - 三费占营收比下降（定量）
  3. 规模效应     - 营收增速+毛利同步改善（定量）
  4. 无形资产     - 研发/品牌投入占比（定量代理）
  5. 转换成本     - 用户留存/复购代理（待AI）
  6. 网络效应     - 平台/用户增长逻辑（待AI）

评分: 每维度 positive=1 / negative=0 / pending=0
通过阈值: 4/6 维度为正面
"""
import pandas as pd
from typing import Dict, Any, Optional, List
from config import ANALYSIS_CONFIG


class MoatChecker:

    DIMENSIONS = [
        "pricing_power",
        "cost_advantage",
        "scale_economy",
        "intangible_assets",
        "switching_costs",
        "network_effects",
    ]

    PASS_THRESHOLD = 3   # 6维度中至少3项正面（量化3项均为可计算上限）

    def __init__(self):
        self.moat_min_score = ANALYSIS_CONFIG.get("moat_min_score", 3)

    def analyze(self, financial_data: Dict, basic_info: Dict = None) -> Dict[str, Any]:
        """
        执行护城河六维度分析

        Args:
            financial_data: 财务数据字典 (indicators / income / balance)
            basic_info:     基本信息（行业、公司描述等）

        Returns:
            score:      正面维度数量
            total:      总维度数
            passed:     是否通过
            dimensions: 各维度详情
            moat_type:  护城河类型标签
        """
        ind = financial_data.get("indicators", pd.DataFrame())
        bal = financial_data.get("balance",    pd.DataFrame())
        inc = financial_data.get("income",     pd.DataFrame())

        checks = {
            "pricing_power":    self._check_pricing_power(ind, inc),
            "cost_advantage":   self._check_cost_advantage(ind),
            "scale_economy":    self._check_scale_economy(ind),
            "intangible_assets": self._check_intangible_assets(ind, inc),
            "switching_costs":  self._check_switching_costs(ind, basic_info),
            "network_effects":  self._check_network_effects(ind, basic_info),
        }

        score = sum(1 for v in checks.values() if v["status"] == "positive")
        passed = score >= self.PASS_THRESHOLD

        return {
            "passed":      passed,
            "score":       score,
            "total":       6,
            "threshold":   self.PASS_THRESHOLD,
            "dimensions":  checks,
            "moat_type":   self._classify_moat(checks),
            "signals":     [f"✅ {v['label']}: {v['detail']}" for v in checks.values() if v["status"] == "positive"],
            "warnings":    [f"❌ {v['label']}: {v['detail']}" for v in checks.values() if v["status"] == "negative"],
            "pending":     [f"⏳ {v['label']}: {v['detail']}" for v in checks.values() if v["status"] == "pending"],
            "summary":     self._build_summary(score, passed, checks),
        }

    # ─── 六维度检查 ────────────────────────────────────────────

    def _check_pricing_power(self, ind: pd.DataFrame, inc: pd.DataFrame) -> Dict:
        """定价权: 毛利率连续2期稳定或上升"""
        gm_col = self._find_col(ind, ["销售毛利率", "毛利率"])
        if gm_col is None:
            return self._dim("定价权", "pending", "需要连续毛利率数据")

        series = ind[gm_col].dropna().head(4)
        if len(series) < 2:
            return self._dim("定价权", "pending", "数据期数不足")

        vals = self._normalize_pct(series)
        latest, prev = float(vals.iloc[0]), float(vals.iloc[1])

        if latest >= 0.40 and latest >= prev - 0.01:
            return self._dim("定价权", "positive",
                             f"高毛利率 {latest:.1%} 且稳定，具备定价权")
        if latest >= 0.25 and latest >= prev - 0.02:
            return self._dim("定价权", "positive",
                             f"毛利率 {latest:.1%}，维持稳定")
        if latest < prev - 0.03:
            return self._dim("定价权", "negative",
                             f"毛利率下滑 {latest:.1%}←{prev:.1%}，定价权受损")
        return self._dim("定价权", "pending",
                        f"毛利率 {latest:.1%}，趋势不明确")

    def _check_cost_advantage(self, ind: pd.DataFrame) -> Dict:
        """成本优势: 三费（销售+管理+研发）占营收比下降"""
        # 尝试直接获取费用率
        expense_cols = ["销售费用率", "管理费用率", "研发费用率",
                        "三费占比", "期间费用率"]
        total_expense_col = self._find_col(ind, ["期间费用率", "三费占营收比"])

        if total_expense_col:
            series = ind[total_expense_col].dropna().head(4)
            if len(series) >= 2:
                vals  = self._normalize_pct(series)
                latest, prev = float(vals.iloc[0]), float(vals.iloc[1])
                declining = latest <= prev * 1.02
                return self._dim(
                    "成本优势",
                    "positive" if declining else "negative",
                    f"费用率 {latest:.1%}（{'下降/稳定' if declining else '上升'}，前期 {prev:.1%}）",
                )

        # 用毛利率 - 净利率 代理运营效率
        gm  = self._latest_pct(ind, ["销售毛利率", "毛利率"])
        nm  = self._latest_pct(ind, ["净利率", "销售净利率", "净利润率"])
        if gm is not None and nm is not None:
            spread = gm - nm
            if spread < 0.15:
                return self._dim("成本优势", "positive",
                                f"毛利-净利差 {spread:.1%}，费用管控良好")
            return self._dim("成本优势", "negative",
                            f"毛利-净利差 {spread:.1%}，费用较高")

        return self._dim("成本优势", "pending", "需要费用明细数据")

    def _check_scale_economy(self, ind: pd.DataFrame) -> Dict:
        """规模效应: 营收增长同时毛利率改善"""
        rev_growth = self._latest_pct(ind, ["营业总收入增速", "营收增速", "营业收入增速"])
        gm_col     = self._find_col(ind, ["销售毛利率", "毛利率"])

        if rev_growth is None:
            return self._dim("规模效应", "pending", "需要营收增速数据")

        gm_improving = False
        if gm_col is not None:
            series = ind[gm_col].dropna().head(3)
            if len(series) >= 2:
                vals = self._normalize_pct(series)
                gm_improving = float(vals.iloc[0]) >= float(vals.iloc[1]) - 0.01

        if rev_growth > 0.15 and gm_improving:
            return self._dim("规模效应", "positive",
                            f"营收增速 {rev_growth:.1%}，毛利率同步改善，规模效应显现")
        if rev_growth > 0.10:
            return self._dim("规模效应", "positive",
                            f"营收增速 {rev_growth:.1%}，具备一定规模成长性")
        if rev_growth < 0:
            return self._dim("规模效应", "negative",
                            f"营收负增长 {rev_growth:.1%}，规模效应减弱")
        return self._dim("规模效应", "pending",
                        f"营收增速 {rev_growth:.1%}，规模效应不明显")

    def _check_intangible_assets(self, ind: pd.DataFrame, inc: pd.DataFrame) -> Dict:
        """无形资产: 研发投入占营收比作为技术护城河代理"""
        rd_rate = self._latest_pct(ind, ["研发费用率", "研发费用/营业收入"])

        if rd_rate is None:
            # 尝试从利润表计算
            rd_abs  = self._latest(inc, ["研究费用", "研发费用", "研发支出",
                                         "Research And Development"])
            rev_abs = self._latest(ind, ["营业总收入", "营业收入"])
            if rd_abs is not None and rev_abs is not None and rev_abs > 0:
                rd_rate = rd_abs / rev_abs

        if rd_rate is not None:
            if rd_rate >= 0.10:
                return self._dim("无形资产", "positive",
                                f"研发投入占比 {rd_rate:.1%}，技术护城河较深")
            if rd_rate >= 0.05:
                return self._dim("无形资产", "positive",
                                f"研发投入占比 {rd_rate:.1%}，持续投入研发")
            return self._dim("无形资产", "negative",
                            f"研发投入占比 {rd_rate:.1%}，技术壁垒有限")

        # 无研发数据时，尝试判断是否为品牌型企业（毛利率高代理品牌溢价）
        gm = self._latest_pct(ind, ["销售毛利率", "毛利率"])
        if gm is not None and gm >= 0.50:
            return self._dim("无形资产", "positive",
                            f"毛利率 {gm:.1%}，可能具备品牌/专利溢价")

        return self._dim("无形资产", "pending",
                        "需要研发费用明细数据，将由AI判断品牌/专利护城河")

    def _check_switching_costs(self, ind: pd.DataFrame, basic_info: Optional[Dict]) -> Dict:
        """转换成本: 定量代理 + 标记AI判断"""
        # 代理指标：应收账款周转天数短 = 客户依赖性强
        ar_days = self._latest(ind, ["应收账款周转天数", "应收账款周转率"])
        if ar_days is not None:
            # 周转天数越短越好（付款快 = 定价权强）
            # 周转率越高越好
            col_name = self._find_col(ind, ["应收账款周转天数"])
            if col_name:
                days = ar_days
                if days < 30:
                    return self._dim("转换成本", "positive",
                                    f"应收账款周转天数仅 {days:.0f}天，客户黏性强")
                if days > 90:
                    return self._dim("转换成本", "negative",
                                    f"应收账款周转天数 {days:.0f}天，客户谈判能力较强")

        return self._dim("转换成本", "pending",
                        "需AI判断：用户迁移成本（合同锁定/数据迁移/习惯壁垒）")

    def _check_network_effects(self, ind: pd.DataFrame, basic_info: Optional[Dict]) -> Dict:
        """网络效应: 用户增长+留存代理，主要依赖AI"""
        # 定量代理：用户增长驱动营收加速
        rev_cols = ["营业总收入增速", "营收增速"]
        rev_col  = self._find_col(ind, rev_cols)
        if rev_col is not None:
            series = ind[rev_col].dropna().head(4)
            if len(series) >= 3:
                vals = [float(v) for v in series.values[:3]]
                # 营收加速 = 可能有网络效应
                if vals[0] > vals[1] > vals[2] and vals[0] > 20:
                    return self._dim("网络效应", "positive",
                                    f"营收持续加速（{vals[2]:.1f}%→{vals[1]:.1f}%→{vals[0]:.1f}%），"
                                    f"可能存在正向网络效应")

        return self._dim("网络效应", "pending",
                        "需AI判断：平台用户数量/活跃度增长，或B2B生态绑定深度")

    # ─── Helpers ───────────────────────────────────────────────

    def _classify_moat(self, checks: Dict) -> str:
        """根据正面维度判断护城河类型"""
        positive = [k for k, v in checks.items() if v["status"] == "positive"]
        if not positive:
            return "无明显护城河"

        type_map = {
            "pricing_power":    "定价权",
            "cost_advantage":   "成本优势",
            "scale_economy":    "规模效应",
            "intangible_assets": "无形资产(研发/品牌)",
            "switching_costs":  "转换成本",
            "network_effects":  "网络效应",
        }
        labels = [type_map[k] for k in positive if k in type_map]
        return " + ".join(labels)

    def _build_summary(self, score: int, passed: bool, checks: Dict) -> str:
        moat_type = self._classify_moat(checks)
        verdict   = "具备护城河" if passed else "护城河薄弱"
        return f"{verdict} ({score}/6 维度) | 来源: {moat_type}"

    def _find_col(self, df: pd.DataFrame, cols: List[str]) -> Optional[str]:
        if df is None or df.empty:
            return None
        return next((c for c in cols if c in df.columns), None)

    def _latest(self, df: pd.DataFrame, cols: List[str]) -> Optional[float]:
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

    def _latest_pct(self, df: pd.DataFrame, cols: List[str]) -> Optional[float]:
        val = self._latest(df, cols)
        if val is None:
            return None
        return val / 100 if abs(val) > 2 else val

    def _normalize_pct(self, series: pd.Series) -> pd.Series:
        vals = series.astype(float)
        return vals / 100 if vals.abs().max() > 2 else vals

    @staticmethod
    def _dim(label: str, status: str, detail: str) -> Dict:
        return {"label": label, "status": status, "detail": detail}
