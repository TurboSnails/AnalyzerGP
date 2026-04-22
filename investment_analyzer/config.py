"""
投资分析工作流 - 配置文件
"""
import os

# ============================================================
# LLM 配置（按优先级排列，用哪个填哪个的key）
# ============================================================
LLM_CONFIG = {
    # 推荐使用 DeepSeek（便宜且中文好）或 Claude
    "model": os.getenv("LLM_MODEL", "deepseek/deepseek-chat"),
    "api_key": os.getenv("LLM_API_KEY", ""),
    "temperature": 0.3,
    "max_tokens": 4096,
}

# 备选模型配置
# "anthropic/claude-sonnet-4-20250514"
# "openai/gpt-4o"
# "qwen/qwen-max"

# ============================================================
# 数据源配置
# ============================================================
DATA_CONFIG = {
    "cache_dir": os.path.join(os.path.dirname(__file__), "output", ".cache"),
    "cache_ttl_hours": 4,  # 缓存过期时间（小时）
}

# ============================================================
# 分析参数配置
# ============================================================
ANALYSIS_CONFIG = {
    # 第一关: 跌幅量化阈值
    "min_drop_from_high": 0.50,       # 最低跌幅要求 50%
    "min_underperform_sector": 0.20,  # 跑输行业至少 20%
    "pe_pb_percentile_max": 0.20,     # PE/PB 历史分位数上限 20%

    # 第二关: 真反转验证
    "reversal_pass_threshold": 4,     # 7维度中至少4项为正面信号

    # 第三关: 护城河
    "moat_min_score": 7,              # 至少一项 >= 7分

    # 仓位管理
    "max_position_famous": 0.20,      # 超跌知名股单只上限 20%
    "max_position_contrarian": 0.15,  # 冷门反转股单只上限 15%
    "max_position_small_cap": 0.10,   # 小盘股单只上限 10%

    # 止损
    "price_stop_loss": 0.15,          # 价格止损 15-20%
    "time_stop_months": 18,           # 时间止损 18个月

    # 止盈
    "take_profit_fib": [0.50, 0.618], # 黄金分割位止盈
}

# ============================================================
# 报告配置
# ============================================================
REPORT_CONFIG = {
    "output_dir": os.path.join(os.path.dirname(__file__), "output"),
    "language": "zh-CN",
}

# ============================================================
# Layer 0: 宏观定轨配置
# ============================================================
LAYER0_CONFIG = {
    "broad_index": "sh000300",          # 沪深300
    "ma_period": 200,                   # 200日均线
    "pe_bear_percentile": 0.30,         # PE分位 < 30% = 低估
    "pe_bull_percentile": 0.70,         # PE分位 > 70% = 高估
    "yield_trend_days": 60,             # 国债收益率趋势回看天数
    "yield_threshold_bp": 0.10,         # 超过10bp算趋势性变化
}

# ============================================================
# Layer 2: Alpha 仓位基本面筛选阈值
# ============================================================
LAYER2_CONFIG = {
    # A类: 价值/红利型
    "type_a": {
        "min_roe": 0.15,                # ROE > 15%
        "max_debt_ratio": 0.60,         # 资产负债率 < 60%
        "min_dividend_yield": 0.02,     # 股息率 > 2%
        "min_cf_growth": 0.08,          # 经营现金流增速 > 8%
        "min_profit_growth": 0.10,      # 扣非净利润增速 > 10%
        "min_gross_margin": 0.20,       # 毛利率 > 20%
        "pass_threshold": 3,            # 至少3项通过
    },
    # B类: 第二曲线成长型
    "type_b": {
        "min_rule_of_40": 40,           # 营收增速 + 净利润率 > 40
        "min_gross_margin_tech": 0.40,  # 科技公司毛利率 > 40%
        "min_gross_margin_consumer": 0.30,
        "revenue_accel_quarters": 3,    # 连续3季度营收加速
        "pass_threshold": 3,            # 至少3项通过
    },
    # 时间止损
    "time_stop_reassess_months": 18,
    "time_stop_reduce_months": 24,
}

# ============================================================
# 仓位管理配置
# ============================================================
POSITION_CONFIG = {
    "alpha_max_single": 0.10,           # 单标的上限 10%
    "alpha_max_total_normal": 0.25,     # Alpha层正常上限 25%
    "alpha_max_total_bear": 0.12,       # 熊市早期上限减半 12%
    "confidence_high": 0.80,            # 高确信度阈值
    "confidence_mid": 0.60,             # 中确信度阈值
    "confidence_low": 0.40,             # 低确信度阈值
}
