"""
ai_app4 配置系统。

继承 RAGSettings，新增 Wealth AI Agent 专属配置段。
ai_app4 使用端口 8004，配置前缀仍为 RAG_（与框架保持一致），
新增环境变量前缀 WEALTH_ 用于 Wealth AI 专属字段。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from rag_framework.core.config import RAGSettings


class WealthSettings(RAGSettings):
    """
    Wealth AI Agent 配置。

    所有 RAGSettings 字段均可通过 RAG_* 环境变量覆盖，
    新增字段通过 WEALTH_* 环境变量覆盖。
    """

    model_config = SettingsConfigDict(
        env_file=[".env", "ai_app4/.env"],
        env_file_encoding="utf-8",
        env_prefix="WEALTH_",
        extra="ignore",
    )

    # ── 覆盖基类默认值 ───────────────────────────────────────────────────────
    active_domain: str = "wealth"  # 覆盖 RAGSettings 的 "android"

    # ── Wealth AI 核心行为 ───────────────────────────────────────────────────
    math_tool_enabled: bool = True               # 是否启用数学计算工具箱
    reflection_threshold: float = 0.35           # 触发 query_reflection 的 top_ce 阈值
    max_loop_count: int = 2                      # 检索-反思最大循环次数

    # ── 查询改写与拆解 ───────────────────────────────────────────────────────
    enable_query_decomposition: bool = True      # 是否启用多路子查询拆解
    enable_query_translation: bool = True        # 是否启用中英文 Query 翻译
    max_sub_queries: int = 4                     # 单轮最大子查询数

    # ── 检索参数 ─────────────────────────────────────────────────────────────
    retriever_top_k: int = 10                    # 单路检索返回文档数
    cross_encoder_top_k: int = 5                 # CrossEncoder 精排后截取的文档数

    # ── LLM 生成参数（投资分析风格）──────────────────────────────────────────
    wealth_system_prompt: str = (
        "你是一位专业的全球资产配置分析师，精通美股科技股、A股/港股、"
        "宏观经济（美联储政策、CPI/PPI、VIX）以及量化交易策略（网格交易、凯利公式）。"
        "回答基于检索到的财报和宏观数据，严禁凭空编造数字。"
        "涉及仓位计算时，必须调用工具函数得出精确结果，不可口算。"
    )

    # ── 会话与历史 ───────────────────────────────────────────────────────────
    max_history_per_session: int = 20            # 单会话最大消息数
    default_token_budget: int = 4096             # 默认 token 预算

    # ── 可观测性 ─────────────────────────────────────────────────────────────
    enable_trace: bool = True                    # 是否输出全链路 trace
    enable_latency_breakdown: bool = True        # 是否输出 PhaseTimer 延迟拆解
