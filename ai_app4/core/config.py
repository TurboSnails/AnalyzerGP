"""
ai_app4 配置系统。

继承 RAGSettings，新增客服系统专属配置段。
ai_app4 使用端口 8004，配置前缀仍为 RAG_（与框架保持一致），
新增环境变量前缀 CS4_ 用于客服专属字段。
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from rag_framework.core.config import RAGSettings


class CS4Settings(RAGSettings):
    """
    ai_app4 商业化客服系统配置。

    所有 RAGSettings 字段均可通过 RAG_* 环境变量覆盖，
    新增字段通过 CS4_* 环境变量覆盖。
    """

    model_config = SettingsConfigDict(
        env_file=[".env", "ai_app4/.env"],
        env_file_encoding="utf-8",
        env_prefix="CS4_",
        extra="ignore",
    )

    # ── 客服系统核心行为 ─────────────────────────────────────────────────────
    enable_escalation: bool = True           # 是否启用转人工判断
    escalation_sentiment_threshold: str = "negative"  # 触发转人工的情感阈值
    escalation_intents: list[str] = Field(
        default_factory=lambda: ["escalation_request", "complaint"]
    )
    enable_agent_handoff: bool = False       # 是否启用 WebSocket 坐席桥接
    enable_torch_models: bool = True         # 是否启用 PyTorch 任务模型

    # ── 会话与持久化 ─────────────────────────────────────────────────────────
    session_ttl_seconds: int = 3600          # 会话过期时间
    max_history_per_session: int = 20        # 单会话最大消息数

    # ── LLM 生成参数（客服风格） ─────────────────────────────────────────────
    cs_system_prompt: str = (
        "你是专业的客服助手，态度友好、耐心细致。"
        "回答基于检索到的知识库内容，如信息不足请诚实告知用户。"
        "遇到投诉或情绪激动用户时，先安抚情绪再提供解决方案。"
    )
