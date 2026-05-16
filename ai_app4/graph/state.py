"""
ai_app4 LangGraph 状态定义。

在 ai_app3 RagState 基础上扩展商业化客服专属字段：
  - intent / sentiment / entities：PyTorch 模型推理结果
  - escalation_triggered：是否已触发转人工
  - agent_id：当前接管的坐席 ID
  - tenant_id：多租户标识
"""
from typing import TypedDict, Any


class CS4State(TypedDict):
    """ai_app4 Agentic 客服状态定义。"""

    # ── 用户输入 ─────────────────────────────────────────────────────────────
    user_message: str
    tenant_id: str
    user_id: str

    # ── PyTorch 模型推理结果 ─────────────────────────────────────────────────
    intent: str                              # 意图分类结果
    intent_score: float
    sentiment: str                           # 情感分析结果
    sentiment_score: float
    entities: list[dict]                     # NER 实体列表

    # ── 检索与生成 ───────────────────────────────────────────────────────────
    sub_queries: list[dict]
    retrieved_context: str | None
    kg_context: str | None
    confidence: float
    evaluation_result: dict | None
    retrieval_iterations: int

    # ── 会话历史 ─────────────────────────────────────────────────────────────
    history: list
    summary: str
    token_budget: int
    messages: list
    reply: str
    trimmed: list

    # ── 客服专属 ─────────────────────────────────────────────────────────────
    escalation_triggered: bool
    escalation_reason: str
    agent_id: str | None
    trace: list[dict]
    needs_tool: bool
    tool_results: list[Any]
