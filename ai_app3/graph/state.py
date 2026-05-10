from typing import TypedDict, Any


class RagState(TypedDict):
    """
    Agentic RAG 全局状态定义。

    字段说明：
        user_message: 本轮用户输入
        intent: 意图分析结果（技术问答 / 闲聊 / 澄清 / 多步推理）
        sub_queries: 子查询列表（查询分解产物）
        retrieved_context: 最终合并的检索上下文
        kg_context: 知识图谱补充上下文
        confidence: 检索质量置信度（0~1）
        retrieval_iterations: 已执行的检索-评估-改写轮数
        history: 原始对话历史（user/assistant 的 dict 列表）
        summary: 历史摘要（token 预算耗尽时生成）
        token_budget: 剩余 token 预算（阈值）
        messages: LangChain Message 对象列表（每轮由 build_messages_node 重建）
        reply: AI 本轮最终回复文本
        trimmed: 被裁剪的旧消息
        trace: Agentic 执行轨迹（供前端展示）
        needs_tool: 是否需要执行工具调用
        tool_results: 工具执行结果列表
    """
    user_message: str
    intent: dict | None
    sub_queries: list[dict]
    retrieved_context: str | None
    kg_context: str | None
    confidence: float
    evaluation_result: dict | None
    retrieval_iterations: int
    history: list
    summary: str
    token_budget: int
    messages: list
    reply: str
    trimmed: list
    trace: list[dict]
    needs_tool: bool
    tool_results: list[Any]
