from typing import TypedDict


class RagState(TypedDict):
    """LangGraph 全局状态定义

    字段说明：
        user_message: 本轮用户输入
        history: 原始对话历史（user/assistant 的 dict 列表）
        summary: 历史摘要（token 预算耗尽时生成）
        token_budget: 剩余 token 预算（阈值，非递减计数器）
        retrieved_context: 混合检索结果（参考资料文本）
        messages: LangChain Message 对象列表（每轮由 build_messages_node 重建）
        reply: AI 本轮最终回复文本
        trimmed: 被裁剪的旧消息（不丢弃，保留用于回溯）
    """
    user_message: str
    history: list
    summary: str
    token_budget: int
    retrieved_context: str | None
    messages: list
    reply: str
    trimmed: list
