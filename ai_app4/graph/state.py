"""
ai_app4 Wealth AI Agent 状态定义。

基于需求文档「全球资产与宏观经济多步推演智能助理」设计，
在 ai_app3 RagState 基础上扩展投资分析专属字段：
  - sub_queries / rewritten_queries：本地 Qwen 拆解与改写产物
  - top_ce：CrossEncoder 精排最高分（自旋锁反思的核心阈值依据）
  - needs_tool / tool_calls / tool_results：策略计算工具调用链路
  - math_result：结构化数学计算结果
  - kg_context：知识图谱补充上下文（预留）
"""
from typing import TypedDict, Any


class WealthState(TypedDict):
    """Wealth AI Agent 全局状态定义。

    所有节点共享同一状态对象，节点只返回需要更新的字段字典。
    LangGraph MemorySaver 按 thread_id（即 user_id）持久化完整状态。
    """

    # ── 用户输入 ─────────────────────────────────────────────────────────────
    user_message: str                    # 本轮用户输入（可能被 reflection 改写）
    user_id: str                         # 用户标识（用于 MemorySaver thread_id）

    # ── 查询分析（analyze_and_route 节点写入）─────────────────────────────────
    time_sensitive: bool                 # 是否包含时效性关键词（"今天"、"最新"、"实时"等）
    entities: list[dict]                 # NER 实体提取结果
    #   每项格式: {"type": "ticker|macro_indicator|date", "value": "...", "confidence": 0.95}

    # ── 子查询与改写 ─────────────────────────────────────────────────────────
    sub_queries: list[dict]              # 拆解后的子查询列表
    #   每项格式: {"text": "...", "domain": "macro_econ|corp_earnings|all", "weight": 1.0}
    rewritten_queries: list[str]         # 改写后的查询文本列表（用于 reflection 循环）

    # ── 检索（parallel_retrieval 节点写入）───────────────────────────────────
    retrieved_context: str | None        # 合并后的检索上下文文本
    kg_context: str | None               # 知识图谱补充上下文（预留）
    retrieval_iterations: int            # 已执行的检索-评估轮数（防无限循环）

    # ── 评估（evaluate_and_rerank 节点写入）──────────────────────────────────
    confidence: float                    # 综合检索置信度（0.0~1.0）
    top_ce: float                        # CrossEncoder 最高分（核心阈值依据）
    evaluation_result: dict | None       # 评估详细结果

    # ── 策略推演（strategy_reasoning 节点写入）───────────────────────────────
    needs_tool: bool                     # 是否需要执行数学计算工具
    tool_calls: list[dict]               # 待执行的工具调用描述
    #   每项格式: {"tool_name": "...", "arguments": {...}}

    # ── 工具执行（execute_math_tool 节点写入）────────────────────────────────
    tool_results: list[Any]              # 工具执行原始结果列表
    math_result: dict | None             # 结构化数学结果
    #   格式示例: {"tool_name": "kelly_criterion_calc", "result": {...}}

    # ── 生成（merge_and_generate / generate 节点写入）────────────────────────
    reply: str                           # 最终回复文本

    # ── 会话历史 ─────────────────────────────────────────────────────────────
    history: list[dict]                  # 对话历史（user/assistant 的 dict 列表）
    summary: str                         # 历史摘要（token 预算耗尽时生成）
    token_budget: int                    # 剩余 token 预算
    messages: list                       # LangChain / OpenAI 格式消息列表
    trimmed: list                        # 被裁剪的旧消息

    # ── 可观测性 ─────────────────────────────────────────────────────────────
    trace: list[dict]                    # 全链路执行轨迹
    #   每项格式: {"node": "...", "latency_ms": 123.4, ...}
