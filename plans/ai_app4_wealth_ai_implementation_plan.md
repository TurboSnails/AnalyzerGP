# ai_app4 — Wealth AI Agent v4.0 实施计划

> 基于 `doc/需求文档.md` 设计的全球资产与宏观经济多步推演智能助理实施方案。
> 版本: v2.0（已完成） | 日期: 2026-05-17

---

## 一、架构决策（多方辩论结论）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 是否保留 ai_app4 骨架 | ✅ 保留 | `main.py`、`lifespan.py`、`core/*`、`api/chat.py` 是通用 FastAPI + DI 基础设施，与业务无关 |
| 图编排层如何处理 | 🔄 彻底重写 | `graph/` 目录完全替换为 Wealth AI 的 7 节点 + 4 条件边拓扑，移除客服专属节点 |
| 是否依赖 ai_app3 | ❌ 不直接依赖 | 参考 ai_app3 的 `decompose_node`、`tool_calling` 设计模式，但代码独立编写，避免历史包袱 |
| 领域插件位置 | `domains/wealth/` | 复用 `DomainPlugin` 抽象，定义 `macro_econ` + `corp_earnings` 双域 |
| 数学工具挂载方式 | Python 硬函数 + `tool_registry` | 凯利公式、网格交易计算器注册到 `rag_framework.llm.tool_registry`，LLM 通过 JSON 调用 |

---

## 二、LangGraph 状态图拓扑

```mermaid
graph TD
    START --> ANALYZE[analyze_and_route<br/>本地Qwen查询拆解+翻译]
    ANALYZE --> RETRIEVE[parallel_retrieval<br/>Tantivy+Dense并发多域召回]
    RETRIEVE --> EVALUATE[evaluate_and_rerank<br/>CrossEncoder精排+top_ce]

    EVALUATE -->|top_ce < 0.35 & loop < 2| REFLECT[query_reflection<br/>反思改写Query]
    REFLECT --> RETRIEVE

    EVALUATE -->|top_ce >= 0.35 或 loop >= 2| STRATEGY[strategy_reasoning<br/>主模型多步推演]

    STRATEGY --> NEEDSTOOL{需要计算工具?}
    NEEDSTOOL -->|Yes| MATH[execute_math_tool<br/>Python硬计算]
    NEEDSTOOL -->|No| FINAL[Final Response<br/>直接流式输出]

    MATH --> MERGE[merge_and_generate<br/>合并数字+话术生成]
    MERGE --> FINAL
```

### 节点清单

| 节点 | 职责 | 状态写入 | 实现状态 |
|------|------|---------|---------|
| `analyze_and_route` | 本地 Qwen 提取特征、中英文 Query 拆解、生成 sub_queries | `sub_queries`, `rewritten_queries`, `trace` | ✅ 已完成 |
| `parallel_retrieval` | 并发检索 macro_econ / corp_earnings / all，Tantivy + Dense 召回 | `retrieved_context`, `retrieval_iterations`, `top_ce` | ✅ 已完成 |
| `evaluate_and_rerank` | CrossEncoder 精排，提取真实 top_ce（加权融合 confidence） | `confidence`, `top_ce`, `evaluation_result` | ✅ 已完成 |
| `query_reflection` | Qwen 反思未命中原因，金融术语化改写 + 规则降级 | `user_message`（改写后）, `trace` | ✅ 已完成 |
| `strategy_reasoning` | 主模型（MiniMax）多步推演，识别是否需要工具 | `reply`（中间思考）, `needs_tool`, `tool_calls` | ✅ 已完成 |
| `execute_math_tool` | 从 tool_call 提取参数，调用 Python 硬计算 | `tool_results`, `math_result` | ✅ 已完成 |
| `merge_and_generate` | 合并计算结果与 LLM 话术，生成严谨金融报告 | `reply`（最终回复） | ✅ 已完成 |

### 条件边清单

| 条件边 | 判断逻辑 | 分支 | 实现状态 |
|--------|---------|------|---------|
| `after_analyze` | sub_queries 数量 > 1 时走多路，否则单路 | `parallel_retrieval`（统一） | ✅ 已完成 |
| `after_evaluate` | `top_ce < 0.35` 且 `loop < max_loop` → reflection，否则 → strategy | `query_reflection` / `strategy_reasoning` | ✅ 已完成 |
| `after_strategy` | `needs_tool == True` → tool，否则 → final | `execute_math_tool` / `merge_and_generate` | ✅ 已完成 |
| `after_tool` | 工具执行后必须合并生成 | `merge_and_generate` | ✅ 已完成 |

---

## 三、四阶段实施路线图（全部完成）

### 阶段一：骨架迁移（Week 1）✅

目标：不加任何新知识，先用 LangGraph 把现有能力框起来。

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 1-1 | 重构 `graph/state.py` — `CS4State` → `WealthState` | `ai_app4/graph/state.py` | ✅ |
| 1-2 | 重构 `graph/builder.py` — 7 节点 + 4 条件边拓扑 | `ai_app4/graph/builder.py` | ✅ |
| 1-3 | 实现基础节点 — analyze/route/retrieve/generate | `ai_app4/graph/nodes.py` | ✅ |
| 1-4 | 实现条件边 — after_evaluate / after_strategy | `ai_app4/graph/conditional_edges.py` | ✅ |
| 1-5 | 调整 API — SSE 流式输出适配 | `ai_app4/api/chat.py` | ✅ |
| 1-6 | 调整配置 — WealthSettings | `ai_app4/core/config.py` | ✅ |
| 1-7 | 验证骨架 — Graph 编译 + 基础链路测试 | — | ✅ |

### 阶段二：数据录入与跨域融合（Week 2-3）✅

目标：让资料库丰富起来，打通中英文混合路由。

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 2-1 | 创建 Wealth 领域插件 — WealthDomainPlugin | `domains/wealth/wealth_domain/plugin.py` | ✅ |
| 2-2 | 编写索引构建脚本 — init_wealth_db.py | `domains/wealth/scripts/init_wealth_db.py` | ✅ |
| 2-3 | 增强查询拆解 — 中英文 Query 派发 | `ai_app4/graph/nodes.py` | ✅ |
| 2-4 | 增强并发检索 — 按 domain 路由融合 | `ai_app4/graph/nodes.py` | ✅ |
| 2-5 | 录入测试数据 — 2 份财报 + 美联储纪要 | `domains/wealth/data/` | ✅ |

### 阶段三：自旋锁反思机制（Week 4-5）✅

目标：不浪费任何线上坏 Case，卡死 RAG 底层质量。

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 3-1 | 实现评估节点 — 真实 top_ce 提取 + 加权融合 | `ai_app4/graph/nodes.py` | ✅ |
| 3-2 | 实现反思节点 — LLM 金融术语化改写 | `ai_app4/graph/nodes.py` | ✅ |
| 3-3 | 完善条件边 — top_ce < 0.35 且 loop < max | `ai_app4/graph/conditional_edges.py` | ✅ |
| 3-4 | 接入 Trace — retrieval_trace + latency_breakdown | `ai_app4/graph/nodes.py` | ✅ |

### 阶段四：挂载计算工具箱（Week 6+）✅

目标：让 Agent 具备真正的行动力，完成商业级闭环。

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 4-1 | 数学工具函数 — 4 个计算工具 | `ai_app4/service/math_tools.py` | ✅ |
| 4-2 | 注册工具 — tool_registry | `ai_app4/service/tools.py` | ✅ |
| 4-3 | 策略推演节点 — 主模型识别工具需求 | `ai_app4/graph/nodes.py` | ✅ |
| 4-4 | 工具执行节点 — 提取参数执行 | `ai_app4/graph/nodes.py` | ✅ |
| 4-5 | 合并生成节点 — 数字 + 话术合并 | `ai_app4/graph/nodes.py` | ✅ |
| 4-6 | 端到端测试 — 10 项复合测试 | `ai_app4/tests/test_phase4_nodes.py` | ✅ |

---

## 四、数学工具箱详情

| 工具名 | 功能 | 输入参数 | 输出字段 |
|--------|------|---------|---------|
| `kelly_criterion_calc` | 凯利公式最优仓位 | `win_rate`, `avg_gain_pct`, `avg_loss_pct`, `total_capital` | `optimal_fraction`, `recommended_position`, `half_kelly_position` |
| `grid_trading_cost_estimator` | 网格交易成本估算 | `lower_bound`, `upper_bound`, `num_grids`, `total_capital`, `fee_pct` | `grid_interval`, `num_trades`, `total_fee_cost` |
| `portfolio_drawdown_estimator` | 组合回撤压力测试 | `holdings`（列表）, `shock_scenario` | `weighted_drawdown`, `max_loser`, `risk_level` |
| `compound_growth_calculator` | 复利增长计算器 | `principal`, `annual_return_pct`, `years`, `monthly_contribution` | `final_value`, `total_contribution`, `total_return` |

工具注册在 `lifespan.py` 启动时通过 `register_wealth_tools()` 完成，LLM 通过 `TOOL_CALL: {...}` JSON 格式调用。

---

## 五、关键测试覆盖

| 测试文件 | 用例数 | 覆盖内容 | 状态 |
|---------|-------|---------|------|
| `test_phase2_nodes.py` | 3 | 查询拆解、并发检索、跨域融合 | ✅ 通过 |
| `test_phase3_nodes.py` | 5 | 真实 top_ce、加权 confidence、LLM 反思、条件边、端到端 reflection 循环 | ✅ 通过 |
| `test_phase4_nodes.py` | 10 | 4 个数学工具、工具注册、直接执行、TOOL_CALL 解析、execute_math_tool、merge_and_generate、after_strategy | ✅ 通过 |
| `scripts/verify_refactor.py` | 4 | 语法检查、导入检查、配置类型检查、插件接口检查 | ✅ 通过 |

---

## 六、关键风险与缓解（已实现）

| 风险 | 缓解措施 | 状态 |
|------|---------|------|
| 中英文 Query 拆解质量不稳定 | 规则 + LLM 混合，sub_queries 带 weight 字段 | ✅ 已缓解 |
| `top_ce` 阈值 0.35 可能过严/过松 | 配置化 `reflection_threshold` | ✅ 已配置 |
| 数学工具参数提取幻觉 | 严格 JSON Schema + 参数范围检查 + 嵌套 JSON 解析器 | ✅ 已缓解 |
| 跨领域检索结果融合冲突 | 按 `weight` 加权排序 + 去重 | ✅ 已缓解 |
| 索引为空时系统崩溃 | 所有检索节点增加空结果兜底，v2 collections 缺失时回退 legacy path | ✅ 已缓解 |

---

## 七、 WealthState 字段定义（最终版）

```python
class WealthState(TypedDict):
    # 用户输入
    user_message: str
    user_id: str

    # 分析与改写（analyze_and_route 写入）
    sub_queries: list[dict]           # [{"text": "...", "domain": "macro_econ", "weight": 1.0}]
    rewritten_queries: list[str]      # 改写后的查询列表

    # 检索（parallel_retrieval 写入）
    retrieved_context: str | None     # 合并后的检索上下文
    retrieval_iterations: int         # 已检索次数（防无限循环）

    # 评估（evaluate_and_rerank 写入）
    confidence: float                 # 综合置信度（0.6 * top_ce + 0.4 * base_conf）
    top_ce: float                     # CrossEncoder 最高分（核心阈值依据）
    evaluation_result: dict | None    # 评估详情（含 branches/rerank_docs/latency）

    # 策略推演（strategy_reasoning 写入）
    needs_tool: bool                  # 是否需要执行数学工具
    tool_calls: list[dict]            # 待执行的工具调用描述 [{"name": "...", "arguments": {...}}]

    # 工具执行（execute_math_tool 写入）
    tool_results: list[Any]           # 工具执行原始结果
    math_result: dict | None          # 结构化数学结果

    # 生成（merge_and_generate / generate 写入）
    reply: str                        # 最终回复文本

    # 会话历史
    history: list[dict]               # 对话历史
    summary: str                      # 历史摘要
    token_budget: int                 # 剩余 token 预算
    messages: list                    # LangChain Message 对象

    # 可观测性
    trace: list[dict]                 # 全链路执行轨迹
```

---

## 八、启动命令

```bash
# 构建 Wealth 领域索引（首次运行必需）
uv run python -m domains.wealth.scripts.init_wealth_db

# 启动 ai_app4（端口 8004，需在 .env 中配置或确认）
uv run python -m uvicorn ai_app4.main:app --host 0.0.0.0 --port 8004

# 运行全部测试
uv run python -m ai_app4.tests.test_phase2_nodes
uv run python -m ai_app4.tests.test_phase3_nodes
uv run python -m ai_app4.tests.test_phase4_nodes

# 冒烟验证
uv run python scripts/verify_refactor.py
```

---

*本计划已全部实施完成。ai_app4 Wealth AI Agent v4.0 具备完整的 RAG 检索、反思自纠错、数学工具调用能力，可作为全球资产与宏观经济分析的生产级智能助理运行。*
