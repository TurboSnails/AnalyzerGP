# ai_app4 - Wealth AI Agent 全球资产与宏观经济多步推演智能助理

> 版本: 1.0 | 最后更新: 2026-05-17

---

## 1. 系统定位

ai_app4 是面向 **全球资产配置与宏观经济分析** 的商业级 Agentic RAG 智能助理，基于 LangGraph `StateGraph` 状态图编排构建。系统在完全复用 `rag_framework` 检索管道（Dense + HyDE + BM25 + RRF + Rerank）的基础上，引入面向金融投资领域的核心增强能力：

- **三轨融合检索**：本地 RAG 知识库 + 实时金融 API + 网络搜索，按查询特征动态启用
- **查询分析与路由**：时效性检测、NER 实体提取、中英文 Query 翻译、子查询拆解
- **自旋锁反思（Self-RAG）**：检索质量评估 → 金融术语化改写 → 循环检索（最多 2 次）
- **策略推演与工具调用**：凯利公式、网格交易、组合回撤、复利计算等纯 Python 硬计算
- **商业级增强**：数据来源标注、投资合规免责声明、API 配额管理、内存缓存层

运行端口：**8004**（与 ai_app1:8000、ai_app2:8001、ai_app3:8002 独立共存）。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户请求层                                   │
│                    POST /chat  { "message": "..." }                      │
│                    SSE 流式响应  trace → content → done                  │
│                    POST /chat/json  — 非流式 JSON 响应                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI 应用层 (main.py)                          │
│  - 挂载 chat_router（端口 8004）                                         │
│  - 静态文件服务 /ui                                                      │
│  - lifespan: 注册 WealthDomainPlugin → 构建 WealthContainer              │
│              → 初始化三轨检索器 → 注册数学工具 → 预热模型                │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       Chat API 路由层 (api/chat.py)                       │
│  1. 接收请求 → 从 checkpointer 加载线程状态                               │
│  2. 组装 input_state（WealthState，含 history/summary/token_budget）     │
│  3. 调用 graph.ainvoke 执行完整 Wealth AI Graph                           │
│  4. SSE 推送: trace → content(逐字) → done                               │
│     Graph 内部：                                                         │
│     analyze_and_route → parallel_retrieval → evaluate_and_rerank        │
│     → [query_reflection → parallel_retrieval] 循环                       │
│     → strategy_reasoning → [execute_math_tool → merge_and_generate]     │
│     → generate_final → END                                               │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Wealth AI LangGraph 状态图 (graph/)                      │
│                                                                          │
│   START ──→ analyze_and_route ──→ parallel_retrieval ──→ evaluate_and_rerank
│                                                                            │
│                                          [Conditional Edge: after_evaluate]│
│                                                                            │
│                     ┌────────────────────┴────────────────────┐          │
│                     │ (top_ce < threshold & loop < max)       │ (top_ce >= threshold 或 loop >= max)
│                     ▼                                         ▼          │
│           query_reflection ─────→ parallel_retrieval    strategy_reasoning│
│                     ↑                                         │          │
│                     │            [Conditional Edge: after_strategy]      │
│                     │                                         │          │
│                     │              ┌──────────┴──────────┐    │          │
│                     │         (needs_tool)          (纯文本) │          │
│                     │              ▼                      ▼  │          │
│                     │    execute_math_tool        generate_final         │
│                     │              │                      │  │          │
│                     │              ▼                      │  │          │
│                     │    merge_and_generate ──────────────┘  │          │
│                     │              │                         │          │
│                     │              ▼                         │          │
│                     └─────────── END ◄───────────────────────┘          │
│                                                                          │
│  节点说明：                                                               │
│  - analyze_and_route    : 时效性检测 + NER + 领域分类 + 子查询拆解       │
│  - parallel_retrieval   : 三轨融合检索 或 HybridRetriever 本地检索       │
│  - evaluate_and_rerank  : CrossEncoder top_ce 提取 + 置信度融合          │
│  - query_reflection     : LLM 驱动金融术语化改写 或 规则降级改写         │
│  - strategy_reasoning   : 主模型多步推演，识别 TOOL_CALL 指令            │
│  - execute_math_tool    : 调用 tool_registry 执行硬计算函数              │
│  - merge_and_generate   : 合并计算结果与 LLM 话术，生成严谨金融报告      │
│  - generate_final       : 纯文本最终回复（无需工具时）                   │
│                                                                          │
│  条件边：                                                                 │
│  - after_evaluate : top_ce < threshold & loop < max → reflection        │
│                     否则 → strategy                                      │
│  - after_strategy : needs_tool → tool | 否则 → final                   │
│                                                                          │
│  状态管理：MemorySaver（内存 checkpointer，按 thread_id 持久化）         │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Wealth AI 专属服务层 (service/)                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ three_track_     │  │ math_tools       │  │ datasources      │       │
│  │ retriever        │  │ 凯利公式/网格交易 │  │ yahoo_finance    │       │
│  │ 三轨融合检索     │  │ 组合回撤/复利     │  │ fred_api         │       │
│  │ 权重融合+来源标注│  │ 纯 Python 硬计算 │  │ tavily_search    │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│  ┌──────────────────┐  ┌──────────────────┐                             │
│  │ cache            │  │ quota            │                             │
│  │ MemoryCache      │  │ QuotaManager     │                             │
│  │ TTL + asyncio.Lock│  │ 按租户/用户/数据源│                             │
│  └──────────────────┘  └──────────────────┘                             │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               rag_framework 混合检索管道 + 数据源抽象层                      │
│                                                                          │
│   rag_framework/                                                         │
│   ├── retrieval/  HybridRetriever  Dense+HyDE+BM25→RRF→Rerank          │
│   ├── datasource/  DataSource 抽象基类 + FetchContext                    │
│   ├── container.py  RAGContainer（WealthContainer 继承扩展）             │
│   └── llm/  OpenAILLMClient / tool_registry                             │
│                                                                          │
│   domains/wealth/  WealthDomainPlugin（领域分类、术语映射、HyDE Prompt）  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.1 目录结构

```
ai_app4/
├── main.py                 # FastAPI 入口，端口 8004
├── lifespan.py             # 生命周期：容器 → 三轨检索器 → 缓存 → 配额 → 工具注册 → 预热
├── api/
│   └── chat.py             # /chat SSE 流式 + /chat/json 非流式
├── core/
│   ├── config.py           # WealthSettings（继承 RAGSettings）
│   ├── container.py        # WealthContainer（继承 RAGContainer）
│   └── context.py          # 全局上下文（避免 nodes.py 循环导入 main.py）
├── graph/
│   ├── builder.py          # StateGraph 构建 + 条件边连接 + MemorySaver 编译
│   ├── state.py            # WealthState 定义
│   ├── nodes.py            # 8 个节点实现
│   └── conditional_edges.py# after_evaluate / after_strategy 条件路由
├── service/
│   ├── math_tools.py       # 4 大计算工具（凯利/网格/回撤/复利）
│   ├── tools.py            # 工具注册 + Wealth 专属工具封装
│   ├── cache.py            # MemoryCache（TTL + asyncio.Lock）
│   ├── quota.py            # QuotaManager（租户/用户/数据源三级配额）
│   ├── datasources/        # 外部数据源实现
│   │   ├── yahoo_finance.py
│   │   ├── fred_api.py
│   │   └── tavily_search.py
│   └── retrieval/
│       └── three_track_retriever.py  # 三轨融合检索器
├── static/
│   └── index.html          # Web 聊天界面（预留）
└── tests/                  # 分阶段节点测试 + 商业组件测试
    ├── test_phase2_nodes.py
    ├── test_phase3_nodes.py
    ├── test_phase4_nodes.py
    └── test_commercial_components.py
```

---

## 3. 核心模块设计

### 3.1 LangGraph 状态图 (graph/)

#### 3.1.1 状态定义 (state.py)

```python
class WealthState(TypedDict):
    # 用户输入
    user_message: str                    # 本轮用户输入
    user_id: str                         # MemorySaver thread_id

    # 查询分析（analyze_and_route 节点写入）
    time_sensitive: bool                 # 是否含时效性关键词
    entities: list[dict]                 # NER 实体（ticker/macro_indicator/date）

    # 子查询与改写
    sub_queries: list[dict]              # 拆解后的子查询（含 domain/weight）
    rewritten_queries: list[str]         # 改写后的查询文本

    # 检索（parallel_retrieval 节点写入）
    retrieved_context: str | None        # 合并后的检索上下文
    kg_context: str | None               # 知识图谱补充（预留）
    retrieval_iterations: int            # 已执行检索-评估轮数

    # 评估（evaluate_and_rerank 节点写入）
    confidence: float                    # 综合检索置信度
    top_ce: float                        # CrossEncoder 最高分（核心阈值依据）
    evaluation_result: dict | None       # 评估详细结果

    # 策略推演与工具
    needs_tool: bool                     # 是否需要数学计算工具
    tool_calls: list[dict]               # 待执行工具调用描述
    tool_results: list[Any]              # 工具执行原始结果
    math_result: dict | None             # 结构化数学结果

    # 生成
    reply: str                           # 最终回复文本

    # 会话历史
    history: list[dict]                  # 对话历史
    summary: str                         # 历史摘要
    token_budget: int                    # 剩余 token 预算
    messages: list                       # LangChain/OpenAI 格式消息
    trimmed: list                        # 被裁剪的旧消息

    # 可观测性
    trace: list[dict]                    # 全链路执行轨迹
```

与 ai_app3 的 `RagState` 对比：
- 新增 `time_sensitive` / `entities`：查询分析产物，用于驱动三轨检索策略
- 新增 `top_ce`：CrossEncoder 精排最高分，作为 Self-RAG 反思的核心阈值信号
- 新增 `needs_tool` / `tool_calls` / `math_result`：策略计算工具调用链路
- 新增 `sub_queries` / `rewritten_queries`：本地 Qwen 拆解与改写产物
- 保留 `kg_context`：知识图谱补充上下文（当前预留）

#### 3.1.2 节点设计 (nodes.py)

| 节点 | 类型 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| `analyze_and_route_node` | async | `user_message` | `time_sensitive`, `entities`, `sub_queries`, `rewritten_queries` | 时效性检测 + NER + 领域分类 + 中英文翻译 |
| `parallel_retrieval_node` | async | `sub_queries`, `entities` | `retrieved_context`, `retrieval_iterations`, `top_ce` | 三轨融合检索或 HybridRetriever 本地检索 |
| `evaluate_and_rerank_node` | async | `retrieved_context`, `top_ce` | `confidence`, `evaluation_result` | 真实 top_ce 与启发式 confidence 加权融合 |
| `query_reflection_node` | async | `user_message` | `user_message`(改写后), `sub_queries` | LLM 驱动金融术语化改写，失败降级规则改写 |
| `strategy_reasoning_node` | async | `retrieved_context`, `history` | `reply`, `needs_tool`, `tool_calls` | 主模型推演，解析 TOOL_CALL JSON 指令 |
| `execute_math_tool_node` | async | `tool_calls` | `tool_results`, `math_result` | 遍历 tool_calls 调用 tool_registry.execute_tool |
| `merge_and_generate_node` | async | `reply`, `math_result` | `reply`(最终) | 合并计算结果与 LLM 话术，追加来源标注与免责声明 |
| `generate_final_node` | async | `reply` | `reply`(最终) | 纯文本回复，追加来源标注与免责声明 |

**LLM 工具调用约定**：`strategy_reasoning_node` 在 system prompt 中注入可用工具描述，要求 LLM 在分析末尾输出严格 JSON 格式的 `TOOL_CALL` 指令。`_parse_tool_calls()` 通过花括号深度计数解析嵌套 JSON，提取工具名与参数。

#### 3.1.3 条件边 (conditional_edges.py)

```python
# after_evaluate: 检索评估后的路由
top_ce = state.get("top_ce", 0.0)
iterations = state.get("retrieval_iterations", 0)
threshold = eval_result.get("reflection_threshold", 0.35)
max_loop = eval_result.get("max_loop_count", 2)

if (top_ce if top_ce > 0 else confidence) < threshold and iterations < max_loop:
    return "reflection"   # 进入自旋锁反思
return "strategy"         # 进入策略推演

# after_strategy: 策略推演后的路由
if state.get("needs_tool", False):
    return "tool"   # 执行数学计算工具
return "final"      # 直接生成纯文本回复
```

**反思循环限制**：最多 `max_loop_count=2` 次检索-反思迭代。超过后强制进入策略推演，避免无限消耗资源。

#### 3.1.4 图构建 (builder.py)

```python
builder = StateGraph(WealthState)

builder.add_node("analyze_and_route", analyze_and_route_node)
builder.add_node("parallel_retrieval", parallel_retrieval_node)
builder.add_node("evaluate_and_rerank", evaluate_and_rerank_node)
builder.add_node("query_reflection", query_reflection_node)
builder.add_node("strategy_reasoning", strategy_reasoning_node)
builder.add_node("execute_math_tool", execute_math_tool_node)
builder.add_node("merge_and_generate", merge_and_generate_node)
builder.add_node("generate_final", generate_final_node)

builder.set_entry_point("analyze_and_route")
builder.add_edge("analyze_and_route", "parallel_retrieval")
builder.add_edge("parallel_retrieval", "evaluate_and_rerank")

builder.add_conditional_edges(
    "evaluate_and_rerank", after_evaluate,
    {"reflection": "query_reflection", "strategy": "strategy_reasoning"}
)
builder.add_edge("query_reflection", "parallel_retrieval")

builder.add_conditional_edges(
    "strategy_reasoning", after_strategy,
    {"tool": "execute_math_tool", "final": "generate_final"}
)
builder.add_edge("execute_math_tool", "merge_and_generate")

builder.add_edge("merge_and_generate", END)
builder.add_edge("generate_final", END)

graph = builder.compile(checkpointer=MemorySaver())
```

---

### 3.2 三轨融合检索 (service/retrieval/three_track_retriever.py)

`ThreeTrackRetriever` 实现 `rag_framework.retrieval.base.Retriever` 接口，将三类数据源统一编排为单一路由器：

| Track | 数据源 | 触发条件 | 权重 | 说明 |
|-------|--------|---------|------|------|
| Track A | 本地 HybridRetriever | 始终启用 | 1.0 | 历史财报、宏观报告、结构化分析 |
| Track B | Yahoo Finance / FRED API | 含 ticker 或宏观指标 | 0.95 | 实时股价、财报数据、宏观指标 |
| Track C | Tavily Search | `time_sensitive=True` | 0.85 | 最新新闻、市场动态、政策解读 |

**核心设计**：
1. **分级触发**：`DataSource.should_fetch()` 根据 `QueryRoute` 和 `FetchContext` 决定是否启用
2. **并发执行**：`asyncio.gather` 并行调用所有启用的 track，`return_exceptions=True` 保证单 track 失败不影响整体
3. **权重融合**：每个 track 的结果按配置权重加权，降序排列后截取 `top_k`
4. **来源标注**：每个 `RetrievedDoc` 携带 `source_name`，供生成阶段溯源
5. **成本控制**：`QuotaManager` 配额检查 + `asyncio.wait_for` 超时降级

**降级策略**：三轨检索器初始化失败或运行时异常时，`parallel_retrieval_node` 自动回退到原生 `HybridRetriever`（Track A 本地检索）。

---

### 3.3 数学计算工具箱 (service/math_tools.py)

纯 Python 实现，无外部依赖，通过 `rag_framework.llm.tool_registry` 注册，可被 LLM 在 tool calling 循环中直接调用：

| 工具函数 | 用途 | 关键参数 |
|----------|------|----------|
| `kelly_criterion_calc` | 凯利公式最优仓位 | `win_rate`, `avg_gain_pct`, `avg_loss_pct`, `current_capital` |
| `grid_trading_cost_estimator` | 网格交易成本估算 | `lower_bound`, `upper_bound`, `num_grids`, `total_capital`, `fee_rate_pct` |
| `portfolio_drawdown_estimator` | 组合最大回撤估算 | `allocations`, `drawdown_scenarios`, `total_capital` |
| `compound_growth_calculator` | 复利增长计算 | `principal`, `annual_return_pct`, `years`, `monthly_contribution` |

所有函数返回结构化 `dict`，包含计算结果和 `warning` 字段（参数不合理时给出警告）。`merge_and_generate_node` 将 `math_result` 格式化为可读文本摘要，并尝试让 LLM 基于计算结果重新润色生成最终回复。

---

### 3.4 缓存与配额管理

#### MemoryCache (service/cache.py)
- 纯内存 dict + `asyncio.Lock` 线程安全
- TTL 自动过期，惰性清理
- 适用场景：股价数据（5 分钟）、宏观指标（1 小时）、搜索结果（短 TTL 或不缓存）

#### QuotaManager (service/quota.py)
- 按 `tenant_id + user_id + source_name` 三级维度追踪
- **硬配额**：超过后拒绝调用
- **软配额**：超过后记录警告但仍允许
- **小时配额**：防止短时 burst
- 默认配额配置：
  - Tavily Search: 日硬 50 / 日软 40 / 小时 10
  - Yahoo Finance: 日硬 500 / 日软 400 / 小时 100
  - FRED API: 日硬 200 / 日软 150 / 小时 50

---

### 3.5 配置系统 (core/config.py)

`WealthSettings` 继承 `RAGSettings`，环境变量前缀 `WEALTH_`（基类字段仍通过 `RAG_*` 覆盖）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `active_domain` | `"wealth"` | 覆盖基类默认 `"android"` |
| `math_tool_enabled` | `True` | 是否启用数学计算工具箱 |
| `reflection_threshold` | `0.35` | 触发 query_reflection 的 top_ce 阈值 |
| `max_loop_count` | `2` | 检索-反思最大循环次数 |
| `enable_query_decomposition` | `True` | 是否启用子查询拆解 |
| `enable_query_translation` | `True` | 是否启用中英文 Query 翻译 |
| `max_sub_queries` | `4` | 单轮最大子查询数 |
| `retriever_top_k` | `10` | 单路检索返回文档数 |
| `cross_encoder_top_k` | `5` | CrossEncoder 精排后截取数 |
| `wealth_system_prompt` | 见代码 | 全球资产配置分析师人设 |
| `yahoo_finance_enabled` | `True` | 是否启用 Yahoo Finance |
| `fred_api_enabled` | `False` | 是否启用 FRED API（需 Key） |
| `tavily_search_enabled` | `False` | 是否启用 Tavily 搜索 |
| `three_track_enabled` | `True` | 是否启用三轨融合检索 |
| `track_a_weight` | `1.0` | 本地 RAG 权重 |
| `track_b_weight` | `0.95` | 金融 API 权重 |
| `track_c_weight` | `0.85` | 网络搜索权重 |
| `enable_source_attribution` | `True` | 是否标注数据来源 |
| `enable_compliance_disclaimer` | `True` | 是否追加投资免责声明 |
| `enable_trace` | `True` | 是否输出全链路 trace |

---

### 3.6 扩展容器 (core/container.py)

`WealthContainer` 继承 `RAGContainer`，frozen dataclass，**无新增字段，仅复用父类构建逻辑**：

```python
@dataclass(frozen=True, slots=True)
class WealthContainer(RAGContainer):
    @classmethod
    def from_settings(cls, settings: WealthSettings | None = None) -> "WealthContainer":
        settings = settings or WealthSettings()
        return super(WealthContainer, cls).from_settings(settings)
```

通过显式传递 `cls`，确保 `RAGContainer.from_settings()` 内部 `return cls(...)` 返回 `WealthContainer` 实例而非 `RAGContainer`。

---

### 3.7 API 接口 (api/chat.py)

| 端点 | 方法 | 响应格式 | 说明 |
|------|------|----------|------|
| `POST /api/chat` | SSE | `text/event-stream` | 流式响应：trace → content(逐字) → done |
| `POST /api/chat/json` | JSON | `application/json` | 非流式，返回完整结果 + 元数据 |

SSE 事件类型：
- `{"type": "trace", "data": [...], "latency_ms": 1234.5}` — 全链路执行轨迹
- `{"type": "content", "data": "A"}` — 逐字推送（模拟打字机效果，间隔 5ms）
- `{"type": "done"}` — 流结束标记

---

## 4. 与其他 app 的对比

| 维度 | ai_app1 | ai_app2 | ai_app3 | ai_app4 |
|------|---------|---------|---------|---------|
| 架构模式 | 线性流水线 | LangGraph 状态图 | Agentic RAG | **商业级 Agentic RAG + 三轨检索** |
| 目标领域 | Android 开发 | Android 开发 | Android 开发 | **全球资产/宏观经济/量化策略** |
| 领域路由 | 规则（中文→android） | 规则 | 意图分类 | **NER + 时效性检测 + 领域分类** |
| 检索引擎 | HybridRetriever | HybridRetriever | HybridRetriever | **HybridRetriever + 三轨融合** |
| 实时数据 | 无 | 无 | 无 | **Yahoo Finance / FRED / Tavily** |
| 数学工具 | 无 | 无 | 基础乘除 | **凯利公式/网格交易/回撤/复利** |
| Self-RAG | 无 | 无 | 有（evaluate→rewrite） | **有（top_ce 驱动 + 金融术语化改写）** |
| 情绪感知 | 无 | 无 | 无 | 无（投资场景暂不需要） |
| 转人工 | 无 | 无 | 无 | 无 |
| 会话持久化 | 内存字典 | MemorySaver | MemorySaver | **MemorySaver** |
| 多租户 | 无 | 无 | 无 | 无（预留 tenant_id） |
| 来源标注 | 无 | 无 | 无 | **有（本地/API/搜索）** |
| 合规声明 | 无 | 无 | 无 | **有（投资免责声明）** |
| API 配额 | 无 | 无 | 无 | **有（三级配额管理）** |
| 缓存层 | 无 | 无 | 无 | **MemoryCache（TTL）** |
| 可观测性 | 日志 + Trace | 日志 | Trace 字段 | **Trace 字段 + 延迟拆解** |

---

## 5. 文件职责一览

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，端口 8004，挂载路由和静态文件 |
| `lifespan.py` | 生命周期：加载配置 → 构建容器 → 初始化三轨检索器 → 注册缓存/配额 → 注册数学工具 → 预热 |
| `core/config.py` | `WealthSettings` — 投资分析专属配置（数据源开关、权重、阈值、系统 prompt） |
| `core/container.py` | `WealthContainer` — 继承 RAGContainer，复用父类构建逻辑 |
| `core/context.py` | 全局上下文单例 — 供 LangGraph 节点获取容器/配置/缓存/配额/三轨检索器，避免循环导入 |
| `graph/state.py` | `WealthState` — 全局状态定义，所有节点共享 |
| `graph/builder.py` | StateGraph 构建：节点注册、边连接、条件边、MemorySaver 编译 |
| `graph/nodes.py` | 8 个节点实现：分析/检索/评估/反思/推演/工具/合并/生成 |
| `graph/conditional_edges.py` | `after_evaluate` / `after_strategy` 条件路由逻辑 |
| `api/chat.py` | `/chat` SSE 流式 + `/chat/json` 非流式接口 |
| `service/math_tools.py` | 4 大数学计算工具纯 Python 实现 |
| `service/tools.py` | 工具注册 — 将 math_tools 注册到 `rag_framework.llm.tool_registry` |
| `service/cache.py` | `MemoryCache` — 异步安全内存缓存，TTL 自动过期 |
| `service/quota.py` | `QuotaManager` — 按租户/用户/数据源三级 API 配额管理 |
| `service/datasources/*.py` | 外部数据源实现：YahooFinanceSource / FredAPISource / TavilySearchSource |
| `service/retrieval/three_track_retriever.py` | `ThreeTrackRetriever` — 三轨融合检索器，统一编排本地+API+搜索 |

---

## 6. 待实现的路线图

按优先级排序：

1. **LLM-based 检索评估**：当前 `evaluate_and_rerank_node` 使用启发式 confidence，后续可接入小 LLM 判断检索充分性
2. **知识图谱增强**：`kg_context` 字段已预留，可接入 `ai_app3` 的 `knowledge_graph.py` 补充实体关系
3. **Redis 持久化**：`MemoryCache` 和 `QuotaManager` 当前为进程内内存，多实例部署时需替换为 Redis
4. **真实股价数据对接**：`YahooFinanceSource` 当前为骨架实现，需接入真实 Yahoo Finance API
5. **Tavily 搜索深度优化**：当前 `search_depth="basic"`，后续支持 `"advanced"` 模式获取更深度分析
6. **多租户隔离**：`tenant_id` 已贯穿 FetchContext，需完善数据层面的租户隔离
7. **流式 LLM 生成**：当前 SSE 为 graph 执行完成后逐字模拟，后续可接入真实 LLM 流式输出
