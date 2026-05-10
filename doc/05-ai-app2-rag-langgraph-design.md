# ai_app2 - Android RAG 问答系统 (LangGraph 重构版)

> 版本: 1.0 | 最后更新: 2026-05-10
> 基于 ai_app1 v2.2 的 LangGraph 重构，复用全部检索管道逻辑，核心流程由手写循环迁移至状态图编排

---

## 1. 系统定位

ai_app2 是 ai_app1 的 **LangGraph 架构升级版本**，面向 Android 开发者的智能问答助手。系统在保留 ai_app1 全部 RAG 检索能力（多路混合检索、RRF 融合、Rerank 精排、Lost-in-Middle 重排）的基础上，将对话流程由手写的顺序函数调用重构为 **LangGraph `StateGraph` 状态图编排**，实现：

- **流程可视化**：每个处理步骤成为图中显式节点，数据流清晰可追溯
- **状态持久化**：通过 `checkpointer` 自动管理会话生命周期，无需手写内存字典
- **工具调用自动化**：LLM 的 tool calling 多轮循环由 LangChain `bind_tools` + Graph 节点自动处理
- **可扩展性**：后续增加意图识别、查询改写、多轮澄清等能力时，只需增删节点和边

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户请求层                                   │
│                    POST /chat  { "message": "..." }                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI 应用层 (main.py)                          │
│  - 挂载 chat_router (端口 8001，与 ai_app1 的 8000 共存)                  │
│  - 静态文件服务 /ui                                                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Chat API 路由层 (chat.py)                         │
│  1. 接收请求 → 从 checkpointer 加载线程状态                               │
│  2. 用户消息入 history → 组装 input_state                                 │
│  3. 调用 graph.ainvoke(input_state) → 触发完整图执行                       │
│  4. 获取 reply → 流式逐字 yield 给客户端                                  │
│  5. Graph 内部自动完成：retrieve → build_messages → llm → save_reply       │
│     → should_summarize? → summarize/trim → END                           │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LangGraph 状态图 (graph/builder.py)                   │
│                                                                          │
│   START ──→ retrieve ──→ build_messages ──→ llm ──→ save_reply          │
│                                              │                           │
│                                              ▼                           │
│                                    should_summarize?                     │
│                                         │                                │
│                    ┌────────────────────┴────────────────────┐           │
│                    ▼                                          ▼          │
│              summarize ──→ trim ──→ END                trim ──→ END     │
│                                                                          │
│  节点说明：                                                               │
│  - retrieve    : 调用 ai_app1.query_db() 执行混合检索（复用）              │
│  - build_messages: SystemMessage + 摘要 + history + 参考资料              │
│  - llm         : ChatOpenAI.bind_tools() + 手工 tool calling 循环         │
│  - save_reply  : 将 assistant 回复追加到 history                          │
│  - summarize   : token 超预算时压缩历史为摘要                             │
│  - trim        : 裁剪 history 到 MAX_HISTORY 条                           │
│                                                                          │
│  状态管理：MemorySaver (内存 checkpointer，可替换为 Redis/Postgres)        │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         混合检索管道 (复用 ai_app1)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                                │
│  │ 路A Dense │  │ 路B HyDE │  │ 路C BM25 │   ← 三路召回                  │
│  │ 向量检索  │  │ 假设问题  │  │ 稀疏全文 │                                │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                                │
│       └─────────────┴─────────────┘                                      │
│                     │                                                    │
│                     ▼                                                    │
│              ┌──────────────┐                                            │
│              │ RRF 融合排名 │   ← Reciprocal Rank Fusion                │
│              └──────┬───────┘                                            │
│                     │                                                    │
│                     ▼                                                    │
│              ┌──────────────┐                                            │
│              │ Rerank 精排  │   ← 多维度线性评分                         │
│              └──────┬───────┘                                            │
│                     │                                                    │
│                     ▼                                                    │
│              ┌──────────────┐                                            │
│              │ Lost-in-Middle│  ← 上下文重排 (最相关→首位/次相关→末位)    │
│              └──────────────┘                                            │
│                                                                          │
│  接口：ai_app2/service/retriever.py 直接 import ai_app1.service.vector_store │
│  的 query_db()，检索逻辑零改动。                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 LangGraph 状态图 (graph/)

#### 3.1.1 状态定义 (state.py)

```python
class RagState(TypedDict):
    user_message: str           # 本轮用户输入
    history: list               # 原始对话历史（user/assistant 的 dict 列表）
    summary: str                # 历史摘要
    token_budget: int           # 剩余 token 预算（阈值，非递减计数器）
    retrieved_context: str | None  # 混合检索结果
    messages: list              # LangChain Message 对象列表（每轮重建）
    reply: str                  # AI 本轮最终回复文本
    trimmed: list               # 被裁剪的旧消息
```

与 ai_app1 的 `SessionData` 对比：
- 增加 `messages` 字段：存储 LangChain `SystemMessage / HumanMessage / AIMessage / ToolMessage`，供 LLM 节点直接使用
- `history` 保持为 dict 列表：用于 summarize 节点生成人类可读的历史文本
- 移除 `AiClient` 单例：LLM 实例在 `builder.py` 模块级创建，通过闭包注入节点

#### 3.1.2 节点设计 (nodes.py)

| 节点 | 类型 | 输入 | 输出 | 替代了 ai_app1 的 |
|------|------|------|------|------------------|
| `retrieve_node` | sync | `user_message` | `retrieved_context` | `session.build_messages` 中的 `query_db(req_msg)` |
| `build_messages_node` | sync | `history`, `summary`, `retrieved_context`, `user_message` | `messages` | `session.build_messages_raw` + `build_messages` |
| `llm_node` | async | `messages` | `reply` | `AiClient.run_agent()` 手写循环 |
| `save_reply_node` | sync | `reply`, `history` | `history` | `chat.py` 中的 `add_assistant_message` |
| `summarize_node` | async | `history` | `summary` | `AiClient.summarize()` |
| `trim_node` | sync | `history` | `history`, `trimmed` | `session.trim_history()` |

**条件边**：`save_reply` → `should_summarize` → `"summarize" | "trim"`

- token 预算检测通过 `_estimate_tokens()` 估算（复用 ai_app1 加权字符数策略）
- 检测发生在 AI 回复入栈之后，确保 AI 已看过本轮消息再决定是否压缩

#### 3.1.3 LLM 节点设计 (llm_node)

ai_app2 使用 `langchain_openai.ChatOpenAI` 替代原生的 `openai.AsyncOpenAI`：

```python
_llm = ChatOpenAI(
    model="MiniMax-M2.7",
    base_url="https://api.minimaxi.com/v1",
    api_key=OPENAI_API_KEY,
    temperature=0.3,
).bind_tools(TOOLS)  # ← 工具绑定
```

`llm_node` 内部保留手工 tool calling 循环（因 MiniMax API 的流式与工具调用格式兼容性问题，暂不使用 `create_react_agent`）：

```
for step in range(MAX_STEPS):
    response = await llm.ainvoke(messages)
    if response.tool_calls:
        执行工具 → 追加 ToolMessage → 继续循环
    else:
        返回 content → 结束
```

与 ai_app1 的 `AiClient.run_agent()` 等价，但工具通过 `bind_tools()` 自动注入，无需手动拼接 `tools` 参数。

#### 3.1.4 图构建 (builder.py)

```python
builder = StateGraph(RagState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("build_messages", build_messages_node)
builder.add_node("llm", _llm_node_wrapper)
builder.add_node("save_reply", save_reply_node)
builder.add_node("summarize", _summarize_node_wrapper)
builder.add_node("trim", trim_node)

builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "build_messages")
builder.add_edge("build_messages", "llm")
builder.add_edge("llm", "save_reply")
builder.add_conditional_edges(
    "save_reply", should_summarize,
    {"summarize": "summarize", "trim": "trim"}
)
builder.add_edge("summarize", "trim")
builder.add_edge("trim", END)

graph = builder.compile(checkpointer=MemorySaver())
```

**checkpointer**：`MemorySaver` 为内存级状态持久化，每个 `thread_id` 对应一个会话。后续可无缝替换为：
- `langgraph.checkpoint.postgres.PostgresSaver`
- `langgraph.checkpoint.redis.RedisSaver`

---

### 3.2 混合检索管道 (复用 ai_app1)

ai_app2 的检索能力完全复用 ai_app1，不做任何修改。

#### 复用方式

```python
# ai_app2/service/retriever.py
from ai_app1.service.vector_store import query_db
__all__ = ["query_db"]
```

检索管道包含的四级架构（Dense / HyDE / BM25 → RRF → Rerank → Lost-in-Middle）以及父子回溯机制、降级策略，均与 ai_app1 文档第 3.1 节完全一致，此处不再重复。

---

### 3.3 会话管理 (graph + checkpointer)

#### 3.3.1 与 ai_app1 的对比

| 维度 | ai_app1 (session.py) | ai_app2 (LangGraph) |
|------|---------------------|---------------------|
| 存储介质 | 手写 `user_sessions: dict[str, SessionData]` | `MemorySaver` checkpointer |
| 生命周期 | 进程级内存，重启丢失 | 内存级，可替换为 Redis/Postgres |
| 状态访问 | 函数调用 `get_session(user_id)` | `graph.get_state(config)` |
| 状态更新 | 手动 `update_summary`, `trim_history` | Graph 节点自动返回更新字典 |
| 并发安全 | 依赖 GIL + dict 原子性 | checkpointer 内部加锁 |
| 可观测性 | 靠 logger 打印 | LangSmith 自动追踪每个节点的输入输出 |

#### 3.3.2 消息生命周期

```
用户请求
    │
    ▼
checkpointer 加载线程 state (若存在)
    │
    ▼
用户消息追加到 history ──────────────────────────────────┐
    │                                                     │
    ▼                                                     │
组装 input_state → graph.ainvoke()                        │
    │                                                     │
    ├── retrieve_node ──→ query_db(req_msg)  ←────────────┘
    │
    ├── build_messages_node
    │       ├── SystemMessage(system prompt)
    │       ├── HumanMessage(历史摘要) (若存在)
    │       ├── HumanMessage/ AIMessage(history)
    │       └── HumanMessage(参考资料)
    │
    ├── llm_node ──→ AI 回复 (含 tool calling 循环)
    │
    ├── save_reply_node ──→ assistant 消息入 history
    │
    ├── should_summarize? (条件边)
    │       ├── 是 → summarize_node → 更新 summary
    │       └── 否 → 跳过
    │
    └── trim_node ──→ history 保留最近 MAX_HISTORY=4

checkpointer 自动保存最终 state
流式返回 reply 给客户端
```

**关键设计原则不变**：summarize 和 trim 仍发生在 AI 回复入栈之后。

---

### 3.4 Token 估算

复用 ai_app1 的加权字符数策略：

```python
cn_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
other_chars = len(text) - cn_chars
tokens = int(cn_chars * 1.5 + other_chars * 0.5)
```

估算由 `build_messages_node` 构建的 `messages` 列表总 token 数，当 `total_tokens >= token_budget` 时触发 summarize。

---

### 3.5 工具定义

ai_app2 使用 LangChain 的 `@tool` 装饰器替代 ai_app1 的手写 JSON Schema：

```python
from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """计算两个数字的乘积"""
    return a * b

TOOLS = [multiply]
```

Schema 自动生成，LLM 通过 `ChatOpenAI.bind_tools(TOOLS)` 获取工具定义。

---

## 4. 数据流

```mermaid
flowchart TD
    A[用户提问] --> B[/chat API]
    B --> C[checkpointer 加载线程 state]
    C --> D[用户消息入 history]
    D --> E[组装 input_state]
    E --> F[graph.ainvoke]
    F --> G{LangGraph StateGraph}

    G --> G1[retrieve_node]
    G1 --> G1a[query_db 混合检索]
    G1a --> G2[build_messages_node]

    G2 --> G2a[SystemMessage]
    G2 --> G2b[摘要/HumanMessage]
    G2 --> G2c[history → HumanMessage/AIMessage]
    G2 --> G2d[参考资料/HumanMessage]
    G2a & G2b & G2c & G2d --> G3[llm_node]

    G3 --> G3a[ChatOpenAI.ainvoke]
    G3a --> G3b{tool_calls?}
    G3b -->|是| G3c[执行工具 → ToolMessage]
    G3c --> G3a
    G3b -->|否| G4[save_reply_node]

    G4 --> G5{should_summarize?}
    G5 -->|是| G6[summarize_node]
    G6 --> G7[trim_node]
    G5 -->|否| G7

    G7 --> G8[checkpointer 保存 state]
    G8 --> H[流式返回 reply]
```

---

## 5. 关键配置

| 配置项 | 文件 | 默认值 | 说明 |
|--------|------|--------|------|
| OPENAI_API_KEY | `.env` (复用 ai_app1/.env) | — | MiniMax API Key |
| CHROMA_DB_PATH | `core/config.py` | 动态计算路径 | 指向 ai_app1/pre/chroma_db |
| MAX_HISTORY | `core/config.py` | 4 | history 保留条数 |
| DEFAULT_TOKEN_BUDGET | `core/config.py` | 4096 | 会话 token 上限 |
| MAX_STEPS | `core/config.py` | 10 | Agent tool calling 最大步数 |
| RRF_K / DENSE_TOP_K / ... | `ai_app1/service/vector_store.py` | 同 ai_app1 | 检索超参数（复用） |

---

## 6. 运行流程

### 6.1 首次部署

```bash
# 1. 安装依赖（ai_app2 与 ai_app1 共用 pyproject.toml）
uv sync

# 2. 复用 ai_app1 的环境变量与索引
#    ai_app2/core/config.py 会自动回退到 ai_app1/.env

# 3. 确保 ai_app1 的索引已构建
uv run python -m ai_app1.pre.verify_phase2

# 4. 启动 ai_app2（端口 8001，与 ai_app1:8000 共存）
uv run python -m uvicorn ai_app2.main:app --host 0.0.0.0 --port 8001
```

### 6.2 API 调用

```bash
# ai_app2 接口与 ai_app1 完全一致
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Android 中 NullPointerException 如何解决？"}'
```

### 6.3 Web UI

访问 `http://localhost:8001/ui`，内置的 `index.html` 与 ai_app1 前端一致（默认 API 地址改为 8001）。

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 编排框架 | LangGraph `StateGraph` | 替代手写的顺序函数调用，流程可视化、节点可独立测试、状态自动持久化 |
| LLM 客户端 | `langchain_openai.ChatOpenAI` | 复用 MiniMax 兼容端点，`bind_tools()` 自动注入工具定义，无需手写 JSON Schema |
| 检索管道 | 完全复用 ai_app1 | 多路混合检索 + Rerank + Lost-in-Middle 是 ai_app1 的核心竞争力，LangGraph 没有现成替代方案，包装为 Node 即可 |
| 状态持久化 | `MemorySaver` | 零配置启动，后续可无缝替换为 Redis/Postgres 实现分布式会话 |
| 会话存储键 | `thread_id` | LangGraph checkpointer 的标准标识符，天然支持多用户隔离 |
| 流式响应 | Graph 完成后逐字 yield | MiniMax API 的流式与工具调用格式兼容性限制，暂采用非流式调用 + 客户端模拟流式 |
| Token 估算 | 复用 ai_app1 加权字符数 | 针对中英文混合场景优化，无需 tiktoken |
| summarize 时机 | AI 回复后 | 与 ai_app1 保持一致，避免 AI 还没看消息就被压缩 |

---

## 8. 已知风险与缓解措施

### 8.1 检索冗余与上下文污染（A）

**状态**：复用 ai_app1，已缓解 ✅
- `query_db()` 的 `seen_ids` 去重 + `rerank_chunks()` 后运行时断言已保留

### 8.2 Rerank 线性评分量级不统一（B）

**状态**：复用 ai_app1，已缓解 ✅
- `normalized_rrf = rrf_score / max_rrf` 归一化逻辑已保留

### 8.3 Token 估算的中英文混合陷阱（C）

**状态**：复用 ai_app1，已缓解 ✅
- `_estimate_tokens()` 在 `nodes.py` 中复用相同算法

### 8.4 Summarize 的上下文断裂（D）

**状态**：复用 ai_app1，风险不变
- 保留最近 `MAX_HISTORY=4` 条原始消息，超出部分才 summarize
- **未来优化**：在 summarize_node 中使用结构化摘要（保留代码片段、行号）

### 8.5 Async 同步客户端阻塞（E）

**状态**：已消除 ✅
- ai_app2 使用 `ChatOpenAI.ainvoke()`，内部自动使用 `AsyncOpenAI`，无需手动管理客户端

### 8.6 模块级冗余实例化（F）

**状态**：已消除 ✅
- ai_app2 的 LLM 实例在 `builder.py` 模块级创建，但被 `graph` 复用，不存在 ai_app1 中 `main.py` 和 `chat.py` 各创建一个的问题

### 8.7 Summarize 输入格式（G）

**状态**：已缓解 ✅
- `summarize_node` 使用 `\n.join(f"{role}: {content}" for m in history)` 生成人类可读文本

### 8.8 路径硬编码（H）

**状态**：已消除 ✅
- `CHROMA_DB_PATH` 使用 `os.path.join(os.path.dirname(__file__), ...)` 动态计算，指向 ai_app1/pre/chroma_db

### 8.9 LangGraph 版本兼容性（新增）

**风险**：LangGraph 处于快速迭代期，`astream_events` API 和 `checkpointer` 接口可能在后续版本中发生 breaking change。

**缓解**：
- 当前锁定 `langgraph>=0.4.0`，`langchain-openai>=0.3.0`
- 核心检索逻辑不依赖 LangGraph，即使升级失败也只需调整 `builder.py` 和 `nodes.py`
- 避免使用实验性功能（如 `create_react_agent` 的流式输出），使用稳定的 `ainvoke()` + 手工 tool loop

### 8.10 流式响应延迟（新增）

**风险**：ai_app2 当前采用 `graph.ainvoke()` 非流式调用，获取完整 reply 后再逐字 yield，首字时间（TTFT）等于完整推理时间，用户体验不如 ai_app1 的 `_stream_response` 原生流式。

**缓解**：
- 未来 MiniMax API 流式与工具调用兼容性改善后，可迁移至 `graph.astream_events()` 实现真正的 token 级流式
- 当前在客户端通过减小 chunk_size (2 字符) + 短延迟 (5ms) 模拟流式效果

---

## 9. ai_app1 vs ai_app2 对比总结

| 维度 | ai_app1 | ai_app2 (LangGraph) |
|------|---------|---------------------|
| **代码量** | ~650 行核心逻辑 | ~350 行核心逻辑（LangGraph 接管了循环/状态管理） |
| **会话管理** | 手写内存字典 | `MemorySaver` checkpointer |
| **工具调用** | 手写 `for step in range(10)` 循环 | `bind_tools()` + 节点内循环，Schema 自动生成 |
| **流程编排** | 顺序函数调用（chat.py 中硬编码） | `StateGraph` 显式节点和边，可可视化 |
| **状态可观测** | logger 打印 | LangSmith 自动追踪每个节点输入输出 |
| **扩展性** | 增加新步骤需修改 chat.py 多处 | 增加节点 → 改图即可，不影响其他节点 |
| **流式输出** | 原生 token 级流式（TTFT 快） | 完整回复后模拟流式（TTFT 等于总耗时） |
| **检索能力** | Dense + HyDE + BM25 + RRF + Rerank + L-i-M | **完全复用** |
| **并发性能** | AsyncOpenAI + FastAPI | ChatOpenAI.ainvoke() + FastAPI，等价 |

---

## 10. 后续优化方向

1. **真流式输出**：MiniMax API 支持流式 tool calling 后，迁移至 `graph.astream_events()`
2. **并行召回**：将 Dense / HyDE / BM25 三个检索节点改为并行分支，降低检索延迟
3. **意图识别节点**：在 `retrieve` 前增加 `intent_node`，区分"闲聊"与"技术问答"，闲聊场景跳过检索
4. **查询改写节点**：增加 `rewrite_node`，将用户口语化查询改写为标准技术术语，提升检索精度
5. **持久化升级**：将 `MemorySaver` 替换为 `PostgresSaver`，支持服务重启后恢复会话
6. **结构化摘要**：`summarize_node` 使用 JSON Schema 输出，保留关键代码片段和行号
