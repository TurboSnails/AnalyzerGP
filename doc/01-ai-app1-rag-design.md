# ai_app1 - Android RAG 问答系统设计文档

> 版本: 2.0 | 最后更新: 2026-05-08

---

## 1. 系统定位

ai_app1 是一个面向 Android 开发者的 **智能问答助手**，基于 RAG (Retrieval-Augmented Generation) 架构构建。系统将《Android 开发核心注意事项与避坑指南》作为知识源，通过多路混合检索为 MiniMax-M2.7 大模型提供精准上下文，从而回答 Android 开发中的各类技术问题。

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
│  - 挂载 chat_router                                                    │
│  - 进程级 AiClient 单例                                                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Chat API 路由层 (chat.py)                         │
│  1. 接收请求 → 获取/创建 Session                                        │
│  2. 用户消息入栈 → build_messages                                       │
│  3. 调用 ai_client.run_agent → 获取回复                                │
│  4. AI 回复入栈 → 判断是否 summarize                                    │
│  5. trim_history → 返回 reply                                           │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         会话管理层 (session.py)                            │
│  - SessionData: history / summary / trimmed / token_budget              │
│  - build_messages: system prompt + summary + history + 检索上下文       │
│  - should_summarize: token 预算检测 → 触发压缩                          │
│  - trim_history: 保留最近 MAX_HISTORY=4 条                             │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         混合检索管道 (vector_store.py)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                               │
│  │ 路A Dense │  │ 路B HyDE │  │ 路C BM25 │   ← 三路召回                 │
│  │ 向量检索  │  │ 假设问题  │  │ 稀疏全文 │                               │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                               │
│       └─────────────┴─────────────┘                                     │
│                     │                                                   │
│                     ▼                                                   │
│              ┌──────────────┐                                           │
│              │ RRF 融合排名 │   ← Reciprocal Rank Fusion               │
│              └──────┬───────┘                                           │
│                     │                                                   │
│                     ▼                                                   │
│              ┌──────────────┐                                           │
│              │ Rerank 精排  │   ← 多维度线性评分                        │
│              └──────┬───────┘                                           │
│                     │                                                   │
│                     ▼                                                   │
│              ┌──────────────┐                                           │
│              │ Lost-in-Middle│  ← 上下文重排 (最相关→首位/次相关→末位)   │
│              └──────────────┘                                           │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LLM 交互层 (AiClient.py)                          │
│  - chat(): 普通对话 / summarize                                         │
│  - run_agent(): 工具增强多轮对话 (MAX_STEPS=10)                         │
│  - 底层: openai SDK → MiniMax API (https://api.minimaxi.com/v1)        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 混合检索管道 (vector_store.py)

检索管道采用 **多路召回 → RRF 融合 → 精排 → Lost-in-Middle 重排** 的四级架构。

#### 3.1.1 三路召回

| 路径 | 数据源 | 粒度 | 作用 |
|------|--------|------|------|
| Dense | `android_child` (128字 child chunks) | 细粒度语义匹配 | 捕获 query 与文档的语义相似性 |
| HyDE | `android_hyde` (LLM 生成假设问题) | 问题→问题匹配 | 解决 query 与文档表述不一致问题 |
| BM25 | `android_parent` (全文倒排索引) | 关键词精确匹配 | 捕获专有名词、技术术语 |

#### 3.1.2 RRF 融合 (Reciprocal Rank Fusion)

```python
score(d) = Σ 1 / (rank + RRF_K)    # RRF_K = 60
```

不依赖原始向量距离或 BM25 分值，仅按排名融合，避免不同检索路的分值不可比问题。

#### 3.1.3 Rerank 精排 (reranker.py)

基于四维度线性组合计算 `final_score`：

```python
final_score =
    0.45 * rrf_score      +   # RRF 融合排名
    0.30 * term_overlap   +   # query 词项覆盖率
    0.15 * vector_inv     +   # 向量排名倒数
    0.10 * bm25_inv             # BM25 排名倒数
```

#### 3.1.4 Lost-in-Middle 重排

按 LLM 注意力分布理论重排上下文顺序：
- **最相关** → 首位（LLM 对开头注意力最强）
- **次相关** → 末位（LLM 对结尾注意力次强）
- **其余** → 中间

输入 `[rank1, rank2, rank3, rank4, rank5]` → 输出 `[rank1, rank3, rank4, rank5, rank2]`

#### 3.1.5 降级策略

若 `android_parent` / `android_child` / `android_hyde` 任一 collection 不存在，自动回退至旧版 `android_docs` 单路向量检索（`MAX_DISTANCE=1.2` 阈值过滤）。

---

### 3.2 离线索引构建 (init_vector_db_v2.py)

#### 3.2.1 Parent-Child 架构

| Collection | 内容 | chunk_size | overlap | 用途 |
|------------|------|------------|---------|------|
| `android_parent` | 原始文档分块 | 512字 | 100字 | 直接喂给 LLM 的上下文 |
| `android_child` | parent 内部细分 | 128字 | 25字 | 高精度向量语义匹配 |
| `android_hyde` | LLM 生成的假设问题 | — | — | 问题→问题匹配 |

#### 3.2.2 HyDE 问题生成

对每个 parent chunk，调用 MiniMax-M2.7 生成 3 个开发者可能提出的问题：

```python
prompt = "你是Android开发专家。以下是一段Android开发文档：\n\n{chunk}\n\n" \
         "请生成3个开发者可能会问的问题..."
```

生成后经过三级清洗：
1. 去除 `<think>...</think>` 思维链标签
2. 提取有效问题行（长度>5，含问号/句号）
3. 降级策略：从原始响应中提取含问号行

#### 3.2.3 Chunking 策略

```
按段落分割 → 超长段落按句分割 → 带重叠滑动窗口
```

优先保持段落完整性，避免语义断裂。

---

### 3.3 会话管理 (session.py)

#### 3.3.1 SessionData 结构

```python
class SessionData(TypedDict):
    history: list       # 最近对话记录 (role/content)
    summary: str       # 历史压缩摘要
    trimmed: list       # 被裁剪的旧消息（不丢弃）
    token_budget: int  # 剩余 token 预算 (4096)
```

#### 3.3.2 消息生命周期

```
用户请求
    │
    ▼
add_user_message(history) ──────────────────────────────┐
    │                                                   │
    ▼                                                   │
build_messages()                                        │
    ├── system prompt                                   │
    ├── [历史摘要] (若存在)                              │
    ├── history (最近4条)                               │
    ├── [压缩提示] (若 token 超预算)                     │
    └── 参考资料: query_db(req_msg)  ←──────────────────┘
    │
    ▼
ai_client.run_agent(messages) → AI 回复
    │
    ▼
add_assistant_message(history)
    │
    ▼
should_summarize()? → 是 → ai_client.summarize(history) → update_summary()
    │
    ▼
trim_history() → history 保留最近4条，旧消息移至 trimmed
```

**关键设计原则**：summarize 和 trim_history 都发生在 **AI 回复入栈之后**，确保 AI 已看过本轮消息再执行压缩/裁剪。

#### 3.3.3 Token 估算

采用字符数 / 4 的粗略估算（中文语境下可接受），避免引入 tiktoken 依赖。

---

### 3.4 LLM 客户端 (AiClient.py)

#### 3.4.1 接口设计

| 方法 | 用途 | 工具调用 |
|------|------|----------|
| `chat(messages, use_tools=False)` | 普通对话 / summarize | 可选单轮 |
| `run_agent(messages)` | 工具增强多轮对话 | 强制启用，最多10轮 |

#### 3.4.2 Tool Calling 循环

```
发送 messages + tools → LLM
    │
    ├── 返回 text → 直接返回
    │
    └── 返回 tool_calls → 执行函数 → 追加结果 → 再次请求 → 循环
```

当前注册工具（multiply.py）：
- `multiply(a: int, b: int)` → 计算乘积（示例工具）

---

## 4. 数据流

```mermaid
flowchart TD
    A[用户提问] --> B[/chat API]
    B --> C[获取 Session]
    C --> D[用户消息入 history]
    D --> E[query_db 混合检索]
    E --> F{三路召回}
    F --> F1[Dense 向量检索]
    F --> F2[HyDE 假设问题匹配]
    F --> F3[BM25 全文检索]
    F1 --> G[RRF 融合]
    F2 --> G
    F3 --> G
    G --> H[Rerank 精排]
    H --> I[Lost-in-Middle 重排]
    I --> J[构建 messages]
    J --> K[run_agent 调用 LLM]
    K --> L[AI 回复]
    L --> M[助手消息入 history]
    M --> N{should_summarize?}
    N -->|是| O[summarize 压缩]
    O --> P[更新 summary]
    N -->|否| Q[trim_history]
    P --> Q
    Q --> R[返回 reply]
```

---

## 5. 关键配置

| 配置项 | 文件 | 默认值 | 说明 |
|--------|------|--------|------|
| OPENAI_API_KEY | `.env` | — | MiniMax API Key |
| CHROMA_DB_PATH | `core/config.py` | 绝对路径 | 向量数据库持久化目录 |
| MAX_HISTORY | `session.py` | 4 | history 保留条数 |
| DEFAULT_TOKEN_BUDGET | `session.py` | 4096 | 会话 token 上限 |
| RRF_K | `vector_store.py` | 60 | RRF 平滑常数 |
| DENSE_TOP_K | `vector_store.py` | 10 | 向量检索返回数 |
| HYDE_TOP_K | `vector_store.py` | 5 | HyDE 返回数 |
| BM25_TOP_K | `vector_store.py` | 10 | BM25 返回数 |
| RERANK_TOP_K | `vector_store.py` | 5 | 最终喂给 LLM 的片段数 |

---

## 6. 运行流程

### 6.1 首次部署

```bash
# 1. 安装依赖
uv sync

# 2. 配置环境变量
cp ai_app1/.env.example ai_app1/.env
# 编辑 .env: OPENAI_API_KEY=your_minimax_key

# 3. 构建离线索引
uv run python -m ai_app1.pre.init_vector_db_v2

# 4. 验证索引
uv run python -m ai_app1.pre.verify_phase1  # 索引完整性
uv run python -m ai_app1.pre.verify_phase2  # 检索质量
uv run python -m ai_app1.pre.verify_phase3  # 端到端

# 5. 启动服务
uv run python -m ai_app1.main
```

### 6.2 API 调用

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Android 中 NullPointerException 如何解决？"}'
```

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 向量库 | ChromaDB | 本地持久化、零配置、Python 原生 |
| LLM | MiniMax-M2.7 | 中文能力强、API 兼容 OpenAI 格式 |
| 检索架构 | 多路混合 | 单一检索路召回率不足，混合互补 |
| 会话存储 | 内存字典 | 进程级简单实现，重启后丢失（可扩展至 Redis） |
| Token 估算 | 字符/4 | 避免 tiktoken 依赖，中文场景误差可接受 |
| summarize 时机 | AI 回复后 | 避免 AI 还没看消息就被压缩 |
| trim 时机 | AI 回复后 | 不丢失本轮对话内容 |
