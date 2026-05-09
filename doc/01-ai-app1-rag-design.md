# ai_app1 - Android RAG 问答系统设计文档

> 版本: 2.2 | 最后更新: 2026-05-09

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
| Dense | `android_child` → 回溯 `android_parent` | 细粒度语义匹配 | child 做 128 字向量检索，通过 `parent_id` 回溯取 512 字完整 parent 文本 |
| HyDE | `android_hyde` → 回溯 `android_parent` | 问题→问题匹配 | 假设问题向量匹配，通过 `parent_id` 回溯取完整 parent 文本 |
| BM25 | `android_parent` (全文倒排索引) | 关键词精确匹配 | 直接检索 512 字 parent，捕获专有名词、技术术语 |

> **父子回溯机制**：Dense 与 HyDE 两路均先在小粒度 collection（`android_child` / `android_hyde`）中做向量相似度检索，从命中结果的 `metadata.parent_id` 字段聚合去重，再按子文档的最小距离对父文档排序，最终拉取 `android_parent` 中的完整文本作为上下文。该设计的收益是：细粒度检索提升语义匹配精度，大粒度 parent 保证 LLM 获得完整、连贯的参考内容。

#### 3.1.2 RRF 融合 (Reciprocal Rank Fusion)

```python
score(d) = Σ 1 / (rank + RRF_K)    # RRF_K = 60
```

不依赖原始向量距离或 BM25 分值，仅按排名融合，避免不同检索路的分值不可比问题。

#### 3.1.3 Rerank 精排 (reranker.py)

基于四维度线性组合计算 `final_score`，计算前对 `rrf_score` 做 Min-Max 归一化（使其与 term_overlap、vector_inv、bm25_inv 处于同一 0~1 量级）：

```python
normalized_rrf = rrf_score / max_rrf    # max_rrf = 当前候选中的最大 rrf_score

final_score =
    0.45 * normalized_rrf  +   # RRF 融合排名（归一化后）
    0.30 * term_overlap    +   # query 词项覆盖率（0~1）
    0.15 * vector_inv      +   # 向量排名倒数（0~1）
    0.10 * bm25_inv            # BM25 排名倒数（0~1）
```

#### 3.1.4 Lost-in-Middle 重排

按 LLM 注意力分布理论重排上下文顺序：
- **最相关** → 首位（LLM 对开头注意力最强）
- **次相关** → 末位（LLM 对结尾注意力次强）
- **其余** → 中间

输入 `[rank1, rank2, rank3, rank4, rank5]` → 输出 `[rank1, rank3, rank4, rank5, rank2]`

#### 3.1.5 降级策略

若 `android_parent` / `android_child` / `android_hyde` 任一 collection 不存在，自动回退至旧版 `android_docs` 单路向量检索（`MAX_DISTANCE=1.2` 阈值过滤）。

#### 3.1.6 路A Dense 向量检索优化（已实施）

**优化前问题**：
1. 去重逻辑丢失信息 — 同一 `parent_id` 多个 `child` 命中时仅保留首个
2. 缺少 Distance 阈值过滤 — v2 Dense 路无距离过滤，可能引入噪声
3. `n_results` 偏小 — 128 字 child 粒度下 `DENSE_TOP_K=10` 经去重后 parent 数量不足

**已实施方案**（`vector_store.py`）：
- 新增 `_aggregate_parent_hits()` 通用聚合函数：
  - 过滤 `distance > MAX_CHILD_DISTANCE = 1.3` 的噪声
  - 同一 parent 聚合所有命中 child 的 distance
  - parent 级得分 = `min(distance) - 0.05 * (hit_count - 1)`（命中越多排序越靠前）
- `_query_dense`：增大 `n_results=DENSE_QUERY_K=25`，最终保留 `DENSE_TOP_K=10`
- `_query_hyde`：同步应用相同策略，`n_results=HYDE_QUERY_K=15`，保留 `HYDE_TOP_K=5`

**验证结果**：`verify_phase2` 13/13 通过 ✅

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

采用**加权字符数**估算（针对 Android 开发的中英文混合场景优化）：

```python
# 中文（含全角标点）~1.5 token/字；英文/代码符号 ~0.5 token/字符
cn_chars = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
other_chars = len(text) - cn_chars
tokens = int(cn_chars * 1.5 + other_chars * 0.5)
```

原方案 `字符数 / 4` 对中文严重低估，会导致 `token_budget` 未用完就过早触发 summarize。加权估算无需引入 tiktoken 依赖，同时显著提升中英文混合场景下的准确度。

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
| MAX_CHILD_DISTANCE | `vector_store.py` | 1.3 | child 层面向量距离阈值（过滤噪声） |
| DENSE_QUERY_K | `vector_store.py` | 25 | child 查询量 |
| DENSE_TOP_K | `vector_store.py` | 10 | 向量检索最终 parent 返回数 |
| HYDE_QUERY_K | `vector_store.py` | 15 | HyDE 查询量 |
| HYDE_TOP_K | `vector_store.py` | 5 | HyDE 最终 parent 返回数 |
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
| Token 估算 | 加权字符数 | 中文 ~1.5 token/字 + 英文/代码 ~0.5 token/字符，避免字符/4 对中文的低估 |
| summarize 时机 | AI 回复后 | 避免 AI 还没看消息就被压缩 |
| trim 时机 | AI 回复后 | 不丢失本轮对话内容 |

---

## 8. 已知风险与缓解措施

### 8.1 检索冗余与上下文污染（A）

**风险**：三路召回（Dense/HyDE/BM25）若命中同一 Parent Chunk，Rerank 后可能产生重复内容，浪费 Token 甚至触发 LLM 重复幻觉。

**缓解**：
- `query_db()` 在候选构建阶段已通过 `seen_ids: set[str]` 去重
- `rerank_chunks()` 返回后增加**运行时断言**：`assert len(ids) == len(set(ids))`
- 确保最终喂给 LLM 的 `RERANK_TOP_K` 个片段来自不同的 Parent

### 8.2 Rerank 线性评分量级不统一（B）

**风险**：`final_score = 0.45 * rrf_score + 0.30 * term_overlap + ...` 中，`rrf_score` 范围约 0~0.05，而 `term_overlap` 为 0~1，线性组合时 RRF 贡献被淹没。

**缓解**：`reranker.py` 中对 `rrf_score` 做 Min-Max 归一化：`normalized_rrf = rrf_score / max_rrf`，使所有子项处于 0~1 同一区间后再加权。

### 8.3 Token 估算的中英文混合陷阱（C）

**风险**：Android 开发场景包含大量代码（英文/符号密集），原 `字符数 / 4` 估算对中文严重低估，导致 `token_budget` 未用完就过早触发 summarize。

**缓解**：`session.py` 改进为加权估算：
- 中文字符（含全角标点）：~1.5 token/字
- 英文/数字/代码符号/半角标点：~0.5 token/字符

### 8.4 Summarize 的上下文断裂（D）

**风险**：长跨度多轮对话（如调试 Bug 超过 10 轮）中，第 5 轮的关键报错信息被压缩为概括性摘要，后续追问时 LLM 因摘要模糊而无法精确定位代码行号。

**缓解**：
- 当前保留最近 `MAX_HISTORY=4` 条原始消息，超出部分才 summarize
- **未来优化方向**：考虑使用「关键信息提取」替代简单摘要生成，或在 summarize 时保留关键代码片段、堆栈跟踪等结构化信息
### 8.5 Async 方法中误用同步 OpenAI 客户端（E）

**风险**：[`AiClient.chat()`](ai_app1/service/AiClient.py:37)、[`summarize()`](ai_app1/service/AiClient.py:132)、[`run_agent()`](ai_app1/service/AiClient.py:157) 均为 `async def`，但内部调用 `self.client.chat.completions.create()` 时使用了同步 `openai.OpenAI` 客户端。这会在 I/O 等待期间阻塞整个 FastAPI 事件循环，导致并发请求串行处理，严重降低吞吐量。

**修复**：将客户端替换为 `openai.AsyncOpenAI`，所有 `self.client.chat.completions.create(...)` 改为 `await self.client.chat.completions.create(...)`。已在 v2.2 中实施。

### 8.6 模块级冗余 AiClient 实例化（F）

**风险**：[`main.py`](ai_app1/main.py:10) 在模块导入时创建 `aiClient = AiClient(ai_api_key=OPENAI_API_KEY)`，该实例：
- 从未被使用（FastAPI 路由通过 `chat.py` 的 `get_ai_client()` 获取单例）
- 若 `OPENAI_API_KEY` 为空/缺失，会在服务启动前即崩溃，无法优雅降级

**修复**：删除 `main.py` 中的冗余实例化，完全由 `chat.py` 的依赖注入管理生命周期。已在 v2.2 中实施。

### 8.7 Summarize 输入格式为 Python repr（G）

**风险**：[`AiClient.summarize()`](ai_app1/service/AiClient.py:132) 将对话历史直接转为 `str(history)`，得到 Python 列表字面量表示（如 `"[{'role': 'user', 'content': '...'}, ...]"`）。LLM 难以解析这种机器格式，影响摘要质量。

**修复**：改为 `
`.join(f"{role}: {content}" for m in history) 生成人类可读的对话文本。已在 v2.2 中实施。

### 8.8 验收脚本硬编码绝对路径（H）

**风险**：[`verify_phase1.py`](ai_app1/pre/verify_phase1.py:25) 将 `CHROMA_DB_PATH` 写死为绝对路径 `/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db`，导致脚本在任何其他机器或目录结构下直接失败，违背可移植性原则。

**修复**：从 `ai_app1.core.config` 导入 `CHROMA_DB_PATH`，与生产代码共用同一配置源。已在 v2.2 中实施。
