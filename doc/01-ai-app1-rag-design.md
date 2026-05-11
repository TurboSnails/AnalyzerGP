# ai_app1 - Android RAG 问答系统设计文档

> 版本: 2.5 | 最后更新: 2026-05-11

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
│  2. 用户消息入栈 → build_messages（线程池执行，避免阻塞事件循环）       │
│  3. 调用 ai_client.stream_run_agent → 流式获取回复                     │
│  4. StreamingResponse 逐 token 返回给客户端                             │
│  5. 流结束后后台异步执行 summarize / trim_history                       │
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
│  - chat(): 普通对话 / summarize（非流式，兼容保留）                      │
│  - run_agent(): 工具增强多轮对话（非流式，兼容保留）                     │
│  - stream_chat(): 流式对话，逐 token yield                              │
│  - stream_run_agent(): 流式工具增强多轮对话（生产使用）                  │
│  - 底层: openai.AsyncOpenAI → MiniMax API                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 Embedding 服务 (embedding.py)

#### 3.1.1 设计原则

采用**显式编码**架构：由 `BgeM3EmbeddingService` 本地加载 BGE-M3 模型，显式调用 `encode()` 生成向量后，再通过 `add(embeddings=...)` / `query(query_embeddings=...)` 交给 ChromaDB。不把编码逻辑交给 Chroma 内置 `embedding_function`，带来以下收益：

- **可缓存**：embedding 结果可在应用层缓存，避免重复编码
- **可换模型**：更换模型只需替换 `BgeM3EmbeddingService`，Chroma 侧无感知
- **可插 pipeline**：支持在 encode 前后插入自定义预处理（如文本清洗、维度压缩）

#### 3.1.2 模型加载

```python
class BgeM3EmbeddingService:
    def __init__(self, model_path: str | None = None) -> None:
        self._path = model_path or BGE_M3_PATH
        self._model: SentenceTransformer | None = None
```

- 底层使用 `sentence_transformers.SentenceTransformer` 加载本地 BGE-M3
- **惰性加载**：首次调用 `encode()` 时才加载模型到内存
- **全局单例**：`get_embedding_service()` 提供进程级单例，避免重复加载

#### 3.1.3 向量生成

```python
def encode(self, texts: list[str], batch_size: int | None = 32) -> list[list[float]]:
    ...
    embeddings = self._model.encode(
        texts,
        normalize_embeddings=True,  # L2 normalize
        show_progress_bar=False,
    )
```

- **L2 归一化**：`normalize_embeddings=True`，与原先 Chroma 内嵌编码行为一致
- **分批编码**：长列表自动按 `batch_size` 切分，降低峰值显存占用
- **batch_size=None**：关闭分批，适合小批量低延迟场景

#### 3.1.4 模型路径解析

`BGE_M3_PATH` 在 `core/config.py` 中动态解析：

```python
def _resolve_bge_m3_path() -> str:
    """优先查找含权重的目录（pytorch_model.bin 或 model.safetensors）"""
    for base in (_REPO_ROOT, _AI_APP_ROOT):
        candidate = os.path.join(base, "models", "bge-m3")
        if os.path.isdir(candidate) and has_weights(candidate):
            return candidate
    return os.path.join(_AI_APP_ROOT, "models", "bge-m3")
```

- 优先在仓库根目录和 `ai_app1` 根目录查找 `models/bge-m3/`
- 通过检测 `pytorch_model.bin` 或 `model.safetensors` 确认目录含有效权重
- 支持通过环境变量 `BGE_M3_PATH` 强制覆盖

#### 3.1.5 生命周期管理

- **重置接口**：`reset_embedding_service()` 释放模型引用，用于测试或热替换模型
- **异常处理**：模型目录不存在时抛出 `FileNotFoundError`，提示下载命令

---

### 3.2 混合检索管道 (vector_store.py)

检索管道采用 **多路召回 → RRF 融合 → 精排 → Lost-in-Middle 重排** 的四级架构。

#### 3.2.1 三路召回

| 路径 | 数据源 | 粒度 | 作用 |
|------|--------|------|------|
| Dense | `android_child` → 回溯 `android_parent` | 细粒度语义匹配 | child 做 128 字向量检索，通过 `parent_id` 回溯取 512 字完整 parent 文本 |
| HyDE | `android_hyde` → 回溯 `android_parent` | 问题→问题匹配 | 假设问题向量匹配，通过 `parent_id` 回溯取完整 parent 文本 |
| BM25 | Tantivy 磁盘索引 (`tantivy_bm25/`) | 关键词精确匹配 | jieba 分词 + Rust BM25 引擎，磁盘持久化，捕获专有名词、技术术语 |

> **父子回溯机制**：Dense 与 HyDE 两路均先在小粒度 collection（`android_child` / `android_hyde`）中做向量相似度检索，从命中结果的 `metadata.parent_id` 字段聚合去重，再按子文档的最小距离对父文档排序，最终拉取 `android_parent` 中的完整文本作为上下文。该设计的收益是：细粒度检索提升语义匹配精度，大粒度 parent 保证 LLM 获得完整、连贯的参考内容。

#### 3.2.2 RRF 融合 (Reciprocal Rank Fusion)

```python
score(d) = Σ 1 / (rank + RRF_K)    # RRF_K = 60
```

不依赖原始向量距离或 BM25 分值，仅按排名融合，避免不同检索路的分值不可比问题。

#### 3.2.3 Rerank 精排 (reranker.py)

**已实施方案：CrossEncoder 语义重排（BgeRerankerService）**

引入 `sentence_transformers.CrossEncoder` 对 (query, doc) pair 直接预测语义相关性分数，替代旧版纯规则线性组合。

```python
class BgeRerankerService:
    def __init__(self, model_name_or_path: str | None = None):
        self._path = model_name_or_path or RERANKER_MODEL  # 默认 BAAI/bge-reranker-base
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()  # HF fast tokenizer 非线程安全，需序列化

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        ...
```

**精排流程：**
1. 对每个候选 chunk，构造 `[query, text]` pair
2. CrossEncoder 输出原始 logits → sigmoid 映射到 0~1：`ce_prob = 1 / (1 + exp(-score))`
3. 与 RRF 分数加权融合：
   ```python
   final_score = 0.75 * ce_norm + 0.25 * rrf_norm
   ```
   - CrossEncoder 语义分为主（0.75），RRF 召回分为辅（0.25）
4. 按 `final_score` 降序取 Top `RERANK_TOP_K`

**降级策略（fallback）：**
若 CrossEncoder 模型加载失败或推理异常，自动回退到旧版规则排序：
```python
final_score = 0.80 * normalized_rrf + 0.20 * term_overlap
```

**线程安全：**
- `CrossEncoder.predict()` 底层使用 HuggingFace fast tokenizer（Rust RefCell），不允许并发调用
- `BgeRerankerService` 内部用 `threading.Lock` 序列化所有 `predict` 调用

**模型路径：**
`RERANKER_MODEL` 在 `core/config.py` 中动态解析，优先查找本地 `models/bge-reranker-base/`，不存在则自动从 HuggingFace Hub 下载 `BAAI/bge-reranker-base`。

#### 3.2.4 Lost-in-Middle 重排

按 LLM 注意力分布理论重排上下文顺序：
- **最相关** → 首位（LLM 对开头注意力最强）
- **次相关** → 末位（LLM 对结尾注意力次强）
- **其余** → 中间

输入 `[rank1, rank2, rank3, rank4, rank5]` → 输出 `[rank1, rank3, rank4, rank5, rank2]`

#### 3.2.5 降级策略

若 `android_parent` / `android_child` / `android_hyde` 任一 collection 不存在，自动回退至旧版 `android_docs` 单路向量检索（`MAX_DISTANCE=1.2` 阈值过滤）。

#### 3.2.6 路A Dense 向量检索优化（已实施）

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

#### 3.2.7 三路并发召回（TTFT 优化）

**问题**：三路召回原为串行执行，Dense → HyDE → BM25 顺序等待，总耗时约 185ms，是 TTFT 的最大单点瓶颈。三路之间无数据依赖，天然适合并发。

**实现**：`query_db()` 内部用 `ThreadPoolExecutor(max_workers=3)` 将三路同时提交，主线程等待全部完成后再进入 RRF 融合。

```python
with ThreadPoolExecutor(max_workers=3) as pool:
    f_dense = pool.submit(_query_dense, query, col_child)
    f_hyde  = pool.submit(_query_hyde,  query, col_hyde)
    f_bm25  = pool.submit(bm25_store.search, query, BM25_TOP_K)
    dense_pids   = f_dense.result()
    hyde_pids    = f_hyde.result()
    bm25_results = f_bm25.result()
```

延迟对比：

| 阶段 | 串行 | 并发 | 说明 |
|------|------|------|------|
| Dense | ~90ms | — | BGE-M3 encode + ChromaDB query |
| HyDE | ~90ms | — | BGE-M3 encode + ChromaDB query |
| BM25 | ~5ms | — | Tantivy 磁盘检索 |
| **三路合计** | **~185ms** | **~90ms** | 瓶颈为 Dense/HyDE，BM25 完全隐藏 |

线程安全性：
- **PyTorch 推理**：`encode()` 计算期间释放 GIL，两路同时 encode 安全，共享 CPU 资源
- **ChromaDB 读操作**：`query()` / `get()` 为只读，多线程并发安全
- **Tantivy Searcher**：无状态只读，线程安全

日志中可直接观察实际耗时：`多路召回: dense=X, hyde=X, bm25=X | 耗时=XXms`

#### 3.2.8 路C BM25 稀疏检索实现（Tantivy + jieba）

**原方案问题**（rank-bm25 内存索引）：

| 问题 | 影响 |
|------|------|
| 全量文档 tokenize 后存内存（双份：原文 + token 序列） | 百万文档 → 数 GB 内存，OOM 风险 |
| 服务重启后需重建，首次查询延迟秒级 | 冷启动体验差 |
| 双字滑窗分词，专有名词切割错误率高 | BM25 召回质量下降 |

**现方案**：`bm25_store.py` 基于 Tantivy（Rust 搜索引擎）+ jieba 精确分词重写：

```
jieba.cut(text) → 空格连接 token 串 → Tantivy whitespace 分词器 → BM25Plus 评分
```

核心设计：
- **磁盘持久化**：索引落地至 `tantivy_bm25/`（与 `chroma_db/` 同级），mmap 读取，内存占用与查询量相关，而非文档总量
- **懒加载 + 持久化**：`search()` 首次调用时打开已有索引（毫秒级），索引为空则从 ChromaDB 分批构建（`_BATCH_SIZE=10_000`，避免 OOM）
- **增量写入**：`add_documents([(doc_id, text)])` 追加文档，无需全量重建
- **线程安全**：`_lock` 保护 `_ensure_loaded()` 双检锁，多并发安全
- **接口不变**：`search(query, top_k)` 与 `reload()` 签名与原版完全兼容，`vector_store.py` 无需改动

```python
# Schema 设计
doc_id   : TEXT, stored, tokenizer=raw        # 精确存储/检索，不全文分词
body     : TEXT, stored, tokenizer=whitespace  # jieba 预分词后存入，BM25 匹配目标
raw_text : TEXT, stored, tokenizer=raw        # 原始文本，命中后返回给调用方
```

性能对比：

| 方面 | rank-bm25（旧） | Tantivy + jieba（新） |
|------|-----------------|----------------------|
| 百万文档内存 | ~2 GB | ~几十 MB（热点 block） |
| 冷启动延迟 | 秒级（全量重建） | 毫秒级（打开已有索引） |
| 中文分词精度 | 双字滑窗（低） | jieba 精确模式（高） |
| BM25 计算 | Python（慢） | Rust（快 ~10×） |
| 增量更新 | 必须全量重建 | `add_documents()` 追加 |

---

### 3.3 离线索引构建 (init_vector_db_v2.py)

#### 3.3.1 Parent-Child 架构

| Collection | 内容 | chunk_size | overlap | 用途 |
|------------|------|------------|---------|------|
| `android_parent` | 原始文档分块 | 512字 | 100字 | 直接喂给 LLM 的上下文 |
| `android_child` | parent 内部细分 | 128字 | 25字 | 高精度向量语义匹配 |
| `android_hyde` | LLM 生成的假设问题 | — | — | 问题→问题匹配 |

#### 3.3.2 HyDE 问题生成

对每个 parent chunk，调用 MiniMax-M2.7 生成 3 个开发者可能提出的问题：

```python
prompt = "你是Android开发专家。以下是一段Android开发文档：\n\n{chunk}\n\n" \
         "请生成3个开发者可能会问的问题..."
```

生成后经过三级清洗：
1. 去除 `<think>...</think>` 思维链标签
2. 提取有效问题行（长度>5，含问号/句号）
3. 降级策略：从原始响应中提取含问号行

#### 3.3.3 Chunking 策略

```
按段落分割 → 超长段落按句分割 → 带重叠滑动窗口
```

优先保持段落完整性，避免语义断裂。

---

### 3.4 会话管理 (session.py)

#### 3.4.1 SessionData 结构

```python
class SessionData(TypedDict):
    history: list       # 最近对话记录 (role/content)
    summary: str       # 历史压缩摘要
    trimmed: list       # 被裁剪的旧消息（不丢弃）
    token_budget: int  # 剩余 token 预算 (4096)
```

#### 3.4.2 消息生命周期

```
用户请求
    │
    ▼
add_user_message(history) ──────────────────────────────┐
    │                                                   │
    ▼                                                   │
build_messages() (线程池，避免阻塞事件循环)              │
    ├── system prompt                                   │
    ├── [历史摘要] (若存在)                              │
    ├── history (最近4条)                               │
    ├── [压缩提示] (若 token 超预算)                     │
    └── 参考资料: query_db(req_msg)  ←──────────────────┘
    │
    ▼
ai_client.stream_run_agent(messages) → 流式生成 AI 回复
    │
    ▼
StreamingResponse 逐 token 返回给客户端
    │
    ▼
流结束 → asyncio.create_task(background_maintain_session())
    │
    ├── add_assistant_message(history)
    ├── should_summarize()? → 是 → ai_client.summarize(history) → update_summary()
    └── trim_history() → history 保留最近4条，旧消息移至 trimmed
```

**关键设计原则**：
- `summarize` 和 `trim_history` 发生在 **流式传输结束后**，确保 AI 已完整生成回复
- 通过 `asyncio.create_task(background_maintain_session())` **后台异步执行**，不阻塞客户端下一条请求
- `build_messages()` 中的 `query_db()` 含模型推理，通过 `asyncio.to_thread()` 放入线程池避免阻塞 FastAPI 事件循环

#### 3.4.3 Token 估算

采用**加权字符数**估算（针对 Android 开发的中英文混合场景优化）：

```python
# 中文（含全角标点）~1.5 token/字；英文/代码符号 ~0.5 token/字符
cn_chars = len(re.findall(r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]", text))
other_chars = len(text) - cn_chars
tokens = int(cn_chars * 1.5 + other_chars * 0.5)
```

原方案 `字符数 / 4` 对中文严重低估，会导致 `token_budget` 未用完就过早触发 summarize。加权估算无需引入 tiktoken 依赖，同时显著提升中英文混合场景下的准确度。

---

### 3.5 LLM 客户端 (AiClient.py)

#### 3.5.1 接口设计

| 方法 | 用途 | 流式 | 工具调用 |
|------|------|------|----------|
| `chat(messages, use_tools=False)` | 普通对话 / summarize | 否 | 可选单轮 |
| `run_agent(messages)` | 工具增强多轮对话 | 否 | 强制启用，最多10轮 |
| `stream_chat(messages, use_tools=False)` | 流式普通对话 | 是 | 可选单轮 |
| `stream_run_agent(messages)` | **生产主入口：流式工具增强多轮对话** | 是 | 强制启用，最多10轮 |

#### 3.5.2 Tool Calling 循环（非流式）

```
发送 messages + tools → LLM
    │
    ├── 返回 text → 直接返回
    │
    └── 返回 tool_calls → 执行函数 → 追加结果 → 再次请求 → 循环
```

#### 3.5.3 流式 Tool Calling 循环（生产使用）

`stream_run_agent()` 在流式响应中**增量收集 `tool_calls`**，无需额外非流式请求：

```python
async for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        yield delta.content          # 实时 yield 给客户端
    if delta.tool_calls:
        for tc in delta.tool_calls:
            # 增量拼接 tool_call 各字段
            tool_calls[idx]["function"]["arguments"] += tc.function.arguments
```

- 每轮流式输出直接 yield token，用户首 token 可见时间（TTFT）最优
- tool_calls 在流中增量组装完整后执行工具，追加结果进入下一轮循环
- 最多 `MAX_STEPS=10` 轮，与 `run_agent()` 行为一致

当前注册工具（[`tools.py`](ai_app1/service/tools.py:1)）：
- `multiply(a: int, b: int)` → 计算乘积（示例工具）

---

## 4. 数据流

```mermaid
flowchart TD
    A[用户提问] --> B[/chat API]
    B --> C[获取 Session]
    C --> D[用户消息入 history]
    D --> E[query_db 混合检索<br/>asyncio.to_thread]
    E --> F{三路并发召回}
    F --> F1[Dense 向量检索]
    F --> F2[HyDE 假设问题匹配]
    F --> F3[BM25 全文检索]
    F1 --> G[RRF 融合]
    F2 --> G
    F3 --> G
    G --> H[Rerank 精排<br/>CrossEncoder]
    H --> I[Lost-in-Middle 重排]
    I --> J[构建 messages]
    J --> K[stream_run_agent 流式调用 LLM]
    K --> L[StreamingResponse<br/>逐 token 返回]
    L --> M[流结束 → 后台任务]
    M --> N[助手消息入 history]
    N --> O{should_summarize?}
    O -->|是| P[summarize 压缩]
    P --> Q[更新 summary]
    O -->|否| R[trim_history]
    Q --> R
    R --> S[Session 维护完成]
```

---

## 5. 关键配置

| 配置项 | 文件 | 默认值 | 说明 |
|--------|------|--------|------|
| OPENAI_API_KEY | `.env` | — | MiniMax API Key |
| BGE_M3_PATH | `core/config.py` | 动态解析 | 本地 BGE-M3 模型目录，支持环境变量覆盖 |
| RERANKER_MODEL | `core/config.py` | 动态解析 | CrossEncoder 模型路径，默认 `BAAI/bge-reranker-base`，支持环境变量覆盖 |
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

# 5. 启动服务（自动预热 Embedding / Reranker / BM25 模型）
uv run python -m ai_app1.main
```

**启动预热（`preload_models`）：**
服务启动时通过 `@app.on_event("startup")` 自动预热所有模型和索引：
- Embedding 模型（BGE-M3）
- CrossEncoder Reranker（bge-reranker-base）
- BM25 磁盘索引（Tantivy）

避免首个用户请求承担懒加载成本，提升冷启动体验。

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
| Embedding | BGE-M3 (本地) | 中文语义效果好、支持 L2 normalize、显式编码便于缓存和换模型 |
| 向量库 | ChromaDB | 本地持久化、零配置、Python 原生 |
| LLM | MiniMax-M2.7 | 中文能力强、API 兼容 OpenAI 格式 |
| 检索架构 | 多路混合 | 单一检索路召回率不足，混合互补 |
| 会话存储 | 内存字典 | 进程级简单实现，重启后丢失（可扩展至 Redis） |
| BM25 引擎 | Tantivy (Rust) + jieba | rank-bm25 全量内存方案在大语料下 OOM；Tantivy mmap 磁盘索引内存占用与查询量相关而非文档总量，jieba 分词精度更高，Rust 层 BM25 计算快 ~10× |
| 三路召回执行模式 | ThreadPoolExecutor 并发 | 三路无数据依赖，串行 ~185ms → 并发 ~90ms，TTFT 降低 ~50%；不改调用链（session.py 无感），线程池随 with 块自动回收 |
| Reranker | CrossEncoder (bge-reranker-base) | 语义相关性打分优于规则线性组合；sigmoid 归一化后与 RRF 天然同区间；HF 生态成熟，本地可部署 |
| 响应模式 | StreamingResponse 流式输出 | 用户首 token 即可见，TTFT 最优；流中增量收集 tool_calls，无需额外请求 |
| Session 维护 | 后台异步（create_task） | summarize/trim 不阻塞客户端下一条请求；流结束后再维护，保证 AI 已完整生成回复 |
| 模型预热 | 启动时 preload_models | 避免首个请求承担 Embedding/CrossEncoder 懒加载成本（3~5s） |
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

### 8.2 Rerank 线性评分量级不统一（B）✅ 已修复（CrossEncoder 替代）

**风险**：旧版规则排序 `final_score = 0.45 * rrf_score + 0.30 * term_overlap + ...` 中，`rrf_score` 范围约 0~0.05，而 `term_overlap` 为 0~1，线性组合时 RRF 贡献被淹没。

**修复**：`reranker.py` 中先对 `rrf_score` 做 Min-Max 归一化：`normalized_rrf = rrf_score / max_rrf`，使所有子项处于 0~1 同一区间后再加权。已在 v2.4 前实施。

**v2.5 进一步升级**：引入 `CrossEncoder` 语义重排替代纯规则线性组合。CrossEncoder 直接输出 (query, doc) 语义相关性分数，通过 sigmoid 映射到 0~1，与 RRF 归一化分数天然同区间，彻底消除量级不统一问题。

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

### 8.5 Async 方法中误用同步 OpenAI 客户端（E）✅ 已修复（v2.2）

**风险**：`AiClient.chat()`、`summarize()`、`run_agent()` 均为 `async def`，但内部调用 `self.client.chat.completions.create()` 时使用了同步 `openai.OpenAI` 客户端。这会在 I/O 等待期间阻塞整个 FastAPI 事件循环，导致并发请求串行处理，严重降低吞吐量。

**修复**：将客户端替换为 `openai.AsyncOpenAI`，所有 `self.client.chat.completions.create(...)` 改为 `await self.client.chat.completions.create(...)`。已在 v2.2 中实施。

### 8.6 模块级冗余 AiClient 实例化（F）✅ 已修复（v2.2）

**风险**：`main.py` 在模块导入时创建 `aiClient = AiClient(ai_api_key=OPENAI_API_KEY)`，该实例：
- 从未被使用（FastAPI 路由通过 `chat.py` 的 `get_ai_client()` 获取单例）
- 若 `OPENAI_API_KEY` 为空/缺失，会在服务启动前即崩溃，无法优雅降级

**修复**：删除 `main.py` 中的冗余实例化，完全由 `chat.py` 的依赖注入管理生命周期。已在 v2.2 中实施。

### 8.7 Summarize 输入格式为 Python repr（G）✅ 已修复（v2.2）

**风险**：`AiClient.summarize()` 将对话历史直接转为 `str(history)`，得到 Python 列表字面量表示（如 `"[{'role': 'user', 'content': '...'}, ...]"`）。LLM 难以解析这种机器格式，影响摘要质量。

**修复**：改为 `\n`.join(f"{role}: {content}" for m in history) 生成人类可读的对话文本。已在 v2.2 中实施。

### 8.8 验收脚本硬编码绝对路径（H）✅ 已修复（v2.2）

**风险**：`verify_phase1.py` 将 `CHROMA_DB_PATH` 写死为绝对路径 `/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db`，导致脚本在任何其他机器或目录结构下直接失败，违背可移植性原则。

**修复**：从 `ai_app1.core.config` 导入 `CHROMA_DB_PATH`，与生产代码共用同一配置源。已在 v2.2 中实施。

### 8.9 Rerank 精排信号 double-counting（I）✅ 已修复（方案 A）→ ✅ v2.5 升级为 CrossEncoder

**原风险**：旧公式 `0.45*normalized_rrf + 0.30*term_overlap + 0.15*vector_inv + 0.10*bm25_inv` 中，RRF 本身已是 `Σ 1/(k+rank)` 的三路排名融合，再单独追加 `vector_inv` 和 `bm25_inv` 导致向量/BM25 信号被计两遍，实际有效权重严重偏离标注值。

**v2.4 方案 A**：去掉 `vector_inv` / `bm25_inv`，保留 RRF 作为主信号，term_overlap 作为小量词面精度加成：

```python
final_score = 0.80 * normalized_rrf + 0.20 * term_overlap
```

**v2.5 升级为 CrossEncoder**：引入 `BgeRerankerService`（基于 `sentence_transformers.CrossEncoder`，默认 `BAAI/bge-reranker-base`）替换整个规则线性组合。CrossEncoder 将 query + doc 拼接后直接输出语义相关性分数，经 sigmoid 归一化后与 RRF 融合：

```python
final_score = 0.75 * ce_norm + 0.25 * rrf_norm
```

- 彻底消除手写权重和信号 double-counting 问题
- 降级策略：CrossEncoder 异常时自动回退到方案 A 规则排序

### 8.10 流式响应中的 Session 竞争条件（J）

**风险**：`chat()` 使用 `StreamingResponse` 流式返回 AI 回复，而 `summarize()` 和 `trim_history()` 在流结束后通过 `asyncio.create_task(background_maintain_session())` 后台异步执行。若用户在流尚未结束时立即发送第二条请求，`build_messages()` 可能在 `add_assistant_message()` 和 `trim_history()` 之前读取到旧的 history 状态，导致上下文不一致。

**缓解**：
- 当前场景下用户为单一会话（`default_user`），且正常交互节奏下流结束后才发下一条请求，风险较低
- **长期优化**：将 session 维护操作移到流结束前同步完成，或引入 asyncio.Lock 保护 session 读写

### 8.11 CrossEncoder 线程安全问题（K）

**风险**：`CrossEncoder.predict()` 底层使用 HuggingFace fast tokenizer（Rust RefCell），多个线程同时调用会导致 `AlreadyBorrowedError` 崩溃。

**缓解**：`BgeRerankerService` 内部使用 `threading.Lock` 序列化所有 `predict` 调用。虽然锁会串行化并发请求的 rerank 阶段，但 rerank 仅对 top-20 候选执行，耗时 < 100ms，对整体 TTFT 影响可接受。

### 8.12 启动冷加载延迟（L）

**风险**：Embedding 和 CrossEncoder 模型均为惰性加载，首个请求时才初始化，导致首次 TTFT 显著增加（BGE-M3 加载约 3~5 秒，CrossEncoder 加载约 2~3 秒）。

**缓解**：`main.py` 通过 `@app.on_event("startup") preload_models()` 在服务器启动时预热所有模型和索引：
- `get_embedding_service()._ensure_model()`
- `_get_reranker_service()._ensure_model()`
- `bm25_search("", 1)` 触发 BM25 索引加载

确保首个用户请求到达时所有组件已就绪。

---

## 9. 版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| v2.5 | 2026-05-11 | 引入 CrossEncoder 语义重排（BgeRerankerService）；API 全面流式化（`stream_run_agent` + `StreamingResponse`）；session 后台异步维护；启动自动预热模型 |
| v2.4 | 2026-05-10 | 三路并发召回（ThreadPoolExecutor）；路A Dense 聚合优化；RRF + term_overlap 方案 A |
| v2.2 | 2026-05-08 | 修复 AsyncOpenAI 客户端；删除冗余 AiClient 实例化；修复 summarize 输入格式；修复验收脚本硬编码路径 |
| v2.0 | 2026-05-06 | Parent-Child 架构 + HyDE + BM25 多路混合检索 |
| v1.0 | 2026-05-04 | 初始版本：单路向量检索 + 基础会话管理 |
