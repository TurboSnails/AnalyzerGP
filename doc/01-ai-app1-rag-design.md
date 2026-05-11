# ai_app1 - Android RAG 问答系统设计文档

> 版本: 2.4 | 最后更新: 2026-05-11

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

方案 A：RRF 为主信号，term_overlap 为小量精度加成。RRF 已融合 dense / HyDE / BM25 三路排名，无需再单独引入 `vector_inv` / `bm25_inv`（否则两路排名信号被计两遍）。

```python
normalized_rrf = rrf_score / max_rrf    # max_rrf = 当前候选中的最大 rrf_score

final_score =
    0.80 * normalized_rrf  +   # RRF 融合排名（归一化后）；三路排名已在此汇聚
    0.20 * term_overlap        # query 词项覆盖率（0~1）；补充词面精度，权重宜小
```

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

| 方法 | 用途 | 工具调用 |
|------|------|----------|
| `chat(messages, use_tools=False)` | 普通对话 / summarize | 可选单轮 |
| `run_agent(messages)` | 工具增强多轮对话 | 强制启用，最多10轮 |

#### 3.5.2 Tool Calling 循环

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
| BGE_M3_PATH | `core/config.py` | 动态解析 | 本地 BGE-M3 模型目录，支持环境变量覆盖 |
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
| Embedding | BGE-M3 (本地) | 中文语义效果好、支持 L2 normalize、显式编码便于缓存和换模型 |
| 向量库 | ChromaDB | 本地持久化、零配置、Python 原生 |
| LLM | MiniMax-M2.7 | 中文能力强、API 兼容 OpenAI 格式 |
| 检索架构 | 多路混合 | 单一检索路召回率不足，混合互补 |
| 会话存储 | 内存字典 | 进程级简单实现，重启后丢失（可扩展至 Redis） |
| BM25 引擎 | Tantivy (Rust) + jieba | rank-bm25 全量内存方案在大语料下 OOM；Tantivy mmap 磁盘索引内存占用与查询量相关而非文档总量，jieba 分词精度更高，Rust 层 BM25 计算快 ~10× |
| 三路召回执行模式 | ThreadPoolExecutor 并发 | 三路无数据依赖，串行 ~185ms → 并发 ~90ms，TTFT 降低 ~50%；不改调用链（session.py 无感），线程池随 with 块自动回收 |
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

**修复**：改为 `\n`.join(f"{role}: {content}" for m in history) 生成人类可读的对话文本。已在 v2.2 中实施。

### 8.8 验收脚本硬编码绝对路径（H）

**风险**：[`verify_phase1.py`](ai_app1/pre/verify_phase1.py:25) 将 `CHROMA_DB_PATH` 写死为绝对路径 `/Users/hassan/Documents/workspace/aiFile/fenxiCB/ai_app1/pre/chroma_db`，导致脚本在任何其他机器或目录结构下直接失败，违背可移植性原则。

**修复**：从 `ai_app1.core.config` 导入 `CHROMA_DB_PATH`，与生产代码共用同一配置源。已在 v2.2 中实施。

### 8.9 Rerank 精排信号 double-counting（I）✅ 已修复（方案 A）

**原风险**：旧公式 `0.45*normalized_rrf + 0.30*term_overlap + 0.15*vector_inv + 0.10*bm25_inv` 中，RRF 本身已是 `Σ 1/(k+rank)` 的三路排名融合，再单独追加 `vector_inv` 和 `bm25_inv` 导致向量/BM25 信号被计两遍，实际有效权重严重偏离标注值。

**已实施（方案 A）**：去掉 `vector_inv` / `bm25_inv`，保留 RRF 作为主信号，term_overlap 作为小量词面精度加成：

```python
final_score = 0.80 * normalized_rrf + 0.20 * term_overlap
```

**长期优化方向**：引入 cross-encoder 模型（如 `BAAI/bge-reranker-v2-m3`，与已有 bge-m3 同系列，~1.1GB）替换整个线性组合。cross-encoder 将 query + doc 拼接后直接输出语义相关性分数，无需手写权重。对 top-20 候选精排，CPU（Apple M 系列）约 0.8~1.5 秒，可接受。

```python
from FlagEmbedding import FlagReranker
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)
scores = reranker.compute_score([(query, doc["text"]) for doc in candidates])
```
