# ai_app1 - 多领域 RAG 问答系统设计文档

> 版本: 3.5 | 最后更新: 2026-05-16

---

## 1. 系统定位

ai_app1 是一个支持 **多领域** 的 **智能问答助手**，基于 RAG (Retrieval-Augmented Generation) 架构构建。系统可同时加载多个 DomainPlugin（如 Android 开发、MS MARCO 通用英文问答等），通过统一的知识库 collection + `domain` 元数据过滤实现领域隔离，为本地 Qwen2.5 或远程 MiniMax-M2.7 等大模型提供精准上下文。

**v3.5 核心变化**：改写与生成 LLM 解耦 — `RAGSettings` 新增独立的 `rewriter_llm_*` 配置组，`RAGContainer` 同时持有 `llm`（主生成模型，默认远程 MiniMax）与 `rewriter_llm`（改写专用模型，默认本地 Qwen），改写器（`LLMQueryRewriter`）与最终对话流使用不同 LLM 实例。`ai_app3` 引入 `llm_provider.py` 统一接管 LangChain `ChatOpenAI` 创建，消除全部硬编码模型名与 API Key。

**v3.4 核心变化**：统一 Collection 架构 — 所有领域数据写入同一个 `knowledge_base` collection，通过 `domain` metadata 字段实现领域隔离检索，无需为每个领域维护独立 collection 和 BM25 索引。新增多领域容器并发加载、自动路由（中文→android，英文→默认）、跨领域融合（`domain: "all"`）能力。

**v3.3 回顾**：新增完整的评测与可观测性体系 - Query 自动分类统计、Rewrite/Rerank 效果量化评测、Retrieval Trace 全链路追踪、Latency Breakdown 延迟拆解、Failure Analysis 失败分析闭环、Comprehensive Eval 综合调度器。HybridRetriever 和 SessionManager 已内嵌 Trace 记录与失败样本自动收集。

> **v3.2 回顾**：引入工厂注册表（`rag_framework/core/factories.py`）+ 生命周期协议（`Warmupable`/`Closable`），实现组件热插拔；`RAGContainer` 改为不可变 dataclass；移除所有 import-time 副作用，领域插件改为 lifespan 显式注册；API 层零全局状态，通过 FastAPI `Depends` 注入容器。

---

## 2. 三层架构总览

```
┌─────────────────────────────────────────────────────────┐
│            应用层 (ai_app1/)                              │
│   main.py + api/chat.py  →  FastAPI HTTP 路由            │
│   职责：lifespan 显式注册多领域、创建共享容器、协议化预热     │
└─────────────────────────┬───────────────────────────────┘
                           │  import rag_framework
                           ▼
┌─────────────────────────────────────────────────────────┐
│          领域插件层 (domains/*/)                           │
│   AndroidDomainPlugin / MSMarcoDomainPlugin              │
│   职责：集合命名、术语词典、查询路由规则、HyDE Prompt        │
│   特点：import-time 零副作用，由 lifespan 显式注册           │
│   统一 collection：knowledge_base（通过 metadata.domain 隔离）│
└─────────────────────────┬───────────────────────────────┘
                           │  implement DomainPlugin
                           ▼
┌─────────────────────────────────────────────────────────┐
│        框架层 (rag_framework/ — 可安装 Python 包)          │
│   RAGContainer(frozen) · SessionManager · HybridRetriever│
│   VectorIndexer · STEmbedder · CrossEncoderReranker     │
│   QueryRewriter · OpenAILLMClient / LocalLLMClient      │
│   ChromaVectorStore · BM25Store · Factories / Lifecycle │
│   职责：通用 RAG 管道 + 可插拔组件注册表 + 生命周期协议      │
│   新增：DenseStore.where 过滤、BM25Store.domain 字段、     │
│         HybridRetriever.domain_filter 领域隔离检索         │
└─────────────────────────────────────────────────────────┘
```

### 2.1 目录结构

```
 fenxiCB/
 ├── rag_framework/                  # 框架包源码
 │   └── rag_framework/
 │       ├── container.py            # RAGContainer — 不可变依赖注入总控
 │       ├── core/
 │       │   ├── config.py           # RAGSettings（纯配置，去路径化）
 │       │   ├── factories.py        # 组件工厂注册表（热插拔）
 │       │   ├── lifecycle.py        # Warmupable / Closable 协议
 │       │   ├── registry.py         # 领域插件注册表
 │       │   ├── logger.py           # 结构化日志
 │       │   └── exceptions.py       # 异常基类
 │       ├── domain/base.py          # DomainPlugin 抽象基类
 │       ├── embedding/              # STEmbedder + 自注册工厂
 │       ├── indexing/               # VectorIndexer / chunker / hyde
 │       ├── llm/                    # OpenAILLMClient / LocalLLMClient / tool_registry
 │       ├── rerank/                 # CrossEncoderReranker / FallbackReranker
 │       ├── retrieval/              # ChromaVectorStore / HybridRetriever / BM25Store
 │       │   └── query_rewriter/     # Rule / LLM / Qwen Rewriter
 │       ├── session/                # SessionManager / memory_store
 │       └── eval/                   # 评测框架
 ├── domains/android/
 │   ├── android_domain/plugin.py    # AndroidDomainPlugin（无 import-time 注册）
 │   └── scripts/
 │       ├── init_vector_db_v2.py    # 生产索引脚本
 │       └── init_vector_db.py       # V1（已废弃）
 ├── domains/msmarco/
 │   ├── msmarco_domain/plugin.py    # MSMarcoDomainPlugin（英文问答领域）
 │   └── scripts/
 │       └── download_and_index.py   # SQuAD → 统一 knowledge_base 索引
 ├── ai_app1/                        # 薄应用层
 │   ├── main.py                     # FastAPI app + lifespan（多领域容器组装）
 │   ├── api/chat.py                 # /chat 路由 + 单领域/跨领域融合
 │   ├── scripts/                    # 模型下载辅助脚本
 │   └── tests/test_api.py           # API 端到端测试（lifespan 后注入 mock）
 └── doc/
```

---

## 3. 核心组件设计

### 3.1 依赖注入容器 (RAGContainer)

`rag_framework/container.py` 是系统的组装中心。`RAGContainer` 为 **不可变 frozen dataclass**，通过工厂注册表创建所有组件，实现热插拔：

```python
@dataclass(frozen=True, slots=True)
class RAGContainer:
    settings:        RAGSettings
    embedder:        Embedder              # BGE-M3 向量编码
    vector_store:    VectorStore           # ChromaDB（统一抽象接口）
    retriever:       Retriever             # HybridRetriever（多路召回 + RRF）
    reranker:        Reranker              # CrossEncoderReranker 语义精排
    llm:             LLMClient             # 主生成 LLM（默认远程 MiniMax）
    rewriter_llm:    LLMClient             # 改写专用 LLM（默认本地 Qwen）
    session_store:   SessionStore          # MemorySessionStore
    domain:          DomainPlugin          # Android 领域插件
    rule_rewriter:   QueryRewriter | None  # L1 规则改写器
    llm_rewriter:    QueryRewriter | None  # L2 LLM 改写器

    @classmethod
    def from_settings(cls, settings: RAGSettings | None = None) -> "RAGContainer":
        # 1. 创建主 LLM（用于最终对话生成 / summarize）
        llm = llm_registry.create(settings.llm_backend, ...)
        # 2. 创建 rewriter_llm（用于查询改写；可独立配置 backend/model/base_url）
        rewriter_llm = llm_registry.create(
            settings.resolved_rewriter_llm_backend,
            base_url=settings.rewriter_llm_base_url,
            api_key=settings.rewriter_llm_api_key,
            model=settings.rewriter_llm_model,
            ...
        )
        # 3. 改写器显式注入 rewriter_llm，而非主 llm
        llm_rewriter = rewriter_registry.create("llm", llm=rewriter_llm)
        # ... 其余组件均通过注册表创建，无硬编码 new
        ...

    async def chat_stream(self, query: str, user_id: str):
        manager = SessionManager(...)
        async for chunk in manager.chat_stream(query, user_id):
            yield chunk

    def warmup_targets(self) -> list[Warmupable]:
        """返回所有支持预热的组件，供 lifespan 统一调用。"""
        return [comp for comp in (... ) if isinstance(comp, Warmupable)]
```

**ai_app1/api/chat.py** 使用 FastAPI `Depends` 从 `app.state` 注入容器，零全局状态：

```python
def get_container(request: Request) -> RAGContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:
        raise RuntimeError("RAGContainer 未初始化，请检查 lifespan 是否已执行")
    return container

@router.post("/chat")
async def chat(req: ChatRequest, container: RAGContainer = Depends(get_container)):
    async def content_generator():
        async for chunk in container.chat_stream(req.message, req.user_id):
            yield chunk
    return StreamingResponse(content_generator(), media_type="text/event-stream")
```

---

### 3.2 配置系统 (RAGSettings)

`rag_framework/core/config.py` 基于 Pydantic BaseSettings，环境变量前缀 `RAG_`。

**v3.2 变化**：配置类精简为纯配置，路径解析函数保留但移至工厂函数中作为默认策略，不直接耦合在 `RAGSettings` 的构建流程里。

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `llm_backend` | `"local"` | 主 LLM 后端，支持 `local` / `minimax` / `openai` / `ollama` |
| `llm_base_url` | 按 backend 预设 | `local` 为空串；minimax / openai / ollama 自动填充 |
| `llm_model` | 按 backend 预设 | local: `qwen2.5-1.5b-instruct`；minimax: `MiniMax-M2.7` |
| `llm_api_key` | 回退链解析 | 配置值 → `OPENAI_API_KEY` 环境变量 → backend 默认 |
| `llm_max_tokens` | `512` | 主 LLM 最大生成 token 数 |
| `llm_max_concurrent` | `3` | LLM 并发请求限制 |
| `rewriter_llm_backend` | `""`（空） | 改写专用 LLM 后端；**空则自动回退到 `llm_backend`** |
| `rewriter_llm_base_url` | `""` | 改写 LLM base_url；空则回退到 `llm_base_url` |
| `rewriter_llm_model` | `""` | 改写 LLM 模型名；空则回退到 `llm_model` |
| `rewriter_llm_api_key` | `""` | 改写 LLM API Key；空则回退到 `llm_api_key` |
| `rewriter_llm_max_tokens` | `128` | 改写 LLM 最大生成 token 数（rewrite 场景通常只需短输出） |
| `rewriter_llm_local_model_path` | `""` | 本地改写模型路径；空则回退到 `llm_local_model_path` |
| `embed_backend` | `"sentence_transformer"` | 通过工厂注册表选择 embedder 实现 |
| `vector_store_backend` | `"chroma"` | 通过工厂注册表选择向量库实现 |
| `reranker_backend` | `"cross_encoder"` | 通过工厂注册表选择 reranker 实现 |
| `retriever_backend` | `"hybrid"` | 通过工厂注册表选择检索器实现 |
| `session_store_backend` | `"memory"` | 通过工厂注册表选择会话存储实现 |
| `chroma_db_path` | 动态解析 | 优先 `ai_app1/data/chroma_db`，否则 `pre/chroma_db` |
| `bm25_path` | 依 chroma 父目录 | `<chroma_parent>/tantivy_bm25/` |
| `embed_model_path` | 动态解析 | `models/bge-m3/` 含权重目录 |
| `reranker_model_path` | 动态解析 | 本地 `models/bge-reranker-base/` 或 HF Hub |
| `llm_local_model_path` | 动态解析 | `models/qwen2.5-1.5b-instruct/` |

> **v3.5 配置回退机制**：所有 `rewriter_llm_*` 字段若留空（或仅含空白字符），`RAGSettings` 的 `field_validator` 会自动将其解析为对应主 `llm_*` 的值；`resolved_rewriter_llm_backend` 属性在 `rewriter_llm_backend` 为空时返回 `llm_backend`。这意味着**不填 rewriter 配置 = 完全兼容旧行为**，升级零成本。

配置加载顺序：`.env` → `ai_app1/.env`（`override=False`，不覆盖已有环境变量）。

**各后端预设：**

| backend | base_url | model |
|---------|----------|-------|
| `local` | `''` | `qwen2.5-1.5b-instruct` |
| `minimax` | `https://api.minimaxi.com/v1` | `MiniMax-M2.7` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `ollama` | `http://127.0.0.1:11434/v1` | `qwen2.5:1.5b-instruct-q4_K_M` |

---

### 3.3 工厂注册表 (Component Factories)

`rag_framework/core/factories.py` 实现泛型组件注册表，每个组件类型独立管理：

```python
# 全局注册表实例
embedder_registry     = _Registry[Any]("embedder")
vector_store_registry = _Registry[Any]("vector_store")
llm_registry          = _Registry[Any]("llm")
reranker_registry     = _Registry[Any]("reranker")
session_store_registry = _Registry[Any]("session_store")
rewriter_registry     = _Registry[Any]("rewriter")
retriever_registry    = _Registry[Any]("retriever")
```

各实现类在模块底部通过 `register_xxx()` 自注册，例如 `LocalLLMClient`：

```python
def _create_local_llm(model_path: str = "", ...):
    return LocalLLMClient(model_path=model_path, device="cpu", dtype="float32", ...)

register_llm("local", _create_local_llm)
```

**生命周期协议**（`rag_framework/core/lifecycle.py`）：

```python
@runtime_checkable
class Warmupable(Protocol):
    async def warmup(self) -> None: ...

@runtime_checkable
class Closable(Protocol):
    async def shutdown(self) -> None: ...
```

`lifespan` 通过 `isinstance(comp, Warmupable)` 统一调用，无需关心具体实现类。

---

### 3.4 领域插件系统 (DomainPlugin)

`rag_framework/domain/base.py` 定义抽象接口，将所有 Android 特定知识收敛到 `AndroidDomainPlugin`：

```python
class DomainPlugin(ABC):
    def classify_query(self, query: str, history: list) -> str:
        """返回查询类型：original / semantic / keyword / api"""

    def get_collection_names(self) -> CollectionNames:
        """返回三个 collection 的名称"""
        # android_parent / android_child / android_hyde

    def get_term_mapping(self) -> dict[str, str]:
        """中文术语 → 英文 keyword 映射（L1 规则扩写用）"""

    def rewrite_router_rules(self, query: str, history: list) -> int:
        """返回 0 / 1 / 2 决定扩写级别"""

    def get_hyde_prompt(self, chunk: str) -> str:
        """返回 HyDE 问题生成 Prompt"""
```

**AndroidDomainPlugin 策略**（`domains/android/android_domain/plugin.py`）：

- `classify_query`：检测 Android 组件名（Activity、Fragment、RecyclerView 等）、camelCase、异常命名等代码模式
- `get_collection_names`：`knowledge_base` / `knowledge_base_child` / `knowledge_base_hyde`
- `get_term_mapping`：~25 个高频中文技术词 → 英文 keyword（内存泄漏→memory leak、卡顿→ANR jank 等）
- `rewrite_router_rules`：L0 passthrough / L1 规则扩展 / L2 LLM 重写分级逻辑

> **v3.4 统一 Collection**：所有领域共用同一组 collection（`knowledge_base` / `knowledge_base_child` / `knowledge_base_hyde`），领域隔离通过 metadata 中的 `"domain": "android"` 或 `"domain": "msmarco"` 实现，检索时通过 `where={"domain": {"$eq": "..."}}`（dense）或 `+domain:...`（BM25）过滤。

> **去副作用化**：模块底部不再执行 `register_domain(AndroidDomainPlugin)`，改为 `ai_app1/main.py` lifespan 中显式注册。

---

### 3.5 Embedding 服务 (STEmbedder)

`rag_framework/embedding/sentence_transformer.py`：

- 底层：`sentence_transformers.SentenceTransformer` 加载本地 BGE-M3
- **惰性加载**：首次 `encode()` 时才加载模型
- **L2 归一化**：`normalize_embeddings=True`
- **分批编码**：`batch_size=32`，可设 `None` 关闭
- 模块底部通过 `register_embedder("sentence_transformer", _create_st_embedder)` 自注册到工厂注册表

---

### 3.6 离线索引构建

#### 3.5.1 VectorIndexer（统一索引编排器）

`rag_framework/indexing/indexer.py` 封装完整的三路索引管道：

```python
@dataclass
class IndexConfig:
    chunk_size:       int  = 512   # parent chunk 大小
    overlap:          int  = 64    # parent overlap
    child_chunk_size: int  = 128   # child chunk 大小
    child_overlap:    int  = 25    # child overlap
    enable_child:     bool = True  # 是否生成 child collection
    enable_hyde:      bool = True  # 是否生成 HyDE questions
    enable_bm25:      bool = True  # 是否写入 BM25 索引

@dataclass
class IndexStats:
    total_files:    int
    total_chunks:   int
    hyde_generated: int
    errors:         list[str]
```

**索引流程：**

```
文档输入 (文件路径 / 内存文本)
    │
    ▼
chunker.chunk_text(size=512, overlap=64)  →  parent chunks + UUID
    │
    ├──▶ vector_store.add_batch(parent collection)    ← 语义向量
    ├──▶ BM25Store.add_documents(doc_id, text)         ← 关键词索引
    │
    ├──▶ [enable_child] chunk_text(size=128, overlap=25) per parent
    │         child_id = {parent_id}_c{index}
    │         metadata: {parent_id: ...}
    │         vector_store.add_batch(child collection)
    │
    └──▶ [enable_hyde] generate_hyde_questions(chunks, batch=8)
              asyncio.gather() 并发生成
              vector_store.add_batch(hyde collection)
```

> **注意**：`DenseStore` 已更名为 `ChromaVectorStore` 并实现 `VectorStore` 抽象接口，通过 `vector_store_registry` 注册为 `"chroma"` 后端。

**公共入口：**
- `index_files(file_paths, on_progress?)` — 适合批量文件
- `index_texts(texts, on_progress?)` — 适合内存数据

#### 3.5.2 HyDE 问题生成

`rag_framework/indexing/hyde.py`：

- `generate_hyde_questions(chunks, llm, domain, batch_size=8)` — async 函数
- `asyncio.gather()` 并发生成，控制 LLM 并发压力
- 错误处理：单 chunk 失败返回空字符串，不中断整体
- 三级清洗：去 `<think>` 标签 → 提取有效问题行 → 降级解析含问号行

#### 3.5.3 Chunker

`rag_framework/indexing/chunker.py`：

```
按段落分割 → 超长段落按句分割（正则：(?<=[。！？!?])）→ 带 overlap 滑动窗口
```

优先保持段落完整性，避免语义断裂。

#### 3.5.4 初始化脚本

| 脚本 | 位置 | 用途 | 状态 |
|------|------|------|------|
| `init_vector_db_v2.py` | `domains/android/scripts/` | 生产索引（parent+child+hyde+bm25），130 行，使用 VectorIndexer | **推荐** |
| `init_vector_db.py` | `domains/android/scripts/` | V1 轻量验证（parent-only，78 行） | 已废弃 |

**V2 CLI 参数：**

```bash
uv run python -m domains.android.scripts.init_vector_db_v2 \
    [--data-dir PATH]   # 文档目录（默认 ai_app1/data）
    [--reset]           # 清空已有集合重建
    [--no-hyde]         # 跳过 HyDE 问题生成
```

---

### 3.6 混合检索管道 (HybridRetriever)

`rag_framework/retrieval/fusion.py` 实现四级检索架构：

```
查询扩写 (QueryRewriter)
    │
    ▼
三路并发召回 (ThreadPoolExecutor, max_workers=3)
    ├── 路A Dense:  child collection → parent 回溯
    ├── 路B HyDE:   hyde collection  → parent 回溯
    └── 路C BM25:   Tantivy 磁盘索引
    │
    ▼
Weighted RRF 融合
    │
    ▼
CrossEncoder 精排
    │
    ▼
Lost-in-Middle 重排
    │
    ▼
低置信度兜底 (top_ce < 0.30 → 拒答指令)
```

#### 3.6.1 父子回溯机制

Dense 与 HyDE 两路均在细粒度 collection（child/hyde）做向量检索，从命中的 `metadata.parent_id` 聚合去重，按子文档最小距离对父文档排序，拉取 `android_parent` 的完整文本作上下文。

```python
# 距离阈值过滤 + 多命中加分
max_child_distance = 1.3
parent_score = min(distance) - 0.05 * (hit_count - 1)
```

- `DENSE_QUERY_K=25`（child 候选） → `DENSE_TOP_K=10`（parent 结果）
- `HYDE_TOP_K=5`

**降级策略**：若 child/hyde collection 不存在，直接查 parent collection（`max_distance=1.2`，`n_results=5`）。

#### 3.6.2 BM25 稀疏检索 (Tantivy + jieba)

`rag_framework/retrieval/sparse.py`：

```
jieba.cut(text) → 空格连接 token 串 → Tantivy whitespace 分词器 → BM25Plus 评分
```

| 方面 | Tantivy + jieba（当前） |
|------|------------------------|
| 百万文档内存 | ~几十 MB（热点 mmap block） |
| 冷启动延迟 | 毫秒级（打开已有索引） |
| 中文分词 | jieba 精确模式 |
| BM25 计算 | Rust（~10× Python 速度） |
| 持久化 | 磁盘（`tantivy_bm25/`） |

**Schema 设计（v3.4 新增 `domain` 字段）：**

```python
doc_id   : TEXT, stored, tokenizer=raw        # 精确 ID 存取
body     : TEXT, stored, tokenizer=whitespace  # jieba 预分词后存入
raw_text : TEXT, stored, tokenizer=raw        # 原始文本，命中后返回
domain   : TEXT, stored, tokenizer=raw        # 领域标记（"android"/"msmarco"/...）
```

**统一索引构建**：`BM25Store._build_from_chroma()` 从 `knowledge_base` collection 全量拉取，自动读取每条文档 metadata 中的 `domain` 字段写入 tantivy。`add_documents()` 和 `search()` 均支持 `domain` 参数，实现增量写入和检索过滤。

#### 3.6.3 查询扩写与路由 (QueryRewriter)

**三级分流**（`AndroidDomainPlugin.rewrite_router_rules()`）：

| Level | 触发条件 | 实现类 | 耗时 |
|-------|---------|--------|------|
| **L0** 直通 | 短 query，无代词/模糊词 | — (passthrough) | ~0ms |
| **L1** 规则扩展 | 命中中文→英文术语词典 | `RuleQueryRewriter` | ~1ms |
| **L2** LLM 重写 | 含代词/模糊词/长 query（≥25字）/短追问 | `QwenQueryRewriter`（本地）| ~800ms（MPS） |

**RuleQueryRewriter**（`retrieval/query_rewriter/rule_rewriter.py`）：
- 输出：`[original (w=1.0)] + [keyword (w=0.85, routes=[bm25,dense])]`
- 仅在命中至少一个术语时输出扩写

**QwenQueryRewriter**（`retrieval/query_rewriter/qwen_rewriter.py`）— 默认 L2：
- 本地 Qwen2.5-1.5B-Instruct（`models/qwen2.5-1.5b-instruct/`），MPS/CUDA/CPU 自适应
- **懒加载**：首次调用时加载模型（~12s），后续复用；线程安全双重检查锁
- 携带最近 4 条对话历史（每条截取 80 字）
- 生成 2~3 条独立检索 query；输出 JSON 数组，失败降级按行解析
- 输出：`[original(1.0)] + [semantic(0.90, 0.80)]`
- 日志：`INFO` 级打印 model 路径、query、耗时、扩写结果

**LLMQueryRewriter**（`retrieval/query_rewriter/llm_rewriter.py`）— 通过 `LLMClient` 接口改写：
- 注入 `rewriter_llm`（而非主 `llm`），可独立配置为本地或远程
- 独立线程+事件循环，15 秒超时，失败降级为原始 query
- 输出：`[original(1.0)] + [semantic(0.90, 0.80, 0.70)]`
- **v3.5 变化**：`RAGContainer` 创建时显式传入 `rewriter_llm`，实现改写与生成模型解耦

**QwenQueryRewriter**（`retrieval/query_rewriter/qwen_rewriter.py`）— 本地独立改写器：
- 直接加载本地 transformers 模型，不经过 `LLMClient` 抽象层
- 当 `rewriter_llm_backend=local` 且本地模型路径存在时，优先于 `LLMQueryRewriter` 使用
- 与 `rewriter_llm` 配置互不影响，属于框架级独立实现

**rewriter 选择逻辑**（`container.py`）：

```python
# v3.5：双 LLM 实例创建
llm = llm_registry.create(settings.llm_backend, ...)           # 主生成 LLM
rewriter_llm = llm_registry.create(
    settings.resolved_rewriter_llm_backend, ...
)                                                              # 改写专用 LLM

# L2 改写器：优先本地 Qwen（独立 transformers），其次 LLMQueryRewriter（注入 rewriter_llm）
if rewriter_backend == "local" and Path(model_path).is_dir():
    llm_rewriter = QwenQueryRewriter(model_path, max_new_tokens)
else:
    llm_rewriter = rewriter_registry.create("llm", llm=rewriter_llm)
```

**QueryRoute 元数据：**

```python
@dataclass
class QueryRoute:
    text:   str        # 扩写后的查询文本
    type:   str        # "original" | "semantic" | "keyword" | "api"
    weight: float      # Weighted RRF 权重
    routes: list[str]  # 允许的召回路径子集
```

**Weighted RRF：**

```python
score(d) = Σ weight_i / (rank_i + RRF_K)   # RRF_K = 60
```

#### 3.6.4 并发召回延迟对比

| 阶段 | 串行 | 并发 |
|------|------|------|
| Dense | ~90ms | — |
| HyDE | ~90ms | — |
| BM25 | ~5ms | — |
| **三路合计** | **~185ms** | **~90ms** |

#### 3.6.5 Rerank 精排

**CrossEncoderReranker**（`rerank/cross_encoder.py`）：

```python
# 1. CrossEncoder.predict([query, doc]) → logits
# 2. ce_prob = sigmoid(logit)
# 3. final_score = 0.75 * ce_norm + 0.25 * rrf_norm
```

- 线程安全：`threading.Lock` 序列化 `predict()` 调用
- 降级：CrossEncoder 失败 → FallbackReranker

**FallbackReranker**（`rerank/fallback.py`）：

```python
# 中英混合分词：[一-鿿]{1,2}|[a-zA-Z0-9]+
final_score = 0.80 * (rrf_score / max_rrf) + 0.20 * token_overlap
```

#### 3.6.6 Lost-in-Middle 重排

```
输入: [rank1, rank2, rank3, rank4, rank5]
输出: [rank1, rank3, rank4, rank5, rank2]
```

最相关→首位，次相关→末位，其余居中。

#### 3.6.7 低置信度兜底

| top_ce 值 | 行为 |
|-----------|------|
| `≥ 0.30` | 正常喂参考资料给 LLM |
| `< 0.30` | 追加拒答指令，告知超出知识库范围 |
| 完全无召回 | 同拒答路径 |

`LOW_CONFIDENCE_CE_THRESHOLD = 0.30`（可通过环境变量调整）。

---

### 3.7 会话管理 (SessionManager)

`rag_framework/session/manager.py`：

```python
class SessionManager:
    async def chat_stream(self, query: str, user_id: str) -> AsyncGenerator[str, None]:
        session = self._get_or_create(user_id)
        session.add_user_message(query)

        # 查询扩写 + 混合检索（asyncio.to_thread 避免阻塞事件循环）
        meta = await asyncio.to_thread(self._retrieve, query, session.history)

        # 构建 messages（system + summary + history + 参考资料/拒答指令）
        messages = self._build_messages(session, meta)

        # 流式生成
        full_reply = ""
        async for chunk in self.llm.stream_run_agent(messages):
            yield chunk
            full_reply += chunk

        # 流结束后后台维护
        asyncio.create_task(self._maintain(session, full_reply))

    async def _maintain(self, session, reply):
        session.add_assistant_message(reply)
        if session.should_summarize():
            summary = await self.llm.chat(summarize_prompt(session))
            session.update_summary(summary)
        session.trim_history()  # 保留最近 MAX_HISTORY=4 条
```

**SessionData 结构：**

```python
class SessionData(TypedDict):
    history:      list        # 最近对话 (role/content)
    summary:      str         # 历史压缩摘要
    trimmed:      list        # 被裁剪的旧消息（不丢弃）
    token_budget: int         # 剩余 token 预算 (4096)
```

**Token 估算（加权字符数）：**

```python
# 中文（含全角标点）~1.5 token/字；英文/代码 ~0.5 token/字符
cn_chars    = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
other_chars = len(text) - cn_chars
tokens      = int(cn_chars * 1.5 + other_chars * 0.5)
```

---

### 3.8 LLM 客户端

框架支持两种 LLM 客户端，通过 `llm_backend` / `rewriter_llm_backend` 独立配置切换：

#### OpenAILLMClient（`rag_framework/llm/openai_client.py`）

| 方法 | 用途 | 流式 |
|------|------|------|
| `chat(messages)` | 普通对话 / summarize | 否 |
| `stream_run_agent(messages)` | **生产主入口**：流式工具增强多轮对话 | 是 |

**MiniMax 兼容三参数：**

```python
stream_kwargs["max_tokens"] = LLM_MAX_TOKENS          # 512
stream_kwargs["extra_body"] = {
    "max_completion_tokens": LLM_MAX_TOKENS,
    "tokens_to_generate":   LLM_MAX_TOKENS,
}
```

**Tool Calling 流式循环：**

```python
async for chunk in response:
    delta = chunk.choices[0].delta
    if delta.content:
        yield delta.content                 # 实时 yield 给客户端
    if delta.tool_calls:
        # 增量拼接 tool_call → 执行工具 → 追加结果 → 下一轮
```

最多 `MAX_STEPS=10` 轮。

#### LocalLLMClient（`rag_framework/llm/local_client.py`）

本地 transformers 推理客户端，支持 `local` backend：

- **模型**：本地 `models/qwen2.5-1.5b-instruct/`（或其他 HF 格式模型）
- **设备自适应**：CUDA → MPS → CPU 自动选择
- **线程安全**：双重检查锁懒加载 + asyncio Semaphore 限流
- **预热**：`warmup()` 在 lifespan 中异步加载，避免首请求阻塞
- **权重绑定**：加载时调用 `model.tie_weights()` 确保 `lm_head` 与 `embed_tokens` 共享参数，防止乱码

```python
# 创建示例
llm = llm_registry.create("local",
    model_path="models/qwen2.5-1.5b-instruct",
    max_tokens=512,
    max_concurrent=3,
)
```

**v3.5 双 LLM 实例**：`RAGContainer.from_settings()` 现在同时创建 `llm` 与 `rewriter_llm` 两个实例：
- `llm` — 用于最终对话生成、`summarize`、工具调用循环（默认远程 MiniMax，追求质量）
- `rewriter_llm` — 用于 `LLMQueryRewriter` 查询改写（默认本地 Qwen，追求低成本+低延迟）
- 两者通过完全独立的配置组管理，共享 `llm_max_concurrent` 并发限制
- `warmup_targets()` 同时包含两个实例，lifespan 启动时并行预热

---

### 3.9 应用层 (ai_app1)

#### 3.9.1 main.py — 生命周期管理

```python
# 配置：要同时加载的领域插件
_DOMAIN_CLASSES = []
try:
    from msmarco_domain.plugin import MSMarcoDomainPlugin
    _DOMAIN_CLASSES.append(MSMarcoDomainPlugin)
except Exception:
    pass
try:
    from android_domain.plugin import AndroidDomainPlugin
    _DOMAIN_CLASSES.append(AndroidDomainPlugin)
except Exception:
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    from rag_framework.core.registry import register_domain, get_domain, list_domains
    from rag_framework.core.factories import retriever_registry, session_store_registry
    from rag_framework.retrieval.sparse import BM25Store

    # 1. 显式注册所有领域插件（去 import-time 副作用）
    for cls in _DOMAIN_CLASSES:
        register_domain(cls)
    registered = list_domains()

    # 2. 确定默认激活领域
    default_domain = os.getenv("RAG_ACTIVE_DOMAIN", registered[0])
    if default_domain not in registered:
        default_domain = registered[0]

    # 3. 创建基础容器（共享重型组件）
    base_settings = RAGSettings()
    base_settings = base_settings.model_copy(update={"active_domain": default_domain})
    base_container = RAGContainer.from_settings(base_settings)

    # 4. 创建共享 BM25Store（统一索引，所有领域共用）
    shared_bm25 = BM25Store(
        index_dir=base_settings.bm25_index_dir,
        chroma_path=base_settings.chroma_db_path,
        collection_name="knowledge_base",
    )

    # 5. 为每个领域创建派生容器
    #    共享 embedder / vector_store / llm / reranker / bm25
    #    独立 session_store / retriever（带各自的 domain_filter）
    from dataclasses import replace
    containers: dict[str, RAGContainer] = {}
    for name in registered:
        domain_settings = base_settings.model_copy(update={"active_domain": name})
        domain = get_domain(name)
        retriever = retriever_registry.create(
            domain_settings.retriever_backend,
            settings=domain_settings,
            embedder=base_container.embedder,
            vector_store=base_container.vector_store,
            reranker=base_container.reranker,
            domain=domain,
            sparse_store=shared_bm25,
            domain_filter=domain.name if domain else "",
        )
        session_store = session_store_registry.create(
            domain_settings.session_store_backend,
            default_budget=domain_settings.default_token_budget,
        )
        containers[name] = replace(
            base_container,
            settings=domain_settings,
            domain=domain,
            retriever=retriever,
            session_store=session_store,
        )

    app.state.containers = containers
    app.state.container = containers.get(default_domain)  # 兼容旧代码

    # 6. 并发预热所有 Warmupable 组件（按对象 id 去重）
    seen_ids: set[int] = set()
    warmup_tasks = []
    for c in containers.values():
        for comp in c.warmup_targets():
            cid = id(comp)
            if cid not in seen_ids:
                seen_ids.add(cid)
                warmup_tasks.append(comp.warmup())
    if warmup_tasks:
        await asyncio.gather(*warmup_tasks)
    print(f"[startup] 已加载领域: {list(containers.keys())}")
    yield

    # 7. shutdown：关闭所有 Closable 组件（按对象 id 去重）
    seen_ids = set()
    for c in containers.values():
        for comp in (c.embedder, c.reranker, c.vector_store, c.llm):
            cid = id(comp)
            if cid not in seen_ids and isinstance(comp, Closable):
                seen_ids.add(cid)
                try:
                    await comp.shutdown()
                except Exception as e:
                    print(f"[shutdown] 关闭组件出错: {e}")
```

#### 3.9.2 api/chat.py — HTTP 接入

```python
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    domain: str = ""  # "msmarco", "android", "all", 空字符串自动路由

def _resolve_single_container(
    containers: dict[str, RAGContainer],
    default_container: RAGContainer,
    domain: str,
    query: str = "",
) -> RAGContainer:
    """单领域容器解析：显式指定 > 自动路由（中文→android）> 默认"""
    if domain and domain in containers:
        return containers[domain]
    if query and any("\u4e00" <= ch <= "\u9fff" for ch in query):
        android = containers.get("android")
        if android is not None:
            return android
    return default_container

async def _retrieve_from_domain(
    container: RAGContainer, query: str, user_id: str
) -> list[RetrievedDoc]:
    session = container.session_store.get(user_id)
    routes = container.build_routes(query, session.history)
    result = await container.retriever.retrieve(routes)
    return result.docs

async def _multi_domain_chat_stream(
    containers: dict[str, RAGContainer],
    default_container: RAGContainer,
    query: str,
    user_id: str,
):
    """多领域融合：并行检索所有领域，合并去重后由默认领域 LLM 生成回答。"""
    tasks = [_retrieve_from_domain(c, query, user_id) for c in containers.values()]
    all_docs_per_domain = await asyncio.gather(*tasks)

    seen: set[str] = set()
    merged: list[RetrievedDoc] = []
    for docs in all_docs_per_domain:
        for d in docs:
            if d.id not in seen:
                seen.add(d.id)
                merged.append(d)

    if not merged:
        yield "未检索到任何相关文档，请尝试更换关键词。"
        return

    merged.sort(key=lambda x: x.score, reverse=True)
    top_docs = merged[:6]

    session = default_container.session_store.get(user_id)
    messages = [
        {"role": "system", "content": default_container.domain.system_prompt},
        *session.history,
    ]
    context = "\n\n".join(f"[{i+1}] {d.text}" for i, d in enumerate(top_docs))
    messages.append({"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"})

    full_reply = ""
    async for chunk in default_container.llm.chat_stream(messages, use_tools=False):
        full_reply += chunk
        yield chunk

    session.history.append({"role": "user", "content": query})
    session.history.append({"role": "assistant", "content": full_reply})
    default_container.session_store.save(session)

@router.post("/chat")
async def chat(req: ChatRequest, request: Request):
    containers = getattr(request.app.state, "containers", {}) or {}
    default_container = getattr(request.app.state, "container", None)
    if not containers and default_container is not None:
        containers = {"default": default_container}  # 兼容旧测试
    if not containers or default_container is None:
        raise RuntimeError("RAGContainer 未初始化，请检查 lifespan")

    is_multi_domain = req.domain.strip().lower() == "all"
    if is_multi_domain:
        stream = _multi_domain_chat_stream(containers, default_container, req.message, req.user_id)
    else:
        container = _resolve_single_container(containers, default_container, req.domain, req.message)
        stream = container.chat_stream(req.message, req.user_id)

    async def content_generator():
        async for chunk in stream:
            if chunk:
                yield chunk
    return StreamingResponse(content_generator(), media_type="text/event-stream")
```

#### 3.9.3 tests/test_api.py — API 测试

- pytest 9.0.3 + pytest-asyncio 1.3.0
- 7 个测试函数：健康检查、SSE 流式、输入校验、用户隔离、CORS
- **关键技巧**：`TestClient` 会触发 lifespan，导致真实模型加载并覆盖测试注入的 mock 容器。因此在 `with TestClient(...)` 上下文**内部**再注入 mock，确保 lifespan 执行完成后覆盖：

```python
@pytest.fixture
def client():
    from ai_app1.main import app
    with TestClient(app, raise_server_exceptions=True) as c:
        # lifespan 已执行完毕，现在覆盖为纯内存 mock
        app.state.container = build_test_container()
        app.state.containers = {"default": app.state.container}
        yield c
```

- 无外部依赖，可在 CI 中运行

```bash
uv run pytest ai_app1/tests/test_api.py -v
```

---

### 3.10 评测与可观测性体系

> **v3.3 新增**：从“能回答”迈向“稳定、可观测、可进化、可维护、可优化”。本节涵盖 Query 分类统计、Rewrite/Rerank 效果量化、检索全链路 Trace、延迟拆解、失败分析闭环、综合评测调度器六大模块。

#### 3.10.1 Query 自动分类与统计

`rag_framework/eval/query_classifier.py` 基于规则对评测 query 自动分类，解决“90% recall 掩盖了 10% 的致命短板”问题。

**分类规则（优先级从高到低）：**

| 类型 | 触发条件 | 典型场景 |
|------|---------|---------|
| `anaphora` | 含指代词（这个、上面、那、它）且长度 <20 | 追问“那它怎么释放？” |
| `typo` | 含常见拼写错误（Hanlder, Acitvity 等） | 用户输入错误 |
| `long_query` | 长度 >100 字符 | 复杂场景描述 |
| `adversarial` | 含错误前提反问 | “是不是就不用写 onDestroy 了？” |
| `code_switching` | 中英混合（中文字符>3 且英文单词>1） | “Fragment 生命周期” |
| `multi_hop` | 多个技术概念连接，或 expected_chunk 含 "/" | “Handler + Thread + ANR” |
| `vague` | 极短（<12 字）或无明确技术关键词 | “怎么优化？” |
| `keyword` | 其余（含明确技术关键词的标准问题） | “RecyclerView 缓存机制” |

**聚合输出：**
- [`aggregate_by_type()`](rag_framework/rag_framework/eval/query_classifier.py:129) 按类型分组计算 recall@5、hit@1、MRR、平均延迟
- [`format_type_stats()`](rag_framework/rag_framework/eval/query_classifier.py:159) 输出对齐表格，例如：

```
📊 Query 分类统计
──────────────────────────────────────────────────────────────────────
类型                  数量   Recall@5    Hit@1      MRR   平均延迟
──────────────────────────────────────────────────────────────────────
keyword                 12     91.7%    75.0%   0.812        120ms
vague                    5     40.0%    20.0%   0.320        350ms
multi_hop                3     20.0%     0.0%   0.150        280ms
──────────────────────────────────────────────────────────────────────
```

在 [`run_ranking_eval()`](rag_framework/rag_framework/eval/ranking.py:136) 中已自动集成：每次评测先打印分类统计，再输出总体指标。

#### 3.10.2 Rewrite 效果评测

`rag_framework/eval/rewrite_eval.py` 量化 Query Rewrite 的“ before/after ”收益。

**核心指标：**
- `ΔRecall@5`、`ΔHit@1`、`ΔMRR`：rewrite 后与原始 query 的差值
- **提升 / 下降 / 持平** 数量统计

**流程：**
1. 对原始 query 直接检索（不走 rewrite），记为 Before
2. 走完整 `SessionManager._build_routes()`（含 L0/L1/L2 分流），记为 After
3. 用 [`RewriteComparison`](rag_framework/rag_framework/eval/rewrite_eval.py:28) 记录 delta

```python
@dataclass
class RewriteComparison:
    query: str
    rewritten: str
    before_recall: float;  after_recall: float
    before_hit1: float;    after_hit1: float
    before_mrr: float;     after_mrr: float
    delta_recall: float;   delta_hit1: float;  delta_mrr: float
    is_improved: bool;     is_degraded: bool
```

**CLI 调用：**
```bash
python -m rag_framework.eval.comprehensive_eval rewrite
# 或单独运行
python -m rag_framework.eval.rewrite_eval
```

#### 3.10.3 Rerank 效果评测

`rag_framework/eval/rerank_eval.py` 验证 CrossEncoder 是否真的把正确 chunk 推到了 top1。

**核心指标：**

| 指标 | 含义 |
|------|------|
| `Win` | 正确 chunk 从非 top1 被推到 top1 |
| `Loss` | 正确 chunk 从 top1 被挤出 top1 |
| `Hold` | 原本 top1，rerank 后仍是 top1 |
| `Miss` | 原本未命中，rerank 后也未命中 |
| `avg_rank_delta` | rerank 前后正确 chunk 平均排名变化（负数=上升） |

**流程：**
1. 获取 RRF 融合后的候选列表（rerank 前）
2. 执行 `CrossEncoderReranker.rerank()` 获取 rerank 后列表
3. 对比 ground truth 在两个列表中的排名位置

```python
@dataclass
class RerankComparison:
    before_rank: int      # rerank 前第一个 gt 的位置（1-based）
    after_rank: int       # rerank 后位置
    is_win: bool
    is_loss: bool
    is_hold: bool
    ce_scores: list[tuple[str, float]]
```

#### 3.10.4 Retrieval Trace — 检索全链路追踪

`rag_framework/eval/retrieval_trace.py` 让检索从“黑盒”变成“白盒”。

**数据结构：**
- [`BranchTrace`](rag_framework/rag_framework/eval/retrieval_trace.py:20)：单路召回分支（kind / query_text / weight / status / latency_ms / result_count / top_ids / error）
- [`RerankTrace`](rag_framework/rag_framework/eval/retrieval_trace.py:33)：rerank 阶段（status / latency_ms / input_count / output_count / top_ce_score / error）
- [`RetrievalTrace`](rag_framework/rag_framework/eval/retrieval_trace.py:44)：完整 trace（含 rewrite、branches、rrf、rerank、LiM、final latency）

**集成点：** [`HybridRetriever.retrieve()`](rag_framework/rag_framework/retrieval/fusion.py:93) 中已内嵌 Trace 记录：
```python
timer = PhaseTimer()
trace = RetrievalTrace()
# 每个阶段结束后填充 trace
record_trace(trace)
logger.info(trace.print_trace())
```

**人类可读输出示例：**
```
╔══════════════════════════════════════════════════════════════════════╗
║                         RETRIEVAL TRACE                              ║
╠══════════════════════════════════════════════════════════════════════╣
║ 原始 Query : Android 中 Handler 内存泄漏怎么解决？                     ║
║ 改写 Query : Android Handler memory leak 原因与解决方案                ║
║ 改写类型   : llm        耗时:    820ms                               ║
╠══════════════════════════════════════════════════════════════════════╣
║ 多路召回                                                             ║
║   [✅] dense    q=Android Handler memory lea  n=10    92ms           ║
║   [✅] hyde     q=为什么 Android Handler 会  n=5     88ms           ║
║   [✅] bm25     q=Handler 内存泄漏           n=8      5ms           ║
╠══════════════════════════════════════════════════════════════════════╣
║ RRF 融合     : 输入 3 路 → 输出 12 条   1ms                          ║
║ Rerank [✅]   : 输入 12 → 输出 3   top_ce=0.421   45ms               ║
╠══════════════════════════════════════════════════════════════════════╣
║ 最终结果     : 3 个片段  top_ce=0.421  总耗时=  1051ms               ║
║ Top IDs      : android_parent_001, android_parent_042...             ║
╚══════════════════════════════════════════════════════════════════════╝
```

**全局存储：**
- `record_trace(trace)` → `_active_traces`（环形缓冲，上限 1000 条）
- `get_recent_traces(n)` / `get_traces_by_query(substring)` 供调试和失败分析查询

#### 3.10.5 Latency Breakdown — 延迟精细化拆解

`rag_framework/eval/latency_breakdown.py` 定位 TTFT（Time To First Token）瓶颈。

**PhaseLatency 字段：**
`rewrite_ms` → `classify_ms` → `dense_ms` → `hyde_ms` → `bm25_ms` → `fetch_parents_ms` → `rrf_ms` → `rerank_ms` → `lim_ms` → `total_ms`

**PhaseTimer 用法（已在 `HybridRetriever.retrieve()` 中集成）：**
```python
timer = PhaseTimer()
with timer.phase("rewrite"):
    routes = self._build_routes(...)
timer.record("dense", dense_latency_ms)
latency = timer.finish()   # 返回 PhaseLatency
```

**聚合报告：** [`aggregate_phase_latencies()`](rag_framework/rag_framework/eval/latency_breakdown.py:89) 自动生成：
- 各阶段 mean / P50 / P95 / P99 / 占比
- 自动识别 **bottleneck**（耗时最长的阶段）

```
📊 Latency Breakdown
─────────────────────────────────────────────────────────────────
阶段                     平均(ms)      P50      P95      P99      占比
─────────────────────────────────────────────────────────────────
rewrite                   820.5    800.0   1200.0   1500.0   78.2%
dense                      92.0     90.0    120.0    150.0    8.8%
hyde                       88.0     85.0    110.0    130.0    8.4%
bm25                        5.0      4.0      8.0     10.0    0.5%
rrf                         1.0      1.0      2.0      3.0    0.1%
rerank                     45.0     42.0     60.0     75.0    4.3%
─────────────────────────────────────────────────────────────────
total                    1051.5   1022.0   1500.0   1800.0

🔴 瓶颈阶段: rewrite
```

#### 3.10.6 Failure Analysis System — 失败分析与数据闭环

`rag_framework/eval/failure_analysis.py` 自动收集“问题 query”，为迭代优化提供数据基础。

**FailureCase 字段：**
`query` / `category` / `reason` / `timestamp` / `session_id` / `trace` / `metadata`

**收集维度（由 `FailureCollector` 提供）：**

| 方法 | 触发场景 | 调用位置 |
|------|---------|---------|
| `collect_miss()` | 检索未命中 ground truth | 评测框架 / `SessionManager` |
| `collect_low_ce()` | top_ce < 0.30（置信度低） | `SessionManager.build_messages()` |
| `collect_rerank_loss()` | rerank 把正确 chunk 挤出 top1 | `rerank_eval` |
| `collect_rewrite_degrade()` | rewrite 后 recall 下降 | `rewrite_eval` |
| `collect_explicit_bad()` | 用户明确表达不满意 | 对话层（未来接入） |
| `collect_followup()` | 检测到用户追问 | 对话层（未来接入） |

**存储后端 `FailureStore`：**
- 格式：JSON Lines（`reports/failure_cases.jsonl`）
- 策略：内存 buffer（10 条）+ 增量刷盘，避免频繁 IO
- 查询：`get_by_category()` / `summary()` / `print_summary()`

**集成点：** [`SessionManager.build_messages()`](rag_framework/rag_framework/session/manager.py:132) 中已自动收集：
- 检索结果为空 → `collect_miss()`
- `top_ce < LOW_CONFIDENCE_CE_THRESHOLD` → `collect_low_ce()`

#### 3.10.7 Comprehensive Eval — 综合评测调度器

`rag_framework/eval/comprehensive_eval.py` 是一站式调度器，串联所有评测维度。

**支持命令：**

| 命令 | 说明 |
|------|------|
| `ranking` | 检索排序评测（含 query 分类统计） |
| `rewrite` | Rewrite before/after 对比 |
| `rerank` | Rerank win/loss/hold 验证 |
| `ablation` | 消融实验（多配置对比） |
| `hard` | 困难样本专项评测 |
| `qa` | 端到端 QA + LLM-as-Judge |
| `all` | 默认全量运行（不含 qa） |

**输出：**
- Markdown 报告：`reports/comprehensive_YYYYMMDD_HHMMSS.md`
- JSON 报告：`reports/comprehensive_YYYYMMDD_HHMMSS.json`
- 失败样本汇总：自动打印 `FailureStore` 统计

**CLI 调用：**
```bash
# 全量评测
python -m rag_framework.eval.comprehensive_eval all

# 单项评测
python -m rag_framework.eval.comprehensive_eval ranking
python -m rag_framework.eval.comprehensive_eval rewrite --dataset domains/android/android_domain/eval/benchmark.json
```

---

## 4. 完整数据流

```mermaid
flowchart TD
    A[用户提问 POST /chat<br/>含可选 domain 字段] --> B[chat.py 路由层]
    B --> C{domain == "all"?}
    C -->|是| D1[并行检索所有领域]
    C -->|否| D2[_resolve_single_container<br/>显式 > 中文→android > 默认]
    D1 --> E1[_multi_domain_chat_stream]
    D2 --> E2[SessionManager.chat_stream]
    E1 --> F[获取/创建 SessionData]
    E2 --> F
    F --> G[用户消息入 history]
    G --> H[DomainPlugin.rewrite_router_rules<br/>L0/L1/L2 分流]
    H --> I{扩写级别}
    I -->|L0| J1[原始 query × 1]
    I -->|L1| J2[RuleQueryRewriter<br/>词典扩展 2条]
    I -->|L2| J3[QwenQueryRewriter<br/>本地 Qwen 生成 2~3条]
    J1 & J2 & J3 --> K[asyncio.to_thread → HybridRetriever]
    K --> L{ThreadPoolExecutor 3路并发}
    L --> L1[Dense: child→parent 回溯<br/>where={"domain": {"$eq": "xxx"}}]
    L --> L2[HyDE: hyde→parent 回溯<br/>同上 where 过滤]
    L --> L3[BM25: Tantivy 磁盘检索<br/>boolean query +domain:xxx]
    L1 & L2 & L3 --> M[Weighted RRF 融合]
    M --> N[CrossEncoderReranker 精排<br/>ce × 0.75 + rrf × 0.25]
    N --> O[Lost-in-Middle 重排]
    O --> P{top_ce ≥ 0.30?}
    P -->|是| Q[build_messages: 参考资料]
    P -->|否| R[build_messages: 拒答指令]
    Q & R --> S[stream_run_agent 流式调用 LLM]
    S --> T[StreamingResponse 逐 token 返回]
    T --> U[流结束 → create_task 后台维护]
    U --> V[add_assistant_message]
    V --> W{should_summarize?}
    W -->|是| X[chat summarize → update_summary]
    W -->|否| Y[trim_history 保留最近4条]
    X --> Y
```

---

## 5. 关键配置参考

| 配置项 | 默认值 | 所在位置 |
|--------|--------|----------|
| `llm_backend` | `minimax` | `core/config.py` |
| `llm_model` | `MiniMax-M2.7` | `core/config.py` |
| `LLM_MAX_TOKENS` | `512` | `llm/openai_client.py` |
| `MAX_HISTORY` | `4` | `session/manager.py` |
| `DEFAULT_TOKEN_BUDGET` | `4096` | `session/manager.py` |
| `RRF_K` | `60` | `retrieval/fusion.py` |
| `MAX_CHILD_DISTANCE` | `1.3` | `retrieval/fusion.py` |
| `DENSE_QUERY_K` | `25` | `retrieval/fusion.py` |
| `DENSE_TOP_K` | `10` | `retrieval/fusion.py` |
| `HYDE_TOP_K` | `5` | `retrieval/fusion.py` |
| `BM25_TOP_K` | `10` | `retrieval/fusion.py` |
| `RERANK_TOP_K` | `3` | `retrieval/fusion.py` |
| `LOW_CONFIDENCE_CE_THRESHOLD` | `0.30` | `retrieval/fusion.py` |
| `IndexConfig.chunk_size` | `512` | `indexing/indexer.py` |
| `IndexConfig.child_chunk_size` | `128` | `indexing/indexer.py` |
| `IndexConfig.enable_child` | `True` | `indexing/indexer.py` |
| `IndexConfig.enable_hyde` | `True` | `indexing/indexer.py` |

---

## 6. 运行流程

### 6.1 首次部署

```bash
# 1. 安装依赖（含 rag_framework / android-domain / msmarco-domain 本地包）
uv sync

# 2. 配置环境变量
cp ai_app1/.env.example ai_app1/.env
# 编辑 .env: OPENAI_API_KEY=your_minimax_key

# 3. 构建离线索引（所有领域写入统一 knowledge_base collection）
uv run python -m domains.android.scripts.init_vector_db_v2
uv run python -m domains.msmarco.scripts.download_and_index

# 4. 启动服务（自动预热所有模型）
uv run python -m uvicorn ai_app1.main:app --host 0.0.0.0 --port 8000
```

### 6.2 开发调试

```bash
# 本地源码安装（editable，修改立即生效）
uv pip install -e rag_framework/

# 运行 API 测试
uv run pytest ai_app1/tests/test_api.py -v

# 查看 rewrite 缓存命中率
curl http://localhost:8000/debug/rewrite_cache
```

### 6.3 API 调用

```bash
# 自动路由（中文→android）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Android 中 Handler 内存泄漏怎么解决？"}'

# 强制指定领域
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How many people live in Berlin?", "domain": "msmarco"}'

# 跨领域融合
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "compare Android and iOS", "domain": "all"}'
```

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 包结构 | rag_framework 独立可安装包 | 与 ai_app2/ai_app3 共用框架；隔离领域逻辑与通用逻辑 |
| 领域知识 | DomainPlugin 插件系统 | 集合命名/术语词典/路由规则/HyDE Prompt 等领域特异性内容统一收敛，框架零耦合 |
| DI 模式 | RAGContainer.from_settings() + FastAPI Depends | 一次组装，全请求复用；可测试（mock 注入） |
| **多领域容器** | `dataclasses.replace()` 派生 + 对象 id 去重预热 | 共享重型组件（embedder/llm/reranker），独立 session/retriever；避免重复加载模型 |
| **统一 Collection** | 所有领域写入 `knowledge_base`，`domain` metadata 隔离 | 无需为每个领域维护独立 collection 和 BM25 索引；ChromaDB `where` + Tantivy boolean query 实现运行时过滤；新增领域零索引脚本改动 |
| Embedding | BGE-M3 (本地 STEmbedder) | 中文语义效果优、L2 normalize、显式编码便于缓存和换模型 |
| 向量库 | ChromaDB | 本地持久化、零配置、Python 原生 |
| LLM（生成） | MiniMax-M2.7（默认） | 中文能力强、OpenAI 格式兼容；承担最终对话与 summarize |
| LLM（改写） | Qwen2.5-1.5B-Instruct（默认本地） | 改写任务轻量，本地运行零 API 成本、低延迟；与生成模型解耦避免远程配额浪费 |
| 检索架构 | 三路并发混合检索 | Dense 精度 + HyDE 覆盖 + BM25 关键词，三路互补；ThreadPoolExecutor 并发 ~90ms |
| 父子回溯 | child 检索 → parent 上下文 | 细粒度匹配提升精度，大粒度 parent 保证上下文连贯性 |
| BM25 引擎 | Tantivy (Rust) + jieba | mmap 磁盘索引内存占用小；Rust BM25 ~10× Python；jieba 分词精度高 |
| 查询扩写 | 三级路由 L0/L1/L2 | 80% 简单 query 不需 LLM；规则路由 1ms 分流，平均耗时大幅降低 |
| L2 改写器 | 本地 Qwen2.5-1.5B-Instruct | 避免 L2 改写消耗远程 API 配额；~800ms（MPS）vs ~1500ms（远端）；rewriter_backend=auto 自动选择，本地模型不存在时降级 LLMQueryRewriter |
| 改写/生成模型分离 | `rewriter_llm_*` 独立配置组 + `RAGContainer` 双实例 | 改写走轻量本地模型，生成走远程大模型；独立配置、独立预热、独立回退；不影响原有单 LLM 使用方式 |
| Reranker | CrossEncoder (bge-reranker-base) | 语义相关性打分优于规则线性；sigmoid 归一化后与 RRF 同区间；FallbackReranker 作降级 |
| 低置信度兜底 | ce_score 阈值 0.30 | 复用 reranker 输出无额外开销；明确拒答比基于不相关片段硬答更有价值 |
| 索引编排 | VectorIndexer 统一管道 | 消除 init 脚本与框架的重复逻辑（chunking/HyDE/batch 写入），300 行 → 130 行 |
| 会话存储 | 内存字典 | 进程级简单实现，重启后丢失（可扩展至 Redis） |
| 流式响应 | AsyncGenerator + StreamingResponse | 用户首 token 即可见，TTFT 最优；后台 create_task 维护 session |
| max_tokens 兼容 | max_tokens + extra_body 三参数 | MiniMax 对 OpenAI SDK max_tokens 解析不完全兼容，三参数兜底 |
| 包安装模式 | editable install 推荐 | 非 editable 安装时源码修改不反映到运行时，开发期需 `pip install -e .` |

---

## 8. 已知风险与缓解措施

### 8.1 检索去重（A）

**风险**：三路召回命中同一 Parent 导致重复内容。  
**缓解**：`seen_ids: set[str]` 候选去重；`rerank_chunks()` 后运行时断言。

### 8.2 CrossEncoder 线程安全（B）

**风险**：HF fast tokenizer (Rust RefCell) 不允许并发调用。  
**缓解**：`BgeRerankerService` 内 `threading.Lock` 序列化 `predict()` 调用。

### 8.3 冷启动延迟（C）

**风险**：BGE-M3 ~3-5s、CrossEncoder ~2-3s、QwenQueryRewriter（本地）首次调用 ~12s。  
**缓解**：BGE-M3 / CrossEncoder 在 `preload_models()` 启动时预热；QwenQueryRewriter **懒加载**（首条触发 L2 改写的请求承担加载耗时，后续复用）。

### 8.4 rag_framework 版本漂移（D）

**风险**：非 editable 安装时源码修改不自动反映到 venv，导致源码与运行时不一致。  
**缓解**：开发期使用 `uv pip install -e rag_framework/`（editable mode）；生产部署后更新源码需重新安装。

### 8.5 Session 竞争条件（E）

**风险**：流式传输期间用户发第二条请求，`build_messages()` 可能读到旧 history。  
**缓解**：当前单用户场景影响低；长期方案：asyncio.Lock 保护 session 读写。

### 8.6 知识库外问题幻觉（F）✅ 已修复

**风险**：用户问范围外的问题（iOS 开发等），LLM 在不相关片段上硬答。  
**修复**：`top_ce < 0.30` 时追加拒答指令，不喂参考片段。

### 8.7 LLM Rewrite 成本超过检索（G）✅ 已修复

**风险**：100% 走 LLM rewrite 时 ~1.5s 占 TTFT 60%+。  
**修复**：三级分流 L0/L1/L2，平均耗时 ~300ms，简单 query 0~1ms。

### 8.8 MiniMax max_tokens 兼容（H）✅ 已修复

**风险**：MiniMax 不完全兼容 OpenAI SDK 的 `max_tokens` 字段，实测超长输出。  
**修复**：三参数同发 `max_tokens + max_completion_tokens + tokens_to_generate`。

### 8.9 Summarize 上下文断裂（I）

**风险**：长跨度对话中关键报错被压缩为模糊摘要。  
**缓解**：保留最近 4 条原始消息；未来可考虑结构化摘要（保留代码/堆栈片段）。

---

## 9. 版本变更记录

| 版本 | 日期 | 变更内容 |
|------|------|----------|
| **v3.5** | **2026-05-16** | **改写与生成 LLM 解耦**：`RAGSettings` 新增 `rewriter_llm_backend` / `rewriter_llm_model` / `rewriter_llm_base_url` / `rewriter_llm_api_key` / `rewriter_llm_max_tokens` / `rewriter_llm_local_model_path` 配置组，留空自动回退到主 `llm_*`；`RAGContainer` 新增 `rewriter_llm` 字段，`from_settings()` 同时创建主 `llm` 与 `rewriter_llm` 两个实例；`LLMQueryRewriter` 注入 `rewriter_llm` 而非主 `llm`；`warmup_targets()` 同时预热两个实例；`ai_app3/core/llm_provider.py` 统一接管 LangChain `ChatOpenAI` 创建，消除 `query_engine.py` / `evaluator.py` / `context_compressor.py` / `graph/builder.py` 四处硬编码 `ChatOpenAI(model="MiniMax-M2.7", ...)`；`ai_app1/.env.example` 补充 rewriter_llm 配置模板；`verify_refactor.py` 增加 rewriter_llm 配置类型检查 |
| **v3.4** | **2026-05-16** | **统一 Collection 多领域架构**：所有领域数据写入 `knowledge_base` collection，`domain` metadata 隔离；`ChromaVectorStore.query()` 支持 `where` 过滤；`BM25Store` schema 新增 `domain` 字段，search/add 支持 domain 过滤；`HybridRetriever` 接受 `domain_filter` 并传递至 dense/BM25；工厂注册表透传 `domain_filter`；`AndroidDomainPlugin` / `MSMarcoDomainPlugin` 统一返回 `knowledge_base` 前缀集合名；`VectorIndexer` / `init_vector_db_v2` / `download_and_index` 自动写入 `domain` metadata；`ai_app1/main.py` 多领域并发加载：共享 embedder/vector_store/llm/reranker/bm25，独立 session_store/retriever（`replace()` 派生 + 对象 id 去重预热）；`ai_app1/api/chat.py` 支持 `domain="all"` 跨领域融合与自动路由（中文→android）；`test_api.py` fixture 在 lifespan 后注入 mock 容器 |
| v3.3 | 2026-05-15 | **评测与可观测性体系**：`query_classifier.py` 自动分类 + 按类型统计 recall；`rewrite_eval.py` before/after 量化对比；`rerank_eval.py` CrossEncoder win/loss/hold 验证；`retrieval_trace.py` 检索全链路 Trace（已嵌入 `HybridRetriever`）；`latency_breakdown.py` PhaseTimer + 瓶颈自动识别；`failure_analysis.py` 失败样本收集 + JSON Lines 存储（已嵌入 `SessionManager`）；`comprehensive_eval.py` 一站式评测调度器；`eval/__init__.py` 统一导出 |
| v3.1 | 2026-05-13 | **本地 Qwen 改写器**：新增 `QwenQueryRewriter`（Qwen2.5-1.5B-Instruct，懒加载，线程安全）；`container.py` 按 `rewriter_backend` 自动选择本地/远程改写器；`SessionManager._build_routes()` 加 Rewrite level 日志；`LLMQueryRewriter` / `RuleQueryRewriter` 加 model + 耗时 INFO 日志 |
| v3.0 | 2026-05-13 | **三层架构重构**：rag_framework 独立包 + AndroidDomainPlugin + 薄应用层；删除 ai_app1 代理层；VectorIndexer 统一索引管道；RuleQueryRewriter / LLMQueryRewriter 提取为框架组件；FallbackReranker 独立组件；DenseStore 新增 get_or_create_collection；pytest 测试套件 |
| v2.7 | 2026-05-12 | Rewrite Router 三级分流（L0/L1/L2）；低置信度兜底（top_ce<0.30）；Ollama 集成；LLM max_tokens 兼容修复；RERANK_TOP_K 5→3；`/debug/rewrite_cache` 监控 |
| v2.6 | 2026-05-12 | Query Rewrite + Retrieval Orchestration：RewriteQuery(text,type,weight,routes) + Weighted RRF |
| v2.5 | 2026-05-11 | CrossEncoder 语义重排；API 全面流式化；session 后台异步维护；启动预热 |
| v2.4 | 2026-05-10 | 三路并发召回（ThreadPoolExecutor）；Dense 聚合优化；RRF + term_overlap 方案 A |
| v2.2 | 2026-05-08 | 修复 AsyncOpenAI 客户端；删除冗余实例化；修复 summarize 输入格式；修复硬编码路径 |
| v2.0 | 2026-05-06 | Parent-Child 架构 + HyDE + BM25 多路混合检索 |
| v1.0 | 2026-05-04 | 初始版本：单路向量检索 + 基础会话管理 |
