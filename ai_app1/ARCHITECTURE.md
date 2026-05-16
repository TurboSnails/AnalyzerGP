# ai_app1 架构文档（详细版）

> 面向 Android/Kotlin 开发者的 Python RAG 应用架构解读

---

## 1. 项目定位：ai_app1 是什么？

`ai_app1` 是一个**基于 `rag_framework` 构建的薄应用层**，核心功能一句话概括：

> **用户发问 → 检索相关文档 → 把文档塞给 AI → AI 流式回答**

它不负责具体的检索算法、模型加载、向量存储等「重型逻辑」，这些全部下沉到 `rag_framework` 中。`ai_app1` 只干三件事：

1. **组装**：启动时把各个零件（Embedder、LLM、Retriever 等）按领域配置拼起来
2. **路由**：收到 HTTP 请求时，判断该用哪个领域的知识库回答
3. **转发**：把请求交给框架层的 `SessionManager`，并把流式响应吐给客户端

如果把整个项目类比成 Android App：

| 项目组件 | Android 类比 | 说明 |
|---------|-------------|------|
| `ai_app1` | `Activity` / `Fragment` + `ViewModel` | 最外层，处理用户输入和展示结果 |
| `rag_framework` | `Repository` + `DI Container`（类似 Hilt 的 `ApplicationComponent`） | 所有业务逻辑和依赖注入的中心 |
| `domains/android` | `Plugin Interface` 的实现类 | 告诉框架「Android 领域该用什么提示词、怎么分类查询」 |

与 `ai_app2`（LangGraph 状态图版）和 `ai_app3`（Agentic RAG 版）相比，`ai_app1` 是**最直观、最顺序化**的版本：没有状态图节点编排，也没有 Agent 循环，就是一条笔直的 pipeline。

---

## 2. 整体分层结构

```
┌─────────────────────────────────────────────────────────────┐
│                    ai_app1  (应用层)                          │
│  main.py  ──  lifespan 组装容器、注册路由                      │
│  api/chat.py ──  HTTP 接口、领域路由判断、多领域融合            │
│  "薄包装层：只负责组装和路由，不写业务逻辑"                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用
┌──────────────────────────▼──────────────────────────────────┐
│              rag_framework  (框架层)                          │
│  container.py    ── 依赖注入总控（RAGContainer）              │
│  session/manager ── 会话生命周期 + 消息构建 + 摘要管理          │
│  retrieval/      ── 向量检索、BM25、混合融合、重排序            │
│  embedding/      ── 文本向量化（BGE-M3）                       │
│  llm/            ── 大语言模型客户端（OpenAI / 本地）            │
│  rerank/         ── 交叉编码器重排序（CrossEncoder）            │
│  domain/base.py  ── 领域插件抽象接口                            │
│  core/factories  ── 组件工厂注册表（自注册模式）                 │
│  "所有重型逻辑和可复用组件都在这里"                              │
└──────────────────────────┬──────────────────────────────────┘
                           │ 调用 / 实现接口
┌──────────────────────────▼──────────────────────────────────┐
│              领域插件  (domain层)                             │
│  AndroidDomainPlugin  ── Android 开发专用知识                  │
│  MSMarcoDomainPlugin  ── 通用检索评测领域（可选）               │
│  "告诉框架：这个领域用什么提示词、怎么分类查询、术语怎么映射"     │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 启动流程详解（main.py lifespan）

服务器启动时，`main.py` 的 `lifespan` 会按顺序执行 **7 个步骤**。这相当于 Android 中 `Application.onCreate()` 的初始化阶段。

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 启动阶段 ===
    1. 显式注册所有领域插件
    2. 创建「基础容器」（共享重型组件）
    3. 创建共享 BM25Store（关键词索引，所有领域共用）
    4. 为每个领域创建「派生容器」（共享重型组件，独立 session/retriever）
    5. 并发预热所有 Warmupable 组件
    yield  # ← 服务器开始接受请求
    # === 关闭阶段 ===
    6. 关闭所有 Closable 组件，释放资源
```

### 3.1 步骤拆解

#### 步骤 1：注册领域插件

```python
for cls in _DOMAIN_CLASSES:
    register_domain(cls)
```

- `_DOMAIN_CLASSES` 是一个列表，目前包含 `AndroidDomainPlugin`
- 每个插件必须继承 `DomainPlugin` 抽象基类
- 注册后，框架可以通过 `get_domain("android")` 随时取到该插件实例

> **Kotlin 类比**：就像你有一个 `PluginRegistry` Map，`register_domain()` 相当于 `registry[name] = pluginInstance`。

#### 步骤 2：创建基础容器

```python
base_container = RAGContainer.from_settings(base_settings)
```

这里创建了**第一个** `RAGContainer`，包含所有**共享的、昂贵的**组件：

| 组件 | 作用 | 为什么贵 |
|------|------|---------|
| `embedder` | 把文字变成向量（BGE-M3 模型） | 要加载几百 MB 的神经网络权重 |
| `vector_store` | ChromaDB 向量数据库连接 | 初始化时建立磁盘索引连接 |
| `llm` | 大语言模型客户端 | 连接远程 API 或加载本地模型 |
| `reranker` | 交叉编码器重排序模型 | 也要加载神经网络权重 |

> **Kotlin 类比**：`RAGContainer` 本质上就是 **Hilt 的 `@Singleton Component`**——一次创建，全局共享。它是一个 `frozen dataclass`（不可变数据类），相当于 Kotlin 的 `data class` 但所有字段都是 `val`，构造后不能修改。

#### 步骤 3：创建共享 BM25Store

```python
shared_bm25 = BM25Store(
    index_dir=settings.bm25_index_dir,
    chroma_path=settings.chroma_db_path,
    collection_name="knowledge_base",
)
```

- BM25 是「传统关键词搜索」，和「向量语义搜索」互补
- 所有领域共用同一个 BM25 索引，通过 `domain` metadata 字段区分不同领域的文档

#### 步骤 4：为每个领域创建派生容器

这是最关键的设计。假设注册了 `android` 和 `msmarco` 两个领域：

```python
for name in registered:
    retriever = retriever_registry.create(..., domain_filter=domain.name)
    session_store = session_store_registry.create(...)
    containers[name] = replace(base_container,
        settings=domain_settings,
        domain=domain,
        retriever=retriever,
        session_store=session_store,
    )
```

每个领域得到**自己的** `RAGContainer`，但：

| 属性 | 是否共享 | 说明 |
|------|---------|------|
| `embedder` | ✅ 共享 | 同一个 BGE-M3 实例 |
| `vector_store` | ✅ 共享 | 同一个 ChromaDB 连接 |
| `llm` | ✅ 共享 | 同一个 LLM 客户端 |
| `reranker` | ✅ 共享 | 同一个 CrossEncoder 实例 |
| `session_store` | ❌ 独立 | 每个领域有自己的内存会话存储 |
| `retriever` | ❌ 独立 | 每个领域有自己的检索器，带 `domain_filter` |
| `domain` | ❌ 独立 | 指向各自的 `DomainPlugin` 实例 |

> **为什么这样设计？** 模型加载耗内存（BGE-M3 约 2GB），如果每个领域都独立加载，内存会爆炸。但会话历史必须隔离（不能让 Android 的问题污染 msmarco 的会话）。

> **Kotlin 类比**：相当于 `baseContainer.copy(retriever = newRetriever, sessionStore = newSessionStore)`——复用不变的部分，替换需要独立的部分。

#### 步骤 5：并发预热

```python
warmup_tasks = []
for c in containers.values():
    for comp in c.warmup_targets():
        warmup_tasks.append(comp.warmup())
await asyncio.gather(*warmup_tasks)
```

- 预热 = 提前加载模型到 GPU/内存，避免第一个用户请求时卡顿
- 用 `asyncio.gather` 并发执行，缩短启动时间
- 通过 `id(comp)` 去重，避免共享组件被重复预热

> **Kotlin 类比**：相当于 `coroutineScope { components.map { async { it.warmUp() } }.awaitAll() }`。

---

## 4. 一次对话的完整数据流

当用户在前端输入问题并点击发送，到看到流式回答，中间经历了什么？

### 4.1 HTTP 层（api/chat.py）

```
POST /chat
Body: { "message": "RecyclerView 卡顿怎么优化？", "user_id": "user_001", "domain": "" }
```

#### 4.1.1 领域路由判断

```python
def _resolve_single_container(containers, default_container, domain, query):
    # 1. 用户显式指定？如 "android"
    if domain and domain in containers:
        return containers[domain]
    
    # 2. 自动路由：含中文 → android
    if query and _has_chinese(query):
        return containers.get("android", default_container)
    
    # 3. 回退默认领域
    return default_container
```

在这个例子里，查询包含中文「卡顿」「怎么优化」，所以自动路由到 `android` 领域。

#### 4.1.2 调用容器的 chat_stream

```python
stream = container.chat_stream(req.message, req.user_id)
```

注意：这里不是直接返回字符串，而是返回一个**异步生成器**（`AsyncIterator[str]`），会一个字一个字地 yield 出来。

> **Kotlin 类比**：相当于 `Flow<String>`——不是一次性返回所有数据，而是持续 emit 数据块。

### 4.2 框架层：RAGContainer.chat_stream

`RAGContainer.chat_stream()` 是端到端入口，但它自己不干活，而是委托给 `SessionManager`：

```python
async def chat_stream(self, query: str, user_id: str):
    manager = SessionManager(
        store=self.session_store,
        llm=self.llm,
        retriever=self.retriever,
        domain=self.domain,
        settings=self.settings,
        rule_rewriter=self.rule_rewriter,
        llm_rewriter=self.llm_rewriter,
    )
    async for chunk in manager.chat_stream(query, user_id):
        yield chunk
```

### 4.3 业务核心：SessionManager.chat_stream

`SessionManager.chat_stream()` 是**最核心的方法**，整个对话逻辑都在这里：

```
1. get_session(user_id)
   └─→ 从 session_store 取出该用户的历史对话（内存中，Key-Value 结构）
   
2. add_user_message(session, query)
   └─→ 把用户新问题加入历史
   
3. build_messages(session, query)  ← 【最复杂的一步】
   ├─→ a. _build_raw_messages(session)
   │     └─→ [system_prompt, 历史摘要, history...]
   │
   ├─→ b. _build_routes(query, history)  ← 查询改写 + 分类
   │     ├─→ domain.rewrite_router_rules(query) → 返回 level (0/1/2)
   │     ├─→ level=2 → llm_rewriter.rewrite()  (AI 智能改写)
   │     ├─→ level=1 → rule_rewriter.rewrite()  (规则替换，如中文术语→英文)
   │     └─→ level=0 → domain.classify_query()   (仅分类，不改写)
   │     返回：list[QueryRoute]，每个 Route 决定走哪些检索路径
   │
   ├─→ c. retriever.retrieve(routes)  ← 混合检索（见 4.4）
   │     └─→ 返回 list[RetrievedDoc]，每个文档有 text / score / id
   │
   ├─→ d. 组装最终 messages
   │     ├─→ 如果没检索到文档 → 插入兜底提示（让 AI 道歉）
   │     ├─→ 如果置信度太低 → 插入低置信度兜底提示
   │     └─→ 正常情况 → 追加 "参考资料：\n[文档1]...\n[文档2]..."
   │
   └─→ e. Failure Analysis 收集
         └─→ 未命中 / 低置信度 的查询会被自动记录到失败样本集

4. llm.chat_stream(messages, use_tools=False)
   └─→ 流式调用大模型，一个字一个字返回

5. add_assistant_message(session, full_reply)
   └─→ 把 AI 的完整回答加入历史

6. should_summarize(session)?
   └─→ 如果历史 token 数超过预算 → summarize() 压缩历史

7. trim_history(session)
   └─→ 历史条数超过上限时，裁掉最老的

8. store.save(session)
   └─→ 保存回内存存储
```

#### 以 "RecyclerView 卡顿怎么优化？" 为例：

| 步骤 | 发生了什么 |
|------|-----------|
| `_build_routes()` | `AndroidDomainPlugin.rewrite_router_rules()` 判断：含中文、长度 > 25 → level=2 → 调用 LLM 改写 |
| LLM 改写 | 可能扩写成 "RecyclerView scrolling performance optimization Android best practices" |
| `classify_query()` | `AndroidDomainPlugin.classify_query()` 判断：含 `RecyclerView` 组件名、有驼峰命名 → type="api", routes=["bm25", "dense"] |
| `retrieve()` | 同时走 BM25 关键词检索 + Dense 向量检索（见 4.4） |
| 组装 messages | system_prompt + 历史 + "参考资料：\n[1] RecyclerView 使用 ViewHolder 模式...\n[2] 避免在 onBindViewHolder 中做耗时操作..." |
| LLM 生成 | 模型根据参考资料组织语言，流式返回 |

### 4.4 检索核心：HybridRetriever.retrieve

`HybridRetriever` 是检索的心脏，采用**多路召回 + 融合 + 精排**策略：

```
输入：list[QueryRoute]（来自改写后的查询）
        │
        ▼
┌─────────────────────────────────────┐
│  多路异步并发召回                      │
│  ├─ dense 路：向量检索（child collection）│
│  ├─ hyde 路：HyDE 伪文档向量检索        │
│  └─ bm25 路：关键词检索                │
│  每路独立 timeout，一路失败不影响其他路   │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Weighted RRF 融合                    │
│  把多路的文档列表按 RRF 公式合并打分      │
│  score = Σ weight / (rank + k)        │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  拉取 parent 文本                      │
│  child 是小块，parent 是大块，返回 parent  │
│  保证上下文完整性                       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  CrossEncoder Rerank 精排             │
│  用重排序模型对候选文档重新打分          │
│  timeout 时降级为 RRF 分数             │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  Lost-in-Middle 重排                  │
│  最相关的放开头，次相关的放结尾          │
│  缓解 LLM 对中间内容注意力衰减的问题      │
└─────────────────────────────────────┘
        │
        ▼
输出：list[RetrievedDoc]（按相关性排序，通常取 top 3~6）
```

> **Kotlin 类比**：`retrieve()` 是一个 `suspend` 函数，内部用 `async/await`（Python 的 `asyncio.gather`）并发执行三路检索，类似于 Kotlin 的 `coroutineScope { listOf(async { dense() }, async { bm25() }).awaitAll() }`。

---

## 5. 核心组件契约详解

### 5.1 RAGContainer — 依赖注入容器

```python
@dataclass(frozen=True, slots=True)
class RAGContainer:
    settings: RAGSettings
    embedder: Embedder
    vector_store: VectorStore
    retriever: Retriever
    reranker: Reranker
    llm: LLMClient
    rewriter_llm: LLMClient
    session_store: SessionStore
    domain: DomainPlugin
    rule_rewriter: QueryRewriter | None
    llm_rewriter: QueryRewriter | None
```

- `frozen=True`：构造后不可变（线程安全）
- `slots=True`：内存优化，不创建 `__dict__`
- 通过 `from_settings()` 类方法统一构造，所有组件由**工厂注册表**创建

> **Kotlin 类比**：
> ```kotlin
> @Singleton
> data class RAGContainer(
>     val settings: RAGSettings,
>     val embedder: Embedder,
>     val retriever: Retriever,
>     // ... 所有依赖一次性注入
> )
> ```

### 5.2 SessionManager — 会话编排器

`SessionManager` 是**有状态**的（持有 `store`、`llm`、`retriever` 等依赖），每次 `chat_stream()` 处理一个完整对话轮次。它不负责 HTTP，只负责：

- 会话 CRUD（内存中的 `SessionData`）
- Token 预算监控（估算历史长度，触发摘要）
- 消息构建（system + summary + history + retrieved context）

> **Kotlin 类比**：相当于一个 `ViewModel`，持有 `Repository` 引用，把多个数据源（session store、retriever、llm）的数据编排成最终的 `messages` 列表。

### 5.3 DomainPlugin — 领域契约

```python
class DomainPlugin(ABC):
    @property @abstractmethod def name(self) -> str: ...
    @property @abstractmethod def system_prompt(self) -> str: ...
    @abstractmethod def classify_query(self, query, history) -> QueryRoute: ...
    @abstractmethod def get_collection_names(self) -> CollectionNames: ...
```

每个领域必须实现这 4 个抽象方法。框架通过多态调用，不知道也不关心具体是 Android 还是医学领域。

以 `AndroidDomainPlugin` 为例：

| 方法 | Android 实现 |
|------|-------------|
| `name` | 返回 `"android"` |
| `system_prompt` | 从 `prompts/system.txt` 读取，定义 AI 的 Android 专家身份 |
| `classify_query` | 检查是否含 `Activity`、`RecyclerView` 等组件名 → 决定走 BM25 还是 Dense |
| `get_collection_names` | 返回 `("knowledge_base", "knowledge_base_child", "knowledge_base_hyde")` |
| `rewrite_router_rules` | 短查询/含代词 → level=2（LLM 改写）；命中中文术语 → level=1（规则改写） |
| `get_term_mapping` | 返回 `zh_to_en.json`，如 `{"卡顿": "jank", "内存泄漏": "memory leak"}` |

> **Kotlin 类比**：`DomainPlugin` 是一个 `interface`（Python 的 `ABC` 抽象基类），`AndroidDomainPlugin` 是它的 `class AndroidDomainPlugin : DomainPlugin` 实现。

### 5.4 _Registry — 组件工厂注册表

```python
class _Registry(Generic[T]):
    def register(self, name: str, factory: Callable[..., T]) -> None: ...
    def create(self, name: str, **kwargs) -> T: ...
```

每个组件类型（embedder、llm、retriever 等）都有一个独立的 `_Registry` 实例。实现类在模块底部**自注册**：

```python
# rag_framework/retrieval/fusion.py 底部
def _create_hybrid_retriever(...):
    return HybridRetriever(...)

register_retriever("hybrid", _create_hybrid_retriever)
```

> **Kotlin 类比**：相当于一个 `Map<String, (args) -> T>`，或者 Dagger 的 `@Binds` + `@IntoMap` 组合。`create()` 就是 `map[key]?.invoke(args)`。

---

## 6. 多领域融合模式（domain="all"）

当用户指定 `domain="all"` 时，不走单领域流程，而是走特殊路径：

```python
async def _multi_domain_chat_stream(containers, default_container, query, user_id):
    # 1. 并行从所有领域检索
    tasks = [_retrieve_from_domain(c, query, user_id) for c in containers.values()]
    all_docs = await asyncio.gather(*tasks)
    
    # 2. 合并去重 + 按分数排序
    merged = merge_and_deduplicate(all_docs)
    top_docs = merged[:6]
    
    # 3. 用默认领域的 system_prompt + LLM 生成回答
    messages = build_messages(default_container, top_docs, query)
    async for chunk in default_container.llm.chat_stream(messages):
        yield chunk
    
    # 4. 会话历史保存到默认领域
    save_session(default_container, query, full_reply)
```

关键点：
- **检索是并行的**：`asyncio.gather` 同时向 android 和 msmarco 发起检索
- **生成只用一个 LLM**：避免多个 LLM 同时输出造成混乱
- **去重按 doc.id**：同一篇文档在不同领域被检索到时只保留一份

---

## 7. 新增领域的方法

只需两步：

1. **创建领域包**：在 `domains/` 下新建 `xxx_domain/` 包，实现 `DomainPlugin` 子类（参考 `domains/android/`）
2. **注册到应用层**：在 `ai_app1/main.py` 的 `_DOMAIN_CLASSES` 中 import 并追加：

```python
try:
    from xxx_domain.plugin import XxxDomainPlugin
    _DOMAIN_CLASSES.append(XxxDomainPlugin)
except Exception:
    pass
```

索引构建时，在文档 metadata 中写入 `"domain": "xxx"` 字段即可，无需修改 collection 名称。

---

## 8. 文件职责总览

| 文件 | 职责 | 改动频率 |
|------|------|---------|
| `ai_app1/main.py` | FastAPI 入口、lifespan 组装容器、注册路由 | 新增领域时改 `_DOMAIN_CLASSES` |
| `ai_app1/api/chat.py` | HTTP 接口、`/chat` 路由、领域路由判断、多领域融合 | 极少改动 |
| `rag_framework/container.py` | `RAGContainer` 定义、`from_settings()` 工厂、生命周期方法 | 极少改动 |
| `rag_framework/session/manager.py` | `SessionManager` 完整对话逻辑 | 调优对话流程时改动 |
| `rag_framework/retrieval/fusion.py` | `HybridRetriever`：多路召回 + RRF + Rerank | 调优检索时改动 |
| `rag_framework/domain/base.py` | `DomainPlugin` 抽象基类、`QueryRoute`、`CollectionNames` | 新增领域概念时改动 |
| `rag_framework/core/factories.py` | `_Registry` 泛型注册表、所有组件类型的注册表实例 | 新增组件类型时改动 |
| `domains/android/plugin.py` | Android 领域逻辑实现 | Android 业务调优时改动 |

---

## 9. Kotlin ↔ Python 概念速查表

| Python 概念 | Kotlin 等价物 | 说明 |
|------------|--------------|------|
| `class MyClass:` | `class MyClass` | 基础类定义 |
| `@dataclass(frozen=True)` | `data class` + 所有字段用 `val` | 不可变数据类，自动实现 `__init__`、`__eq__` |
| `ABC` + `@abstractmethod` | `interface` + 抽象方法 | 抽象基类，强制子类实现 |
| `Protocol` | `interface` | 结构子类型（duck typing）|
| `async def` / `await` | `suspend fun` | 协程/异步函数 |
| `async for chunk in stream:` | `stream.collect { chunk -> }` 或 `for (chunk in stream)` | 异步迭代器 |
| `yield` | `emit()` (in Flow) | 生成器，逐步产出数据 |
| `asyncio.gather(*tasks)` | `awaitAll(tasks)` | 并发等待多个协程 |
| `asyncio.to_thread(sync_func)` | `withContext(Dispatchers.IO)` | 把同步代码放到线程池执行 |
| `dict[str, T]` | `Map<String, T>` | 字典/映射 |
| `list[T]` | `List<T>` | 列表 |
| `T \| None` | `T?` | 可选类型（Nullable）|
| `**kwargs` | `vararg` / 命名参数 | 关键字参数展开 |
| `from __future__ import annotations` | 不需要 | 支持前向引用（Python < 3.10 需要）|
| `isinstance(obj, Protocol)` | `obj is MyInterface` | 运行时协议检查 |

---

## 10. 常见调试入口

如果你想跟踪一次请求的全链路，可以查看日志：

1. **检索链路**：看 `retrieval_logger` 输出，包含每路检索耗时、RRF 结果、Rerank 分数
2. **会话状态**：看 `session_logger` 输出，包含 rewrite level、route type、历史长度
3. **首字延迟**：看 `chat_logger` 输出的 `TTFT`（Time To First Token）

---

*本文基于 ai_app1 v2.3.0 代码撰写。如有结构变更，请同步更新。*
