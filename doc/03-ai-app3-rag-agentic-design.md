# ai_app3 - Android RAG 问答系统 (Agentic RAG 第三代)

> 版本: 2.0 | 最后更新: 2026-05-13
> 基于 ai_app1 v2.x (rag_framework) 的 Agentic RAG 升级，引入 LangGraph 编排、Self-RAG、查询分解、知识图谱增强与自适应上下文压缩

---

## 1. 系统定位

ai_app3 是 ai_app1 的 **LangGraph + Agentic RAG 架构升级版本**，面向 Android 开发者的智能问答助手。系统在完全复用 ai_app1 的 `rag_framework` 检索管道（Dense+HyDE+BM25+RRF+Rerank+Lost-in-Middle）的基础上，引入第三代 RAG 核心能力：

- **LangGraph StateGraph 编排**：将对话流程由手写顺序调用重构为显式状态图，流程可视化、节点可独立测试
- **Self-RAG（自我反思式检索）**：检索结果由 LLM 评估充分性，不足时自动触发查询改写或知识图谱扩展，形成检索-评估-改写的迭代闭环
- **查询分解（Query Decomposition）**：将复杂多步问题拆分为独立子查询，分别检索后合并上下文
- **轻量知识图谱（Lightweight KG）**：基于现有 ChromaDB 元数据构建实体关系网络，补充向量检索的盲区（如类间继承、API 调用链）
- **自适应上下文压缩**：超长上下文自动压缩，提取结构化关键事实，生成带层次化引用的回答
- **意图感知路由**：闲聊场景跳过检索直接回复，降低延迟与成本

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              用户请求层                                   │
│                    POST /chat  { "message": "..." }                      │
│                    SSE 流式响应 (trace + content + done)                 │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         FastAPI 应用层 (main.py)                          │
│  - 挂载 chat_router (端口 8002，与 ai_app1:8000 共存)                    │
│  - 静态文件服务 /ui                                                      │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Chat API 路由层 (chat.py)                         │
│  1. 接收请求 → 从 checkpointer 加载线程状态                               │
│  2. 组装 input_state（含 trace 初始空列表）                               │
│  3. 调用 graph.ainvoke 执行完整 Agentic RAG Graph                         │
│  4. SSE 推送: trace → content(逐字) → done                               │
│     Graph 内部：                                                         │
│     intent → decompose → retrieve → evaluate → [rewrite|expand_kg]      │
│     → build_messages → llm → self_check → save_reply → summarize? → trim│
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   Agentic RAG LangGraph 状态图 (graph/)                    │
│                                                                          │
│   START ──→ intent ──┬──→ decompose ──→ retrieve ──→ evaluate ──┬──→ generate
│                      │                                           │
│                      │                                           ├──→ rewrite ──→ retrieve (循环)
│                      │                                           │
│                      │                                           └──→ expand_kg ──→ evaluate (循环)
│                      │
│                      └──→ direct_response ────────────────────────────────┐
│                                                                           │
│   generate: build_messages ──→ llm ──→ self_check ──→ save_reply ────────┤
│                                                                           │
│   save_reply ──→ should_summarize? ──┬──→ summarize ──→ trim ──→ END     │
│                                      └──→ trim ──→ END                   │
│                                                                          │
│  节点说明：                                                               │
│  - intent         : 意图分析（技术/闲聊/澄清/多步推理）                    │
│  - decompose      : 查询分解为子查询列表                                   │
│  - retrieve       : 多路子查询检索（via rag_framework）+ KG 扩展          │
│  - evaluate       : LLM 评估检索充分性，输出 confidence / gaps            │
│  - rewrite        : 基于反馈改写查询，迭代优化                             │
│  - expand_kg      : 知识图谱实体关系扩展                                   │
│  - build_messages : System + 摘要 + history + 压缩后上下文 + 关键事实      │
│  - llm            : ChatOpenAI.bind_tools() + 手工 tool calling 循环      │
│  - self_check     : 回答自检（启发式 + 扩展点）                            │
│  - save_reply     : assistant 回复入 history                              │
│  - summarize      : token 超预算时压缩历史为摘要                          │
│  - trim           : 裁剪 history 到 MAX_HISTORY=4 条                       │
│                                                                          │
│  条件边：                                                                 │
│  - after_intent      : 闲聊/无需检索 → direct_response                    │
│  - after_evaluate    : 充分 → generate | 不足 → rewrite / expand_kg      │
│  - should_summarize  : token 预算检测 → summarize / trim                  │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Agentic 服务层 (service/)                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │
│  │ retriever        │  │ evaluator        │  │ context_compressor│       │
│  │ rag_framework 适配│  │ 检索质量评估     │  │ 自适应压缩       │       │
│  │ query_context()  │  │ 决策下一步       │  │ 关键事实提取     │       │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘       │
│  ┌──────────────────┐  ┌──────────────────┐                             │
│  │ query_engine     │  │ knowledge_graph  │                             │
│  │ 意图分析         │  │ 轻量 KG（内存图）  │                             │
│  │ 查询分解         │  │ 实体提取/关系共现  │                             │
│  │ 查询改写         │  │ 基于 ChromaDB 构建  │                             │
│  │ 上下文合并       │  │ 启动时懒加载       │                             │
│  └──────────────────┘  └──────────────────┘                             │
│  ┌──────────────────┐                                                   │
│  │ tools            │  search_docs / evaluate_answer / multiply          │
│  └──────────────────┘                                                   │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               rag_framework 混合检索管道 (复用 ai_app1)                      │
│                                                                          │
│   service/retriever.py (adapter)                                         │
│   └── RAGContainer.retriever.retrieve(query)                             │
│         → HybridRetriever                                                │
│              Dense + HyDE + BM25 → Weighted RRF → Rerank → Lost-in-Middle│
│                                                                          │
│   ChromaDB (android_parent / android_child / android_hyde)              │
│   BM25 (Tantivy sparse index)                                            │
│   路径由 rag_framework.core.config.RAGSettings 统一管理                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块设计

### 3.1 检索适配器 (service/retriever.py) ← v2 新增

ai_app3 v2 废弃了直接调用 `ai_app1.service.vector_store.query_db()` 的方式，改为通过 `rag_framework.RAGContainer` 统一接入检索能力。

```python
# service/retriever.py
from rag_framework.container import RAGContainer
from rag_framework.core.config import get_settings

_container: RAGContainer | None = None

def _get_container() -> RAGContainer:
    global _container
    if _container is None:
        _container = RAGContainer.from_settings(get_settings())
    return _container

def query_context(query: str, top_k: int = 5) -> str | None:
    """
    执行混合检索，将 RetrievalResult.docs 格式化为纯文本上下文。
    接口与原 query_db() 兼容，内部使用 HybridRetriever。
    """
    result = _get_container().retriever.retrieve(query, top_k=top_k)
    parts = [doc.text for doc in result.docs if doc.text]
    return "\n\n".join(parts) or None
```

**接口变化对比：**

| 维度 | ai_app3 v1 (旧) | ai_app3 v2 (新) |
|------|----------------|----------------|
| 检索入口 | `ai_app1.service.vector_store.query_db()` | `ai_app3.service.retriever.query_context()` |
| 底层实现 | ai_app1 内部函数 | `rag_framework.HybridRetriever.retrieve()` |
| 返回类型 | `str \| None` | `str \| None`（适配层转换） |
| 路径配置 | ai_app3/core/config.py 硬编码 | `rag_framework.RAGSettings` 统一管理 |

### 3.2 Agentic RAG 状态图 (graph/)

#### 3.2.1 状态定义 (state.py)

```python
class RagState(TypedDict):
    user_message: str
    intent: dict | None              # 意图分析结果
    sub_queries: list[dict]          # 子查询列表
    retrieved_context: str | None    # 合并后的检索上下文
    kg_context: str | None           # KG 扩展上下文
    confidence: float                # 检索质量置信度
    evaluation_result: dict | None   # 完整评估结果
    retrieval_iterations: int        # 已执行迭代轮数
    history: list
    summary: str
    token_budget: int
    messages: list
    reply: str
    trimmed: list
    trace: list[dict]                # Agentic 执行轨迹
    needs_tool: bool
    tool_results: list[Any]
```

相比 ai_app1 的 `SessionData`，ai_app3 新增：
- `intent` / `sub_queries`：查询理解与分解产物
- `confidence` / `evaluation_result`：Self-RAG 评估结果
- `retrieval_iterations`：防止无限循环的计数器
- `kg_context`：知识图谱补充信息
- `trace`：前端可视化用的执行轨迹

#### 3.2.2 节点设计 (nodes.py)

| 节点 | 类型 | 说明 |
|------|------|------|
| `intent_node` | sync | LLM 分析意图（技术/闲聊/澄清/多步推理） |
| `decompose_node` | sync | 复杂问题拆分子查询 |
| `retrieve_node` | sync | 多路子查询检索（via `query_context()`）+ KG 扩展 |
| `evaluate_node` | sync | LLM 评估检索充分性 |
| `rewrite_node` | sync | 查询改写迭代 |
| `expand_kg_node` | sync | KG 实体关系扩展 |
| `build_messages` | sync | 组装 LLM messages（含压缩上下文） |
| `llm_node` | async | LLM 生成 + tool calling |
| `direct_response` | async | 闲聊快速回复 |
| `self_check_node` | sync | 回答自检 |
| `save_reply_node` | sync | 保存回复 |
| `summarize_node` | async | 历史压缩 |
| `trim_node` | sync | 裁剪历史 |

#### 3.2.3 条件边 (conditional_edges.py)

| 条件边 | 决策逻辑 |
|--------|----------|
| `after_intent` | `needs_retrieval=True` 且 `intent≠casual` → decompose；否则 → direct_response |
| `after_evaluate` | `confidence≥0.65` → generate；`iter≥2` → 降级 generate；缺实体关系 → expand_kg；否则 → rewrite |
| `should_summarize` | token 超预算 → summarize；否则 → trim |

#### 3.2.4 迭代检索闭环

```
第1轮: retrieve → evaluate (conf=0.45, gaps=["缺少 ViewModel 与 LiveData 关系"])
           ↓
       expand_kg (命中 ViewModel-LiveData 共现文档)
           ↓
       evaluate (conf=0.72) → generate
```

最大迭代轮数由 `MAX_REWRITE_ITERATIONS=2` 控制，避免无限循环。

---

### 3.3 查询引擎 (service/query_engine.py)

#### 3.3.1 意图分析 (intent_analysis)

```json
{
  "intent": "technical|casual|clarify|multi_step",
  "needs_retrieval": true,
  "reason": "用户询问 Android 技术问题"
}
```

- `casual`：问候、闲聊、非技术话题 → 跳过检索，直接回复
- `technical`：标准技术问答 → 正常检索链路
- `multi_step`：涉及多个知识点 → 触发查询分解
- `clarify`：问题模糊 → 可扩展为追问澄清（当前版本保留扩展点）

#### 3.3.2 查询分解 (decompose_query)

对 `multi_step` 或长查询，拆分为最多 `MAX_SUB_QUERIES=3` 个子查询：

```json
[
  {"sub_query": "Android ViewModel 的生命周期", "confidence": 0.92, "reason": "核心知识点"},
  {"sub_query": "LiveData 与 ViewModel 如何配合使用", "confidence": 0.88, "reason": "关联知识点"}
]
```

低置信度子查询（`< SUB_QUERY_MIN_CONFIDENCE=0.55`）自动过滤。

#### 3.3.3 查询改写 (rewrite_query)

基于前次评估反馈（如 "缺少实体关系信息"、"未找到有效结果"），LLM 改写查询用词，提升召回率：

```
原查询: "ViewModel 怎么用"
改写后: "Android Jetpack ViewModel 创建与使用最佳实践"
```

---

### 3.4 检索评估器 (service/evaluator.py)

#### 3.4.1 评估维度

```json
{
  "sufficient": true,
  "confidence": 0.78,
  "gaps": ["缺少代码示例"],
  "reason": "上下文覆盖了概念解释，但缺少具体代码"
}
```

- `sufficient`：上下文是否足以回答查询（需同时满足 `confidence >= 0.65`）
- `gaps`：缺失的信息类型，用于决策下一步动作

#### 3.4.2 决策逻辑

| 条件 | 动作 | 说明 |
|------|------|------|
| sufficient=True | `generate` | 进入回答生成 |
| iteration >= 2 | `generate` | 已达最大迭代，降级生成 |
| gaps 含 "关系/关联/依赖/调用/继承" | `expand_kg` | 知识图谱补充实体关系 |
| 其他 | `rewrite` | 查询改写后重新检索 |

---

### 3.5 上下文压缩器 (service/context_compressor.py)

#### 3.5.1 自适应压缩

当上下文 token 数超过 `target_budget=2048`（DEFAULT_TOKEN_BUDGET 的一半）时：

1. 调用 LLM 压缩保留关键信息
2. 回退策略：截断到目标预算对应的字符数

#### 3.5.2 关键事实提取 (extract_key_facts)

从上下文中提取结构化关键事实，用于生成带引用的回答：

```json
[
  {"fact": "ViewModel 通过 ViewModelProvider 创建", "source": "Jetpack 架构组件文档", "relevance": 0.95}
]
```

#### 3.5.3 Prompt 构建 (build_prompt_context)

最终 LLM prompt 结构：

```
【参考资料】
<压缩后的上下文>

【关键事实提炼】
1. ... (来源: ...)
2. ...

【用户问题】
<query>
```

---

### 3.6 轻量知识图谱 (service/knowledge_graph.py)

#### 3.6.1 设计约束

- **零外部依赖**：不引入 Neo4j / ArangoDB 等图数据库，纯 Python 内存实现
- **路径由 rag_framework 管理**：ChromaDB 路径通过 `rag_framework.core.config.get_settings().chroma_db_path` 自动解析
- **懒加载**：首次调用时构建，构建完成后缓存于进程内存

#### 3.6.2 实体提取规则

| 类型 | 正则规则 | 示例 |
|------|----------|------|
| 大驼峰类名 | `\b[A-Z][a-zA-Z0-9]{2,}\b` | Activity, ViewModel, LiveData |
| 全大写缩写 | `\b[A-Z]{2,6}\b` | NPE, ANR, OOM |
| 小驼峰方法 | `\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b` | onCreate, findViewById |

#### 3.6.3 图谱结构与扩展检索

```python
{
  "nodes": {"ViewModel": {"docs": ["doc_1", "doc_5"], "freq": 12}},
  "edges": [("ViewModel", "LiveData", 5), ("Activity", "Intent", 8)],
  "doc_entities": {"doc_1": ["ViewModel", "LiveData"]}
}
```

当向量检索召回不足时，提取查询中的实体，在图中查找一跳邻居实体，拉取关联文档作为补充上下文。

---

### 3.7 工具定义 (service/tools.py)

| 工具 | 用途 | 说明 |
|------|------|------|
| `search_docs` | 文档检索 | 调用 `query_context()`，供 LLM 主动调用实现 Self-RAG |
| `evaluate_answer` | 回答自检 | 启发式评估，低延迟 |
| `multiply` | 示例工具 | 验证 tool calling 链路 |

---

## 4. 数据流

```mermaid
flowchart TD
    A[用户提问] --> B[/chat API SSE]
    B --> C[checkpointer 加载线程 state]
    C --> D[组装 input_state]
    D --> E[graph.ainvoke]
    E --> F{LangGraph StateGraph}

    F --> F1[intent_node]
    F1 --> F1a{needs_retrieval?}
    F1a -->|否| F1b[direct_response]
    F1a -->|是| F2[decompose_node]

    F2 --> F3[retrieve_node]
    F3 --> F3a["query_context() × N子查询\n(rag_framework HybridRetriever)"]
    F3a --> F3b[KG 扩展]
    F3b --> F4[evaluate_node]

    F4 --> F4a{confidence>=0.65?}
    F4a -->|是| G[build_messages]
    F4a -->|否| F4b{iter<2?}
    F4b -->|是| F4c{gaps含关系?}
    F4c -->|是| F4d[expand_kg]
    F4d --> F4
    F4c -->|否| F4e[rewrite_node]
    F4e --> F3
    F4b -->|否| G

    G --> G1[SystemMessage + 摘要 + history + 压缩上下文 + 关键事实]
    G1 --> H[llm_node]
    H --> H1[ChatOpenAI.ainvoke]
    H1 --> H2{tool_calls?}
    H2 -->|是| H3[执行工具 → ToolMessage]
    H3 --> H1
    H2 -->|否| I[self_check_node]

    I --> J[save_reply_node]
    J --> K{should_summarize?}
    K -->|是| L[summarize_node]
    L --> M[trim_node]
    K -->|否| M
    M --> N[checkpointer 保存 state]
    N --> O[SSE: trace → content → done]
```

---

## 5. 关键配置

| 配置项 | 来源 | 默认值 | 说明 |
|--------|------|--------|------|
| OPENAI_API_KEY | `ai_app1/.env` | — | MiniMax API Key |
| chroma_db_path | `rag_framework.RAGSettings` | `ai_app1/data/chroma_db` | 自动解析，无需硬编码 |
| bm25_index_dir | `rag_framework.RAGSettings` | `ai_app1/data/tantivy_bm25` | 自动解析 |
| MAX_HISTORY | `ai_app3/core/config.py` | 4 | history 保留条数 |
| DEFAULT_TOKEN_BUDGET | `ai_app3/core/config.py` | 4096 | 会话 token 上限 |
| MAX_STEPS | `ai_app3/core/config.py` | 10 | Agent tool calling 最大步数 |
| RETRIEVAL_CONFIDENCE_THRESHOLD | `ai_app3/core/config.py` | 0.65 | 检索质量合格线 |
| MAX_REWRITE_ITERATIONS | `ai_app3/core/config.py` | 2 | 最大检索改写轮数 |
| MAX_SUB_QUERIES | `ai_app3/core/config.py` | 3 | 查询分解最大子查询数 |
| SUB_QUERY_MIN_CONFIDENCE | `ai_app3/core/config.py` | 0.55 | 子查询最低置信度 |
| ENABLE_KNOWLEDGE_GRAPH | `ai_app3/core/config.py` | True | 是否启用 KG |
| embed_model_path / reranker_model_path | `rag_framework.RAGSettings` | 自动解析 | Embedding 与 Reranker 模型路径 |

---

## 6. 运行流程

### 6.1 首次部署

```bash
# 1. 安装依赖（共用 pyproject.toml，rag_framework 已包含在内）
uv sync

# 2. 确保 ai_app1 的索引已构建（ai_app3 复用同一份向量库）
uv run python -m ai_app1.scripts.init_vector_db_v2

# 3. 复用 ai_app1/.env（rag_framework 自动读取）

# 4. 启动 ai_app3（端口 8002，与 ai_app1:8000 共存）
uv run python -m uvicorn ai_app3.main:app --host 0.0.0.0 --port 8002
```

### 6.2 API 调用

```bash
curl -N -X POST http://localhost:8002/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Android 中 ViewModel 和 LiveData 是什么关系？"}'
```

SSE 事件类型：
- `data: {"type":"trace","payload":[...]}` — Agentic 执行轨迹
- `data: {"type":"content","payload":"..."}` — 逐字回复内容
- `data: {"type":"done","payload":{"elapsed_sec":1.23}}` — 完成标记

### 6.3 Web UI

访问 `http://localhost:8002/ui`，右侧侧边栏实时展示 Agentic 执行轨迹（intent → decompose → retrieve → evaluate → ...）。

---

## 7. 设计决策记录

| 决策 | 选择 | 理由 |
|------|------|------|
| 检索层 | `rag_framework.RAGContainer` | ai_app1 已完成检索管道迁移至 rag_framework，ai_app3 直接复用，保持单一数据源 |
| 检索适配器 | `service/retriever.py` 薄包装 | 隔离 rag_framework 接口变化，对 nodes.py 和 tools.py 保持兼容签名 |
| 路径管理 | `rag_framework.RAGSettings` | 统一管理 chroma_db / bm25 / model 路径，消除 ai_app3 内的路径硬编码 |
| Agentic 编排 | LangGraph StateGraph + 条件边循环 | 解决 ai_app1 "检索一次即生成"的限制，支持迭代式检索-评估-改写闭环 |
| 意图识别 | LLM 分类 + `needs_retrieval` 标志 | 闲聊场景无需检索，降低延迟与 API 成本 |
| 知识图谱 | 轻量内存图（无外部数据库） | 零部署成本，基于现有 ChromaDB 数据自动构建 |
| 评估器 | LLM 评估 + 启发式决策 | 评估准确性与延迟的折中 |
| 上下文压缩 | LLM 压缩 + 截断回退 | 避免超长上下文稀释注意力，同时保留关键代码片段 |
| 流式响应 | SSE (trace + content) | 前端可实时观察 Agentic 思考过程，提升可解释性 |
| 会话持久化 | MemorySaver（同 ai_app1） | 零配置启动，后续可替换为 Redis/Postgres |

---

## 8. 已知风险与缓解措施

### 8.1 rag_framework 版本耦合（v2 新增）

**风险**：ai_app3 通过 `service/retriever.py` 适配层依赖 `rag_framework`，若 `RAGContainer` 或 `HybridRetriever` 接口发生变更，适配层需同步更新。

**缓解**：
- `service/retriever.py` 为唯一接触点，接口变更只需修改一处
- `query_context()` 对 nodes.py 和 tools.py 保持兼容签名，内部变更透明

### 8.2 RAGContainer 首次初始化延迟（v2 新增）

**风险**：`RAGContainer.from_settings()` 首次初始化需加载 Embedding 模型、Reranker 模型，可能带来数秒启动延迟。

**缓解**：
- 进程级单例（`_container` 全局变量），仅在第一次检索请求时延迟初始化
- 与 ai_app1 共用 .venv，模型已缓存在磁盘，加载速度可接受

### 8.3 意图分析延迟

**风险**：每轮请求增加一次 LLM 调用（意图分析），带来 200~500ms 额外延迟。

**缓解**：
- 意图分析 prompt 极简，只要求输出 JSON
- 未来可替换为本地分类器（BERT/蒸馏模型）

### 8.4 查询改写循环风险

**风险**：改写后的查询可能与原查询偏离，导致答非所问。

**缓解**：
- `MAX_REWRITE_ITERATIONS=2` 严格限制循环深度
- trace 中记录每轮改写，便于人工审计

### 8.5 复用 ai_app1 的已知风险

| 风险 | 状态 | 说明 |
|------|------|------|
| 检索冗余与上下文污染（A） | ✅ 已缓解 | `rag_framework.HybridRetriever` 内部已处理 `seen_ids` 去重 |
| Rerank 线性评分量级不统一（B） | ✅ 已缓解 | `normalized_rrf` 归一化在框架层保留 |
| Token 估算中英文混合陷阱（C） | ✅ 已缓解 | `_estimate_tokens()` 复用相同加权字符数算法 |
| Summarize 的上下文断裂（D） | 风险不变 | `MAX_HISTORY=4` 保留 |
| LangGraph 版本兼容性 | 风险不变 | 锁定稳定 API，避免实验性功能 |
| 流式响应延迟 | ✅ 已缓解 | SSE 真流式（trace + content） |

---

## 9. ai_app1 vs ai_app3 对比总结

| 维度 | ai_app1 (rag_framework) | ai_app3 (Agentic RAG) |
|------|------------------------|----------------------|
| **代码量** | 框架调用为主 | ~700 行 Agentic 层 |
| **编排框架** | 手写顺序 / SessionManager | LangGraph StateGraph + 条件边循环 |
| **检索策略** | 单路一次检索 | **多路子查询 + 迭代检索 + KG 扩展** |
| **检索层** | `RAGContainer.chat_stream()` 内含 | `service/retriever.py` 适配层 → `RAGContainer.retriever` |
| **查询理解** | 无 | **意图分析 + 查询分解 + 改写** |
| **检索评估** | 无 | **LLM 评估充分性 + 决策下一步** |
| **知识增强** | 无 | **轻量知识图谱（实体关系网络）** |
| **上下文压缩** | 无 | **自适应压缩 + 关键事实提取** |
| **回答自检** | 无 | **Self-RAG 启发式自检** |
| **流式输出** | 原生 token 级 | SSE 真流式（trace + content） |
| **可观测性** | logger | **前端实时展示 Agentic 执行轨迹** |
| **检索能力** | Dense+HyDE+BM25+RRF+Rerank+L-i-M | **完全复用（via rag_framework adapter）** |
| **会话管理** | MemorySessionStore | MemorySaver checkpointer |

---

## 10. 后续优化方向

1. **真 token 级流式**：MiniMax API 支持流式 tool calling 后，迁移至 `graph.astream_events()`
2. **本地意图模型**：将意图分析替换为本地 BERT/蒸馏模型，降低 LLM 调用延迟
3. **多跳 KG 查询**：当前仅支持 1-hop 邻居扩展，可扩展为 2-hop 多跳推理
4. **结构化摘要**：`summarize_node` 使用 JSON Schema 输出，保留关键代码片段和行号
5. **持久化升级**：将 `MemorySaver` 替换为 `PostgresSaver`，支持服务重启后恢复会话
6. **检索结果缓存**：高频查询缓存子查询分解结果与检索上下文，降低重复检索成本
7. **A/B 评估框架**：离线对比 ai_app1 与 ai_app3 的回答质量，量化 Agentic RAG 的收益
