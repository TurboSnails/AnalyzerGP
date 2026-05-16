# ai_app4 架构文档

> 商业级分级 Agentic RAG — 从 ai_app1 单线流水线到生产就绪的智能客服

---

## 为什么需要 ai_app4？

ai_app1 是一条笔直的流水线：检索 → 生成。面对真实用户时，它有三个无法回避的缺陷：

| 问题 | ai_app1 的表现 |
|------|--------------|
| 用户在多个领域来回问 | 跨语言追问时 session 断档，AI 不知道"上个问题"是什么 |
| 一句话包含多个领域子问题 | 只能选一个领域检索，另一半靠 LLM 凭记忆猜 |
| 负面情绪 / 投诉 / 转人工 | 没有情绪感知，统一当知识问答处理 |

ai_app4 的核心思路：**用最便宜的方案解决当前这个问题，能不用 Agent 就不用**。大多数问题（约 80%）其实是简单单域问题，只有少数需要完整的 Agent 推理。

---

## 整体架构：分级漏斗

```
用户问题进来
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Tier 0：缓存层                                       │
│  相同问题 → 直接返回缓存，0 成本，<10ms               │
└──────────────────────────┬──────────────────────────┘
                           │ 未命中
                           ▼
┌─────────────────────────────────────────────────────┐
│  Tier 1：classify 节点（本地 PyTorch 小模型）         │
│  意图分类 + 情感分析 + NER，10~30ms                   │
│  输出：intent / sentiment / entities / escalation    │
└──────────────────────────┬──────────────────────────┘
                           │ 按 intent 分流
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
   投诉/转人工         闲聊/简单         知识问答/复杂
   escalate 节点    generate 节点      retrieve 节点
          │                │                │
          ▼                ▼                ▼
     handoff              END           evaluate 节点
      (END)                           ┌────┴────┐
                                   高置信    低置信
                                   generate  rewrite
                                      │        │
                                      │     retrieve（循环，最多3次）
                                      ▼
                                   save_reply
                                      │
                                     END
```

---

## 核心概念：状态机（State Machine）vs 流水线

ai_app1 是**线性流水线**：每个请求都走完全相同的步骤，无论问题简单还是复杂。

ai_app4 是**有状态的图（LangGraph StateGraph）**：每个节点根据当前状态决定下一步走哪条边，简单问题走短路径，复杂问题可以循环。

```
流水线（ai_app1）：  A → B → C → D（固定顺序，不可回头）

状态图（ai_app4）：  A → B → 判断 → C 或 D
                              ↑         │
                              └─────────┘ （可以循环）
```

**Kotlin 类比**：ai_app1 是一个普通函数调用链；ai_app4 是一个 `StateFlow` 驱动的状态机，状态转移由条件逻辑决定。

---

## 状态定义（CS4State）

所有节点共享同一个状态对象，节点只返回**需要更新的字段**（不需要完整返回所有字段）：

```python
class CS4State(TypedDict):
    # 用户输入
    user_message: str       # 用户当前这句话
    user_id: str
    tenant_id: str          # 多租户标识（不同客户的私有知识库）

    # Tier 1 分类结果（classify 节点写入）
    intent: str             # "general_inquiry" | "escalation_request" | "complaint" | "chitchat"
    intent_score: float
    sentiment: str          # "positive" | "neutral" | "negative"
    sentiment_score: float
    entities: list[dict]    # NER 实体，如 [{"type": "order_id", "value": "12345"}]

    # 检索与生成（retrieve/evaluate/generate 节点写入）
    retrieved_context: str  # 检索到的文档原文（拼接后）
    confidence: float       # 检索质量置信度，决定是否需要改写重试
    retrieval_iterations: int  # 已检索次数，防止无限循环
    reply: str              # LLM 生成的最终回复

    # 会话历史（save_reply 节点维护）
    history: list[dict]     # [{"role": "user", "content": "..."}, ...]
    summary: str            # 历史摘要（历史过长时压缩）
    token_budget: int       # 剩余 token 预算

    # 客服专属（escalate/handoff 节点写入）
    escalation_triggered: bool
    escalation_reason: str
    agent_id: str | None    # 接管的人工客服 ID

    # 可观测性
    trace: list[dict]       # 每个节点追加自己的耗时和关键数据
```

---

## 节点详解

### classify 节点 — 分级漏斗的第一刀

```
输入：user_message
输出：intent / sentiment / entities / escalation_triggered
耗时：10~30ms（本地 PyTorch 模型推理）
```

这是整个系统最关键的节点，它的输出决定后续走哪条路径：

```python
# 判断逻辑（conditional_edges.py: after_classify）
if escalation_triggered or sentiment == "negative":
    → escalate（转人工路径，成本最低，速度最快）
elif intent == "chitchat":
    → generate（直接生成，跳过检索）
else:
    → retrieve（标准知识问答路径）
```

为什么用本地 PyTorch 模型而不是 LLM？
- LLM 调用一次要 500ms~2s，成本高
- 分类任务模型（BERT/DistilBERT 级别）本地推理 10~30ms，成本接近 0
- **不需要联网，不增加 API 费用**

### retrieve 节点 — 双引擎检索

```
输入：user_message / entities
输出：retrieved_context / retrieval_iterations
```

优先使用 LlamaIndex，失败则降级到 HybridRetriever：

```
LlamaIndex QueryEngine（支持更复杂的索引结构，如 Summary / KG）
        ↓ 失败或未配置
HybridRetriever（BM25 + Dense + Rerank，来自 rag_framework）
```

这种设计叫**降级策略（Graceful Degradation）**：主路径失败时自动换备用路径，而不是直接报错。

### evaluate 节点 — 检索质量门控

```
输入：retrieved_context / retrieval_iterations
输出：confidence（0.0~1.0）
```

```python
# conditional_edges.py: after_evaluate
if confidence >= 0.6 or iterations >= 3:
    → generate（置信度够了，或已重试3次，生成回答）
else:
    → rewrite（改写查询，重新检索）
```

**最多循环 3 次**：防止无限重试消耗资源。3 次仍然置信度低，就强制进入 generate，让 LLM 基于现有信息尽力回答，并在回复中说明不确定性。

### escalate + handoff 节点 — 转人工路径

```
escalate：生成安抚话术（根据 sentiment 调整措辞）
handoff ：标记 agent_id = None，等待坐席系统分配
```

为什么转人工要有"安抚话术"单独一个节点？因为用户情绪已经是 `negative`，这时候的回复措辞比内容更重要，应该走专门优化过的模板，而不是让 LLM 自由发挥。

---

## 三个核心问题的解决方案

### 问题 1：用户来回问多个领域

**当前方案（ai_app4 已实现）：**
- `MemorySaver` 是 LangGraph 内置的会话持久化机制，按 `thread_id`（即 `user_id`）存储完整的状态图快照
- 每轮对话结束后，整个 `CS4State`（包括 `history`、`summary`）都被持久化
- 下一轮进来时，状态自动恢复——不管上轮是什么领域，历史都在

```python
# builder.py
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# 调用时传入 thread_id，LangGraph 自动恢复该用户的状态
await graph.ainvoke(
    {"user_message": "上个问题我还没懂"},
    config={"configurable": {"thread_id": user_id}}
)
```

**ai_app1 的断档问题在这里被彻底解决**：历史不再分散在各个领域的独立字典里，而是统一存在 MemorySaver 里，追问永远有上下文。

### 问题 2：一句话包含多个领域子问题

**当前状态：** 基础框架已搭好（entities 字段预留了多域信息），但问题拆解逻辑尚未实现。

**设计方案（待实现）：**

在 classify 节点之后，新增 `decompose` 节点：

```
classify → decompose → 并行 retrieve（多个子问题） → merge → evaluate → generate
```

```python
# 示例：sub_queries 字段
sub_queries = [
    {"text": "Room 数据库事务怎么实现", "domain": "android"},
    {"text": "PostgreSQL 事务 ACID 机制",  "domain": "database"},
]

# 并行检索
results = await asyncio.gather(*[
    retrieve_for(q["text"], domain=q["domain"])
    for q in sub_queries
])

# 合并：去重 + 按分数排序 + 截取 top_k
merged_context = merge_and_deduplicate(results)[:6]
```

**触发条件**：当 classify 输出的 `entities` 包含多个不同领域的关键实体时，进入 decompose 节点；单域问题跳过，保持快速路径。

### 问题 3：负面情绪 / 投诉

**已完整实现**：`classify → escalate → handoff` 路径，根据 sentiment 动态调整安抚措辞。

---

## 多租户支持

`tenant_id` 字段贯穿整个 State，用于：

```
检索时过滤：domain_filter = f"{tenant_id}_{domain}"
  → 租户 A 只能检索自己的文档，租户 B 的文档完全隔离

会话存储：thread_id = f"{tenant_id}_{user_id}"
  → 不同租户的同名用户互不干扰
```

---

## 可观测性：trace 字段

每个节点都往 `trace` 列表里追加一条记录：

```python
trace = [
    {"node": "classify", "intent": "general_inquiry", "sentiment": "neutral", "latency_ms": 18.3},
    {"node": "retrieve",  "has_context": True,          "latency_ms": 312.1},
    {"node": "evaluate",  "confidence": 0.82},
    {"node": "generate",  "reply_length": 287,           "latency_ms": 891.4},
    {"node": "save_reply"},
]
```

这让每次请求的完整链路都可以被记录、分析和复现，是生产环境必备的调试能力。

---

## 与其他 app 的对比

| 维度 | ai_app1 | ai_app3 | ai_app4 |
|------|---------|---------|---------|
| 架构模式 | 线性流水线 | 基础 Agentic RAG | 分级 Agentic RAG |
| 领域路由 | 规则（含中文→android） | 无多域 | 意图分类（PyTorch 模型）|
| 情绪感知 | 无 | 无 | 情感分析（PyTorch 模型）|
| 转人工 | 无 | 无 | escalate + handoff 节点 |
| 会话持久化 | 内存字典（进程内） | 无 | LangGraph MemorySaver |
| 多租户 | 无 | 无 | tenant_id 全链路隔离 |
| 跨域追问 | 会断档 | 无 | 完整历史恢复，不断档 |
| 多域子问题 | 选一个域 | 选一个域 | 拆解并行（待实现）|
| 可观测性 | 日志 | 日志 | trace 字段（结构化）|

---

## 文件职责一览

| 文件 | 职责 |
|------|------|
| `main.py` | FastAPI 入口，挂载路由 |
| `lifespan.py` | 启动时初始化 CS4Container，注册 PyTorch 模型 |
| `core/container.py` | CS4Container：在 RAGContainer 基础上扩展 torch_models 管理 |
| `core/config.py` | CS4Settings：客服专属配置（cs_system_prompt、escalation 规则等）|
| `core/context.py` | 全局上下文传递（避免 nodes.py 循环导入 main.py）|
| `graph/state.py` | CS4State 定义，所有节点共享的状态结构 |
| `graph/builder.py` | StateGraph 构建，节点注册，边连接，编译图 |
| `graph/nodes.py` | 所有节点实现（classify / retrieve / evaluate / rewrite / generate / escalate / handoff / save_reply）|
| `graph/conditional_edges.py` | 条件路由逻辑（after_classify / after_evaluate）|
| `api/chat.py` | HTTP 接口，将请求转换为 CS4State 并调用图 |
| `service/` | 待扩展：sub_query 拆解、多域合并、缓存层 |

---

## 待实现的路线图

按优先级排序：

1. **Tier 0 缓存层**：在 api/chat.py 层拦截，完全相同的问题直接返回，节省 90% 的重复查询成本
2. **多域子问题拆解**：`decompose` 节点 + 并行检索 + 结果合并，解决"一句话跨域"问题
3. **LLM-based 评估**：用小 LLM 替代当前的启发式 `confidence` 计算，准确率更高
4. **坐席分配对接**：`handoff` 节点接入真实的坐席系统 API，实现真正的人机协作
5. **知识图谱集成**：`kg_context` 字段已预留，可接入 `ai_app3` 的 `knowledge_graph.py`
