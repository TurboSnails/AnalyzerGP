# AGENTS.md — AnalyzerGP (fenxicb)

> 本文件面向 AI 编程助手。阅读者被假设对该项目一无所知。
> 项目文档与注释主要使用中文，因此本文件以中文撰写。

---

## 1. 项目概览

**AnalyzerGP**（仓库名 `fenxicb`）是一个复合型 AI 工具项目，核心围绕 **RAG（检索增强生成）** 技术栈构建，面向 Android 开发者提供智能问答服务。

项目包含三个独立的 FastAPI 应用层、一个可复用的 RAG 框架包、以及一个 Android 领域插件包：

| 组件 | 定位 | 运行端口 | 说明 |
|------|------|---------|------|
| `ai_app1` | Android RAG 问答助手（传统架构） | 8000 | 基于 `rag_framework` 的 `SessionManager` 手写顺序调用 |
| `ai_app2` | LangGraph 重构版 | 8001 | 将对话流程迁移为 `StateGraph` 状态图编排，复用 `rag_framework` 全部能力 |
| `ai_app3` | Agentic RAG 第三代 | 8002 | 引入 Self-RAG、查询分解、轻量知识图谱、自适应上下文压缩、意图感知路由 |
| `rag_framework` | 通用 RAG 框架包 | — | 可安装的 Python 包，提供 embedding、检索、rerank、LLM 客户端、会话管理、评测框架 |
| `domains/android` | Android 领域插件 | — | 实现 `DomainPlugin` 抽象基类，收敛 Android 专用知识（术语映射、查询分类、HyDE Prompt 等） |

三个应用层共享同一个 Python 虚拟环境与依赖管理，但运行时完全独立，各自拥有独立的配置和端口。

---

## 2. 技术栈

| 层级 | 技术选择 | 版本/说明 |
|------|---------|----------|
| 语言 | Python | >=3.13（根项目），`rag_framework` 子包要求 >=3.11 |
| 包管理 | `uv` | 统一依赖管理，使用 `pyproject.toml` + `uv.lock` |
| Web 框架 | FastAPI + uvicorn | ai_app1/2/3 均为 FastAPI 入口 |
| 向量数据库 | ChromaDB | `android_parent` / `android_child` / `android_hyde` 三个集合 |
| 稀疏检索 | Tantivy + jieba | 磁盘级 BM25 索引，Rust 后端 |
| Embedding | sentence-transformers (BGE-M3) | 本地模型，惰性加载 |
| Rerank | CrossEncoder (bge-reranker-base) | 语义精排，线程安全锁保护 |
| LLM 统一接口 | OpenAI-compatible | 支持 MiniMax、OpenAI、Ollama、本地 Qwen2.5-1.5B-Instruct |
| 编排框架 | LangGraph | ai_app2 / ai_app3 使用 `StateGraph` + `MemorySaver` |
| 数据科学 | pandas, numpy | 量化分析基础依赖 |
| 配置管理 | Pydantic Settings | 环境变量前缀 `RAG_`，支持 `.env` 加载 |

---

## 3. 项目结构

```
fenxiCB/
├── pyproject.toml              # 根项目依赖（含 editable 本地包 rag-framework / android-domain）
├── uv.lock                     # 锁定文件
├── setup.sh                    # 新环境初始化脚本（检查 uv → uv sync → 复制 .env 模板）
├── .gitignore                  # 已覆盖 Python 产物、虚拟环境、密钥、模型、数据索引等
│
├── doc/                        # 人工编写的设计文档（中文）
│   ├── README.md               # 项目总览（含旧版 investment_analyzer 说明，注意部分目录描述可能过期）
│   ├── 01-ai-app1-rag-design.md
│   ├── 02-ai-app2-rag-langgraph-design.md
│   └── 03-ai-app3-rag-agentic-design.md
│
├── rag_framework/              # 通用 RAG 框架（editable install）
│   ├── pyproject.toml
│   └── rag_framework/          # 源码根
│       ├── container.py        # RAGContainer — 不可变 frozen dataclass 依赖注入总控
│       ├── core/               # 配置、工厂注册表、生命周期协议、日志、异常、领域注册中心
│       ├── domain/             # DomainPlugin 抽象基类
│       ├── embedding/          # STEmbedder（BGE-M3）
│       ├── indexing/           # VectorIndexer、Chunker、HyDE 问题生成
│       ├── llm/                # OpenAILLMClient、LocalLLMClient、tool_registry
│       ├── rerank/             # CrossEncoderReranker、FallbackReranker
│       ├── retrieval/          # ChromaVectorStore、BM25Store、HybridRetriever、QueryRewriter
│       ├── session/            # SessionManager、MemorySessionStore
│       └── eval/               # 评测与可观测性（13+ 模块）
│
├── domains/android/            # Android 领域插件（editable install）
│   ├── pyproject.toml
│   └── android_domain/
│       ├── plugin.py           # AndroidDomainPlugin 实现
│       ├── prompts/            # system.txt、hyde.txt
│       ├── terms/              # zh_to_en.json（中文术语→英文 keyword 映射）
│       ├── eval/               # benchmark.json、hard_cases.json、qa_benchmark.json、corpus.txt
│       └── scripts/            # init_vector_db.py（已废弃）、init_vector_db_v2.py（生产索引）
│
├── ai_app1/                    # 传统 RAG 应用层
│   ├── main.py                 # FastAPI + lifespan（注册插件 → 创建容器 → 预热 → 关闭）
│   ├── api/chat.py             # /chat POST 路由，StreamingResponse 流式返回
│   ├── data/                   # 源知识文档 + chroma_db/ + tantivy_bm25/
│   ├── scripts/                # 模型下载辅助脚本
│   ├── static/index.html       # 暗色主题 Web 聊天测试页
│   ├── tests/test_api.py       # API 端到端测试（Mock 容器，7 个用例）
│   └── .env / .env.example
│
├── ai_app2/                    # LangGraph 重构版
│   ├── main.py                 # FastAPI 入口（端口 8001）
│   ├── api/chat.py             # 调用 graph.ainvoke，模拟流式逐字 yield
│   ├── core/                   # 配置、容器单例、日志
│   ├── graph/                  # StateGraph 构建器、节点、状态定义
│   ├── service/                # retriever.py 适配器、tools.py
│   ├── static/index.html       # LangGraph 标签聊天页
│   └── .env.example
│
├── ai_app3/                    # Agentic RAG 第三代
│   ├── main.py                 # FastAPI 入口（端口 8002）
│   ├── api/chat.py             # SSE 流式（trace → content → done）
│   ├── core/                   # 配置、日志
│   ├── graph/                  # 完整 Agentic 状态图 + 条件边
│   ├── service/                # retriever、evaluator、query_engine、knowledge_graph、context_compressor、tools
│   ├── static/index.html       # 带 Agentic Trace 侧边栏的聊天页
│   └── .env.example
│
├── models/                     # 本地模型权重（gitignored，运行时下载）
│   ├── bge-m3/
│   ├── bge-reranker-base/
│   └── qwen2.5-1.5b-instruct/
│
└── scripts/
    └── verify_refactor.py      # 重构验证脚本（语法检查 + 导入检查 + 配置类型检查 + 领域插件检查）
```

---

## 4. 构建与运行命令

### 4.1 首次环境初始化

```bash
# 方式一：使用 setup.sh（推荐）
./setup.sh

# 方式二：手动
uv sync
# 然后手动复制各 app 的 .env.example → .env 并填入 API Key
```

`setup.sh` 会：
1. 检查并安装 `uv`
2. 执行 `uv sync`（安装根项目依赖 + 两个 editable 本地包）
3. 为 `ai_app1`、`ai_app2`、`ai_app3` 复制 `.env.example` → `.env`（若不存在）

### 4.2 启动各应用

三个应用可同时运行，分别监听不同端口：

```bash
# ai_app1（传统架构）— 端口 8000
uv run python -m uvicorn ai_app1.main:app --host 0.0.0.0 --port 8000

# ai_app2（LangGraph）— 端口 8001
uv run python -m uvicorn ai_app2.main:app --host 0.0.0.0 --port 8001

# ai_app3（Agentic RAG）— 端口 8002
uv run python -m uvicorn ai_app3.main:app --host 0.0.0.0 --port 8002
```

### 4.3 离线索引构建（必须步骤，否则检索无结果）

```bash
# 生产索引（推荐）：parent + child + hyde + bm25
uv run python -m domains.android.scripts.init_vector_db_v2 [--data-dir PATH] [--reset] [--no-hyde]

# 验证索引
uv run python -m ai_app1.pre.verify_phase2
```

### 4.4 模型下载（若本地模型目录为空）

```bash
# BGE-M3 Embedding
uv run python -m ai_app1.scripts.download_bge_m3

# Qwen2.5-1.5B-Instruct（用于本地 Query Rewriter）
uv run python -m ai_app1.scripts.download_qwen_rewriter
```

---

## 5. 测试指令

### 5.1 API 端到端测试

```bash
uv run pytest ai_app1/tests/test_api.py -v
```

- 使用 `TestClient` + 纯内存 Mock 容器（FakeLLM / FakeRetriever / FakeEmbedder 等）
- **不依赖真实模型、数据库或文件系统**，可在 CI 环境运行
- 测试覆盖：健康检查、SSE 流式、内容拼接、输入校验、用户隔离、CORS

### 5.2 重构验证（冒烟测试）

```bash
uv run python scripts/verify_refactor.py
```

检查项：
1. 关键模块文件语法正确性（AST parse）
2. `rag_framework`、`android_domain`、`ai_app1` 可正常导入
3. `RAGSettings` 类型检查
4. `AndroidDomainPlugin` 接口合规性检查

### 5.3 RAG 离线评测

```bash
# 综合评测调度器（默认运行 ranking + rewrite + rerank + ablation + hard）
uv run python -m rag_framework.eval.comprehensive_eval all

# 单项评测
uv run python -m rag_framework.eval.comprehensive_eval ranking
uv run python -m rag_framework.eval.comprehensive_eval rewrite
uv run python -m rag_framework.eval.comprehensive_eval rerank

# 输出位置：reports/comprehensive_YYYYMMDD_HHMMSS.md 与 .json
```

> **注意**：`rag_framework` 包内部**没有 pytest 测试文件**。所有框架级测试目前通过 `verify_refactor.py` 和 `comprehensive_eval` 完成。

---

## 6. 代码风格与架构约定

### 6.1 语言与注释

- **所有代码注释、文档字符串、设计文档均使用中文**。修改或新增代码时应保持这一惯例。
- 模块级 docstring 通常包含职责说明和生命周期概述。

### 6.2 类型系统

- 全面使用 Type Hints，`from __future__ import annotations` 频繁出现以支持前向引用。
- 抽象基类使用 `ABC` + `@abstractmethod` 定义。
- 数据类优先使用 `@dataclass(frozen=True, slots=True)`（如 `RAGContainer`）。

### 6.3 架构模式

| 模式 | 说明 |
|------|------|
| **工厂注册表（自注册）** | `rag_framework/core/factories.py` 提供泛型 `_Registry`。各实现类在模块底部通过 `register_xxx()` 自注册。`__init__.py` 和 `container.py` 顶部通过副作用导入触发注册（带 `noqa: F401`）。 |
| **依赖注入容器** | `RAGContainer` 为不可变 frozen dataclass，通过 `from_settings()` 按序组装所有组件。应用层通过 FastAPI `app.state.container` 注入，零全局状态。 |
| **生命周期协议** | `Warmupable` / `Closable` Protocol 定义在 `core/lifecycle.py`。`lifespan` 通过 `isinstance` 统一预热和关闭，无需关心具体实现。 |
| **领域插件** | `DomainPlugin` 抽象基类收敛所有领域特定知识。`AndroidDomainPlugin` 在 `lifespan` 中**显式注册**，模块底部不执行 `register_domain`（去 import-time 副作用）。 |
| **查询路由** | `QueryRoute` dataclass 描述扩写后的查询：`text`、`type`、`weight`、`routes`。 |

### 6.4 异步与并发

- FastAPI 路由和 LLM 客户端使用 `async/await`。
- 同步阻塞操作（如向量检索）使用 `asyncio.to_thread()` 避免阻塞事件循环。
- 多路并发检索使用 `ThreadPoolExecutor(max_workers=3)`。
- LLM 并发限制通过 `asyncio.Semaphore` 实现。

### 6.5 模型加载策略

- **惰性加载**：Embedding 模型、Reranker、本地 LLM 均在首次调用时加载（双重检查锁保证线程安全）。
- **预热**：`lifespan` 中统一调用所有 `Warmupable` 组件的 `warmup()`，避免首请求阻塞。

### 6.6 异常体系

`rag_framework/core/exceptions.py` 定义了分层异常：
- `RAGError`（基类）
- `ConfigError`、`ModelLoadError`、`LLMError`、`RetrievalError`、`RerankError`、`SessionError`、`IndexingError`、`DomainError`

新增异常时应继承自 `RAGError` 或对应层级子类。

### 6.7 配置管理

- 统一使用 `RAGSettings`（Pydantic `BaseSettings`），环境变量前缀为 `RAG_`。
- 配置加载顺序：`.env` → `ai_app1/.env`（`override=False`，不覆盖已有环境变量）。
- 路径解析为动态计算（如优先 `ai_app1/data/chroma_db`，否则 `pre/chroma_db`），不硬编码绝对路径。

---

## 7. 安全注意事项

### 7.1 API Key 管理

- API Key（MiniMax / OpenAI / HuggingFace）存储在各 app 的 `.env` 文件中。
- `.env` 已被 `.gitignore` 保护，**切勿将含真实 Key 的 `.env` 提交到仓库**。
- `ai_app1/.env` 是事实上的主配置源，`ai_app2` 和 `ai_app3` 通常复用同一文件或各自有一份 `.env.example`。

### 7.2 CORS

- 所有 FastAPI 应用均配置了 `allow_origins=["*"]`、`*` methods、`*` headers。
- **这是开发/本地测试配置**，若部署到公网，应收紧为具体域名。

### 7.3 无认证层

- 当前三个应用均**无身份认证或授权机制**。`user_id` 仅用于会话隔离，不做任何校验。
- 公网部署时必须在前置网关（Nginx / API Gateway）补充认证，或在应用层增加鉴权中间件。

### 7.4 工具调用安全

- `tool_registry` 注册的函数会被 LLM 在 tool calling 循环中自动执行。
- 注册工具时必须确保参数校验和副作用控制，避免暴露危险操作（如文件删除、系统命令执行）。

---

## 8. 开发工作流提示

### 8.1 修改代码前的必读事项

1. **确认目标应用层**：`ai_app1`、`ai_app2`、`ai_app3` 是三个独立运行时，修改其中一个通常不影响其他两个。
2. **框架层修改需谨慎**：`rag_framework` 被三个应用层 + `domains/android` 共同依赖，接口变更会连锁影响。
3. **领域插件去副作用**：不要在 `domains/android/android_domain/plugin.py` 模块底部执行 `register_domain()`，应在目标应用的 `lifespan` 中显式注册。
4. **工厂注册表**：新增组件实现后，务必在模块底部调用对应的 `register_xxx()`，否则 `RAGContainer.from_settings()` 无法发现该实现。

### 8.2 调试与观测

- **日志**：使用 `rag_framework.core.logger` 提供的命名 logger（`chat_logger`、`retrieval_logger` 等）。
- **检索 Trace**：`HybridRetriever.retrieve()` 已内嵌 `RetrievalTrace` 记录，调用 `get_recent_traces()` 可查看最近 1000 条检索链路。
- **延迟拆解**：`PhaseTimer` 已集成在检索流程中，`aggregate_phase_latencies()` 可生成瓶颈报告。
- **失败样本**：`FailureCollector` 自动收集检索未命中、低置信度、rerank loss 等场景，输出到 `reports/failure_cases.jsonl`。

### 8.3 常见陷阱

| 陷阱 | 说明 |
|------|------|
| 忘记构建索引 | 新克隆的仓库 `ai_app1/data/chroma_db/` 为空，必须先运行 `init_vector_db_v2.py` |
| 未设置 API Key | `.env` 中 `OPENAI_API_KEY` 或 `RAG_LLM_API_KEY` 未填会导致 LLM 调用失败 |
| 模型路径不存在 | 本地模式依赖 `models/` 下的权重，需先运行 download 脚本或手动放置 |
| 并发预热遗漏 | 新增 `Warmupable` 组件后，确保 `RAGContainer.warmup_targets()` 能发现它 |
| 自注册遗漏 | 新增 embedder / retriever / llm 实现后，确保模块被 `__init__.py` 或 `main.py` 导入以触发 `register_xxx()` |

---

## 9. 文档索引

| 文档 | 内容 | 可信度 |
|------|------|--------|
| `doc/README.md` | 项目总览（含旧版 investment_analyzer 架构描述） | 部分目录结构已过期，以实际代码为准 |
| `doc/01-ai-app1-rag-design.md` | ai_app1 完整架构、检索管道、会话管理、评测体系 | 高（v3.3，2026-05-15 更新） |
| `doc/02-ai-app2-rag-langgraph-design.md` | ai_app2 LangGraph 状态图设计、节点说明、与 ai_app1 对比 | 高（v2.0） |
| `doc/03-ai-app3-rag-agentic-design.md` | ai_app3 Agentic RAG、Self-RAG、知识图谱、查询分解 | 高（v2.0） |

---

*本文件基于项目实际代码与配置生成。若项目结构或构建流程发生重大变化，请同步更新本文件。*
