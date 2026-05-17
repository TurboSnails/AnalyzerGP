# 项目目录结构规范

> 本文档面向所有团队成员，目的是消除因克隆代码后模型/索引路径不一致导致的问题。
> **原则是：代码默认路径 = 事实目录，不依赖任何环境变量即可跑通。**

---

## 1. 仓库根目录结构（提交到 Git 的部分）

```
 fenxiCB/
 ├── ai_app1/           # 传统 RAG 应用（端口 8000）
 │   ├── data/          # ⭐ 源知识文档 + ChromaDB + BM25（见下文）
 │   ├── scripts/       # 模型下载脚本
 │   ├── api/           # FastAPI 路由
 │   └── ...
 ├── ai_app2/           # LangGraph 版（端口 8001）
 ├── ai_app3/           # Agentic RAG 版（端口 8002）
 ├── ai_app4/           # 商业化客服系统（端口 8004）
 ├── rag_framework/     # 通用 RAG 框架包（editable install）
 ├── domains/           # 领域插件包
 │   ├── android/       # Android 领域
 │   └── msmarco/       # MSMarco 领域（如已添加）
 ├── models/            # ⭐ 本地 AI 模型权重（gitignored，运行时下载）
 ├── scripts/           # 重构验证等公共脚本
 ├── reports/           # 评测报告（gitignored，运行时生成）
 ├── pyproject.toml     # 根项目依赖
 └── uv.lock            # 锁定文件
```

---

## 2. 大文件 / 运行时数据位置（不提交到 Git）

### 2.1 本地模型权重 → `models/`

| 模型 | 默认路径 | 下载脚本 | 说明 |
|------|---------|---------|------|
| Embedding（BGE-M3 或 bge-base-zh） | `models/bge-m3/` | `uv run python ai_app1/scripts/download_bge_m3.py` | 首次运行必须下载 |
| Reranker（bge-reranker-base） | `models/bge-reranker-base/` | 同上，脚本支持 `--preset bge-reranker-base` | 首次运行必须下载 |
| 本地 LLM / Query Rewriter（Qwen2.5-1.5B） | `models/qwen2.5-1.5b-instruct/` | `uv run python ai_app1/scripts/download_qwen_rewriter.py` | 本地模式必须下载 |
| PyTorch 任务模型缓存 | `models/torch_cache/` | 运行时自动下载 | 可选 |

> `.gitignore` 已全局忽略 `/models/`，无需担心误提交。

### 2.2 向量数据库（ChromaDB）→ `ai_app1/data/chroma_db/`

**这是唯一的数据库目录，所有应用共享。**

- 默认路径由 `RAGSettings.chroma_db_path` 控制，**硬编码默认值为 `ai_app1/data/chroma_db`**
- `ai_app2`、`ai_app3`、`ai_app4` 均复用此路径，无需各自维护一份
- 构建命令（在仓库根执行）：
  ```bash
  # Android 领域（三层索引：parent / child / hyde）
  uv run python -m domains.android.scripts.init_vector_db_v2

  # 如需重置（清空后重建）
  uv run python -m domains.android.scripts.init_vector_db_v2 --reset
  ```

### 2.3 稀疏索引（Tantivy BM25）→ `ai_app1/data/tantivy_bm25/`

- 默认路径由 `RAGSettings.bm25_index_dir` 控制，**硬编码默认值为 `ai_app1/data/tantivy_bm25`**
- 与 ChromaDB 共用同一目录父级 `ai_app1/data/`，由 `init_vector_db_v2.py` 统一构建
- 多领域共用同一个 BM25 索引，通过 `domain` metadata 字段区分

### 2.4 LlamaIndex 持久化 → `ai_app1/data/llamaindex/`

- 仅当启用 `RAG_LLAMAINDEX_ENABLED=true` 时使用
- 默认路径：`ai_app1/data/llamaindex`

### 2.5 评测报告 → `reports/`

- 综合评测、失败样本等报告默认输出到仓库根 `reports/`
- 已在 `.gitignore` 中忽略

---

## 3. 路径一致性保证

### 3.1 默认值统一

所有路径默认值已统一到 `rag_framework/rag_framework/core/config.py`：

```python
def _default_chroma_path() -> str:
    return str(_get_repo_root() / "ai_app1" / "data" / "chroma_db")

def _default_bm25_path() -> str:
    return str(_get_repo_root() / "ai_app1" / "data" / "tantivy_bm25")

def _default_llamaindex_path() -> str:
    return str(_get_repo_root() / "ai_app1" / "data" / "llamaindex")
```

其中 `_get_repo_root()` 会自动查找包含 `pyproject.toml` 的最顶层目录（即仓库根）。

### 3.2 环境变量覆盖（仅高级用户需要）

| 环境变量 | 作用 | 99% 场景不需要修改 |
|---------|------|-------------------|
| `RAG_CHROMA_DB_PATH` | 覆盖 ChromaDB 路径 | 保持默认即可 |
| `RAG_BM25_INDEX_DIR` | 覆盖 BM25 索引路径 | 保持默认即可 |
| `RAG_EMBED_MODEL_PATH` | 覆盖 Embedding 模型路径 | 保持默认即可 |
| `RAG_RERANKER_MODEL_PATH` | 覆盖 Reranker 路径 | 保持默认即可 |
| `RAG_LLM_LOCAL_MODEL_PATH` | 覆盖本地 LLM 路径 | 保持默认即可 |

### 3.3 各 app 的 .env 处理

- `ai_app1/.env` 是**主配置源**
- `ai_app2/3/4` 的 `.env` 可选，未设置时会自动回退到 `ai_app1/.env`
- 所有路径配置都在 `rag_framework` 层统一解析，各 app 不复写路径逻辑

---

## 4. 新同事上手检查清单

克隆仓库后，按以下顺序操作：

```bash
# 1. 环境初始化（安装 uv、同步依赖、复制 .env 模板）
./setup.sh

# 2. 填入 API Key
#    编辑 ai_app1/.env，把 OPENAI_API_KEY=your_minimax_key 替换为真实 Key

# 3. 下载模型（首次必须，约 3-5 GB）
uv run python -m ai_app1.scripts.download_bge_m3          # Embedding
uv run python -m ai_app1.scripts.download_qwen_rewriter   # 本地 Query Rewriter

# 4. 构建索引（必须，否则检索无结果）
uv run python -m domains.android.scripts.init_vector_db_v2

# 5. 验证安装
uv run python scripts/verify_refactor.py

# 6. 启动应用
uv run python -m uvicorn ai_app1.main:app --host 0.0.0.0 --port 8000
```

完成后目录结构应如下（仅显示大文件位置）：

```
 fenxiCB/
 ├── ai_app1/data/
 │   ├── Android 开发核心注意事项与避坑指南   # 源文档（示例）
 │   ├── chroma_db/                          # 向量数据库
 │   └── tantivy_bm25/                       # BM25 索引
 ├── models/
 │   ├── bge-m3/                             # Embedding 模型
 │   ├── bge-reranker-base/                  # Reranker 模型
 │   └── qwen2.5-1.5b-instruct/              # 本地 LLM 模型
 └── ...
```

---

## 5. 常见错误

| 现象 | 原因 | 解决 |
|------|------|------|
| `chroma_db_path` 指向 `data/chroma_db` 但找不到文件 | 使用了旧代码默认值 | 拉取最新代码，确认 `_default_chroma_path()` 返回 `ai_app1/data/chroma_db` |
| BM25 索引为空 / 检索不到结果 | 未运行 `init_vector_db_v2` | 执行 `uv run python -m domains.android.scripts.init_vector_db_v2` |
| 模型加载失败 / 路径不存在 | `models/` 下缺少权重 | 运行对应 download 脚本 |
| 不同同事环境表现不一致 | `.env` 中手动覆盖了路径 | 删除 `.env` 中的 `RAG_CHROMA_DB_PATH` / `RAG_BM25_INDEX_DIR`，使用默认值 |

---

*最后更新：2026-05-17*  
*维护者：团队全体，修改路径默认值时必须同步更新本文档*
