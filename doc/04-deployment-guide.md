# 部署配置与开发指南

> 版本: 1.0 | 最后更新: 2026-05-08

---

## 1. 环境要求

| 组件 | 最低版本 | 说明 |
|------|----------|------|
| Python | 3.13 | 项目使用类型提示等 3.13 特性 |
| uv | 最新版 | 包管理与虚拟环境 |
| macOS / Linux | — | 开发环境为 macOS |

---

## 2. 项目初始化

```bash
# 1. 进入项目目录
cd AnalyzerGP

# 2. 创建虚拟环境并安装依赖
uv sync

# 3. 验证安装
uv run python --version  # 应 >= 3.13
```

---

## 3. ai_app1 部署

### 3.1 配置环境变量

创建 `ai_app1/.env`：

```bash
OPENAI_API_KEY=your_minimax_api_key_here
```

> MiniMax API Key 获取: https://www.minimaxi.com/

### 3.2 构建离线索引

**首次部署必须执行**，将 Android 开发文档构建为可检索的向量索引：

```bash
# Phase 1: 构建 Parent-Child + HyDE 索引
uv run python -m ai_app1.pre.init_vector_db_v2
```

预期输出：
```
源文件加载: 15234 字符
Parent chunks: 12 个
Child chunks: 45 个
HyDE 问题总计: 36 个
Phase 1 索引构建完成
```

### 3.3 验证索引

```bash
# 验证索引完整性
uv run python -m ai_app1.pre.verify_phase1

# 验证检索质量
uv run python -m ai_app1.pre.verify_phase2

# 端到端验证
uv run python -m ai_app1.pre.verify_phase3
```

### 3.4 启动服务

```bash
# 开发模式
uv run python -m ai_app1.main

# 或使用 uvicorn 直接启动
uv run uvicorn ai_app1.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后访问: http://localhost:8000/

### 3.5 API 测试

```bash
# 基础健康检查
curl http://localhost:8000/
# → {"msg": "AI Service Running"}

# 聊天请求
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "RecyclerView 的 ViewHolder 为什么会出现空指针？"}'
```

---

## 4. investment_analyzer 部署

### 4.1 配置环境变量

在系统环境变量或 `.zshrc` 中设置：

```bash
# 推荐使用 DeepSeek（便宜且中文好）
export LLM_MODEL="deepseek/deepseek-chat"
export LLM_API_KEY="sk-your-deepseek-key"

# 备选
# export LLM_MODEL="anthropic/claude-sonnet-4-20250514"
# export LLM_API_KEY="sk-your-anthropic-key"
```

### 4.2 安装额外依赖

```bash
cd investment_analyzer

# akshare (A股数据)
pip install akshare

# yfinance (美股/港股)
pip install yfinance

# litellm (LLM 统一接口)
pip install litellm

# matplotlib (图表生成)
pip install matplotlib
```

> 这些依赖也可通过根目录 `pyproject.toml` 统一管理。

### 4.3 运行分析

```bash
# 单只A股分析
cd investment_analyzer
python main.py 600519

# 分析美股
python main.py AAPL

# 快速分析（仅量化，不调用AI）
python main.py 600519 --depth quick

# 生成图表 + 回测
python main.py 600519 --charts --backtest

# 批量分析
python main.py --batch 600519,000001,AAPL
```

### 4.4 使用监控工具

```bash
# 添加观察标的
python monitor.py --add 600519 --note "待确认反转"

# 查看观察名单
python monitor.py --list

# 检查所有观察标的的信号
python monitor.py

# 从观察名单移除
python monitor.py --remove 600519
```

---

## 5. 配置调整

### 5.1 ai_app1 可调参数

| 参数 | 文件 | 说明 |
|------|------|------|
| `CHROMA_DB_PATH` | `core/config.py` | 向量库路径（建议改为相对路径或环境变量） |
| `MAX_HISTORY` | `service/session.py` | 会话保留消息数 |
| `DEFAULT_TOKEN_BUDGET` | `service/session.py` | token 上限 |
| `RRF_K` | `service/vector_store.py` | RRF 平滑常数 |
| `RERANK_TOP_K` | `service/vector_store.py` | 最终返回片段数 |

### 5.2 investment_analyzer 可调参数

| 参数 | 文件 | 说明 |
|------|------|------|
| `min_drop_from_high` | `config.py:ANALYSIS_CONFIG` | 超跌跌幅阈值 (默认50%) |
| `reversal_pass_threshold` | `config.py:ANALYSIS_CONFIG` | 真反转通过阈值 (默认4/7) |
| `moat_min_score` | `config.py:ANALYSIS_CONFIG` | 护城河通过阈值 |
| `alpha_max_single` | `config.py:POSITION_CONFIG` | 单标的上限 (默认10%) |
| `cache_ttl_hours` | `config.py:DATA_CONFIG` | 缓存过期时间 (默认4小时) |

---

## 6. 开发指南

### 6.1 添加新工具到 ai_app1

在 `service/multiply.py` 中注册：

```python
aiTools.append({
    "type": "function",
    "function": {
        "name": "new_tool",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": { ... },
            "required": ["..."]
        }
    }
})

TOOL_FUNCTIONS["new_tool"] = new_tool_function
```

### 6.2 添加新分析器到 investment_analyzer

1. 在 `analyzers/` 下创建新分析器类
2. 在 `main.py` 的 `run_quant_analysis()` 中调用
3. 在 `config.py` 中添加阈值配置
4. 在 `report/generator.py` 中添加报告输出

### 6.3 添加新数据源

1. 在 `data/` 下创建数据获取类
2. 在 `market_router.py` 中注册新的市场识别规则
3. 在 `main.py` 的 `fetch_data()` 中添加路由分支

---

## 7. 常见问题

### Q: ai_app1 检索返回空结果？

- 检查 `CHROMA_DB_PATH` 是否正确指向已构建索引的目录
- 运行 `verify_phase1.py` 确认 collections 存在
- 检查源文档文件是否存在且非空

### Q: investment_analyzer A股数据获取失败？

- 确认 `akshare` 已安装: `pip list | grep akshare`
- 检查网络连接（akshare 依赖东方财富等国内数据源）
- 查看 `output/.cache/` 是否有旧缓存导致返回空数据

### Q: AI 分析未触发？

- 检查环境变量 `LLM_API_KEY` 和 `LLM_MODEL` 是否设置
- 检查 `litellm` 是否已安装
- 使用 `--depth full` 确保不是 quick 模式

### Q: 图表未生成？

- 确认 `matplotlib` 已安装
- 检查 `output/` 目录是否有写权限
- 查看控制台是否有 `matplotlib 未安装` 提示

---

## 8. 目录权限

```bash
# 确保输出目录可写
mkdir -p investment_analyzer/output/.cache
chmod -R 755 investment_analyzer/output

# ai_app1 的 ChromaDB 目录
mkdir -p /path/to/chroma_db
chmod -R 755 /path/to/chroma_db
```

---

## 9. 版本升级

### 更新依赖

```bash
# 更新所有依赖
uv sync --upgrade

# 更新特定包
uv add --upgrade package-name
```

### 重新构建索引（ai_app1）

```bash
# 删除旧索引后重建
rm -rf /path/to/chroma_db/*
uv run python -m ai_app1.pre.init_vector_db_v2
```
