# AnalyzerGP - 项目全局文档

> 版本: 1.0 | 最后更新: 2026-05-08

---

## 项目概述

AnalyzerGP 是一个复合型 AI 工具项目，包含两个独立的子系统：

| 子系统 | 定位 | 核心功能 |
|--------|------|----------|
| **ai_app1** | Android 开发 RAG 问答助手 | 基于 ChromaDB 混合检索 + MiniMax LLM 的 Android 开发问题问答服务 |
| **investment_analyzer** | 投资分析工作流引擎 | 多市场股票量化筛选 + AI 多 Agent 辩论 + Markdown 报告生成 |

两个子系统共享同一个 Python 虚拟环境和依赖管理（`uv` + `pyproject.toml`），但运行时完全独立，各自拥有独立的配置、数据源和业务逻辑。

---

## 项目结构

```
AnalyzerGP/
├── pyproject.toml               # 统一依赖管理 (uv)
├── uv.lock                      # 锁定文件
├── doc/                         # [本文档目录]
│   ├── README.md                # 项目总览
│   ├── 01-ai-app1-rag-design.md
│   ├── 02-investment-analyzer-design.md
│   ├── 03-data-flow-interaction.md
│   └── 04-deployment-guide.md
│
├── ai_app1/                     # Android RAG 问答服务
│   ├── main.py                  # FastAPI 入口
│   ├── .env                     # 环境变量 (API Key)
│   ├── data/
│   │   └── Android 开发核心注意事项与避坑指南   # 源知识文档
│   ├── api/
│   │   └── chat.py              # /chat POST 路由
│   ├── core/
│   │   ├── config.py            # CHROMA_DB_PATH / OPENAI_API_KEY
│   │   └── logger.py            # 统一日志配置
│   ├── pre/                     # 离线索引脚本
│   │   ├── init_vector_db.py    # 旧版单路索引 (已弃用)
│   │   ├── init_vector_db_v2.py # Parent-Child + HyDE 索引
│   │   ├── verify_phase1.py     # 索引验证
│   │   ├── verify_phase2.py     # 检索验证
│   │   └── verify_phase3.py     # 端到端验证
│   └── service/
│       ├── AiClient.py          # MiniMax-M2.7 客户端
│       ├── session.py           # 会话管理 (history/summary/trim)
│       ├── vector_store.py      # 混合检索管道 (Dense+HyDE+BM25+RRF+Rerank)
│       ├── bm25_store.py        # BM25 稀疏检索
│       ├── reranker.py          # 精排 + Lost-in-Middle 重排
│       └── multiply.py          # Tool Functions 注册
│
└── investment_analyzer/         # 投资分析工作流
    ├── main.py                  # CLI 入口
    ├── config.py                # 全量配置 (LLM / 数据源 / 分析阈值 / 报告)
    ├── monitor.py               # 观察名单监控工具
    ├── requirements.txt         # 子项目依赖
    ├── PROJECT_PLAN.md          # 原始项目计划
    ├── agents/
    │   └── base_agent.py        # BaseAgent + Bull/Bear/Judge/FirstPrinciples
    ├── analyzers/
    │   ├── drop_checker.py      # 第一关: 跌幅量化
    │   ├── reversal_checker.py  # 第二关: 真反转验证 (7维度)
    │   ├── moat_checker.py      # 第三关: 护城河验证 (6维度)
    │   ├── fundamental_screener.py  # Layer2: A/B 类基本面双轨筛选
    │   ├── macro_checker.py     # Layer0: 宏观定轨
    │   ├── position_sizer.py    # 仓位管理器
    │   └── backtester.py        # 历史回测
    ├── data/
    │   ├── market_router.py     # 市场识别 (A股/港股/美股)
    │   ├── a_share.py           # A股数据 (akshare)
    │   ├── us_share.py          # 美股/港股数据 (yfinance)
    │   └── cache.py             # 数据缓存装饰器
    ├── report/
    │   ├── generator.py         # Markdown 报告生成器
    │   └── charts.py            # 可视化图表 (matplotlib 暗色主题)
    └── output/                  # 报告输出目录
        └── .cache/              # 数据缓存目录
```

---

## 技术栈

| 组件 | 选择 | 用途 |
|------|------|------|
| Python | >=3.13 | 运行环境 |
| 包管理 | `uv` | 依赖管理 |
| Web 框架 | FastAPI | ai_app1 HTTP 服务 |
| LLM | MiniMax-M2.7 / DeepSeek / Claude | AI 推理 |
| 向量数据库 | ChromaDB | 文档检索 |
| 稀疏检索 | rank-bm25 | BM25 全文检索 |
| 数据科学 | pandas, numpy | 量化分析 |
| 可视化 | matplotlib | 图表生成 |
| A股数据 | akshare | 免费 A股/宏观数据 |
| 美股/港股 | yfinance | 全球市场数据 |
| LLM 统一接口 | litellm | 多模型切换 |

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [01-ai-app1-rag-design.md](01-ai-app1-rag-design.md) | Android RAG 问答系统的架构、检索管道、会话管理、离线索引流程 |
| [02-investment-analyzer-design.md](02-investment-analyzer-design.md) | 投资分析工作流的五层框架、分析器设计、AI Agent 架构、报告生成 |
| [03-data-flow-interaction.md](03-data-flow-interaction.md) | 两个子系统的数据流图、模块间交互关系、共享依赖说明 |
| [04-deployment-guide.md](04-deployment-guide.md) | 环境搭建、索引构建、运行方式、配置说明 |

---

## 免责声明

本项目中的 **investment_analyzer** 仅供个人投资研究参考，不构成任何专业投资建议。使用本工具进行分析时，请自行承担投资风险。