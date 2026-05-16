# ai_app1 架构文档

> 多领域 RAG（检索增强生成）聊天应用

## 概述

ai_app1 是一个基于 `rag_framework` 构建的**薄应用层**，核心功能是：
**用户发问 → 检索相关文档 → 把文档塞给 AI → AI 流式回答**

支持多个知识领域（MS MARCO、Android）同时运行，共享底层模型资源，各自独立维护检索逻辑与会话历史。

---

## 整体分层结构

```
┌─────────────────────────────────────────┐
│           ai_app1  (应用层)              │
│   main.py  ──  api/chat.py              │
│   "薄包装层，只负责组装和路由"            │
└──────────────────┬──────────────────────┘
                   │ 调用
┌──────────────────▼──────────────────────┐
│        rag_framework  (框架层)           │
│  container / session / retrieval / llm  │
│  "所有重型逻辑都在这里"                  │
└──────────────────┬──────────────────────┘
                   │ 调用
┌──────────────────▼──────────────────────┐
│           领域插件  (domain层)           │
│  MSMarcoDomainPlugin / AndroidDomainPlugin│
│  "告诉框架：这个领域用什么提示词、怎么分类查询"│
└─────────────────────────────────────────┘
```

---

## 启动流程（main.py `lifespan`）

服务器启动时按顺序执行 6 个步骤：

```
1. 注册领域插件         → MSMarco、Android 都注册进去
         ↓
2. 创建「基础容器」      → 加载共享的重型组件：
                           - Embedder（把文字变成向量的模型）
                           - VectorStore（ChromaDB，存向量）
                           - LLM（大语言模型客户端）
                           - Reranker（重排序模型）
         ↓
3. 创建共享 BM25Store   → 关键词索引（tantivy），所有领域共用一份
         ↓
4. 为每个领域创建「派生容器」→ 共享重型组件，但各自独立的：
                              - session_store（会话历史）
                              - retriever（检索器，带领域过滤）
         ↓
5. 并发预热所有组件      → 让模型提前加载进内存，不在第一个请求时卡顿
         ↓
6. 等待请求… 关闭时释放资源
```

---

## 一次对话的请求流程（api/chat.py）

```
POST /chat  { message: "...", user_id: "...", domain: "" }
       │
       ▼
  1. 判断领域路由
     - domain="all"   → 多领域融合模式
     - 含中文         → 自动选 android 领域
     - 其他           → 默认领域（msmarco）
       │
       ▼
  2. 单领域模式：container.chat_stream(query, user_id)
     ┌─────────────────────────────────────────┐
     │  SessionManager.chat_stream()            │
     │  a. 取出会话历史                          │
     │  b. Query Rewriter 改写查询              │
     │     (rule_rewriter: 术语替换/扩展)        │
     │  c. DomainPlugin.classify_query()        │
     │     → 决定用哪些检索路径                  │
     │       [dense, bm25, hyde]               │
     │  d. Retriever.retrieve()                 │
     │     → 向量检索 + BM25 检索 + 重排序       │
     │  e. 组装 messages:                       │
     │     [system_prompt + 历史摘要 + 检索到的文档 + 用户问题] │
     │  f. LLM.chat_stream() 流式生成回答        │
     │  g. 更新会话历史                          │
     └─────────────────────────────────────────┘
       │
       ▼
  3. 返回 StreamingResponse（文字一块一块流出来）
```

---

## 多领域融合模式（domain="all"）

当用户指定 `domain="all"` 时走特殊路径：

```
并行检索所有领域（msmarco + android）
        ↓
合并所有文档，去重，按分数排序，取 top 6
        ↓
统一交给默认领域的 LLM 生成回答
```

---

## 核心概念对照

| 概念 | 通俗解释 |
|------|---------|
| **RAGContainer** | 所有零件的「工具箱」，每个领域一个 |
| **DomainPlugin** | 领域的「说明书」——用什么提示词、怎么理解查询 |
| **Embedder** | 把文字变成数字向量（bge-m3 模型） |
| **VectorStore** | 向量数据库（ChromaDB），存所有知识片段的向量 |
| **BM25Store** | 关键词索引（传统搜索），与向量检索互补 |
| **Reranker** | 检索到很多候选文档后，再用一个模型排序，选最相关的 |
| **SessionStore** | 每个用户的对话历史，放在内存里 |
| **QueryRewriter** | 改写用户的问题，让检索更准（比如中文术语翻译成英文） |
| **StreamingResponse** | 像 ChatGPT 一样，回答一个字一个字地流出来 |

---

## 文件职责一览

| 文件 | 职责 |
|------|------|
| `main.py` | 启动、组装容器、注册路由 |
| `api/chat.py` | HTTP 接口、领域路由判断 |
| `rag_framework/container.py` | 工具箱定义，`chat_stream` 入口 |
| `rag_framework/session/manager.py` | 完整对话逻辑（rewrite→retrieve→generate） |
| `rag_framework/retrieval/fusion.py` | 混合检索（dense + bm25 + rerank 融合） |
| `rag_framework/domain/base.py` | 领域插件抽象接口 |

---

## 新增领域的方法

只需两步：

1. 在 `domains/` 目录下创建新的 `xxx_domain` 包，实现 `DomainPlugin` 子类
2. 在 `main.py` 的 `_DOMAIN_CLASSES` 中 import 并追加

索引数据时在 metadata 写入 `domain` 字段即可，无需修改任何 collection 名称（统一使用 `knowledge_base` collection）。
