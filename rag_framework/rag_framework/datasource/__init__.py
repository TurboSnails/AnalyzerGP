"""
Datasource — 外部数据源接入层。

为本地 RAG 之外的实时数据提供统一抽象：
  - 金融 API（Yahoo Finance、FRED、Alpha Vantage）
  - 网络搜索（Tavily、SerpAPI、Bing）
  - 其他外部知识源

核心设计：
  - DataSource 抽象基类统一返回 RetrievedDoc 格式
  - 所有 fetch() 操作内建超时、限流、降级，绝不抛异常中断主流程
  - SourceResult 包装 fetch 结果，显式携带 success / error 状态
"""

from rag_framework.datasource.base import (
    DataSource,
    DataSourceType,
    FetchContext,
    SourceResult,
)

__all__ = [
    "DataSource",
    "DataSourceType",
    "FetchContext",
    "SourceResult",
]
