"""
DataSource 抽象基类 — 外部实时数据源统一接口。

将金融 API、网络搜索、外部知识库等异构数据源收敛为统一抽象，
返回与本地 RAG 完全一致的 RetrievedDoc / RetrievalResult 格式，
确保 ThreeTrackRetriever 可以无差别地融合所有来源的结果。

设计原则：
  1. 零异常传播：fetch() 内建 try/except，失败返回空列表，错误写入 SourceResult
  2. 超时降级：所有网络 IO 设置硬超时，超时后静默降级
  3. 成本可控：每个 DataSource 声明 ttl_seconds + 调用配额，便于上层限流
  4. 来源追溯：每个 RetrievedDoc.metadata 必须包含 source_name 和 fetch_timestamp
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rag_framework.domain.base import QueryRoute
from rag_framework.retrieval.base import RetrievedDoc


class DataSourceType(str, Enum):
    """数据源产出内容的类型。"""

    STRUCTURED = "structured"     # 结构化数据：股价、PE、CPI 等
    UNSTRUCTURED = "unstructured" # 非结构化数据：新闻、政策解读、研报摘要
    HYBRID = "hybrid"             # 混合：同时包含结构化字段和文本描述


@dataclass(frozen=True, slots=True)
class FetchContext:
    """
    fetch() 调用时的上下文信息。

    包含当前 WealthState 中可能帮助 DataSource 更精准获取数据的字段。
    DataSource 按需取用，不强制要求所有字段都存在。
    """

    user_message: str = ""
    history: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    # 已识别的意图类型，帮助 DataSource 判断需要获取什么数据
    intent: str = ""
    # 租户标识，用于多租户场景下的数据隔离（预留）
    tenant_id: str = ""
    # 当前检索迭代次数，用于 reflection 循环中的策略调整
    retrieval_iteration: int = 0

    def get_entities_by_type(self, entity_type: str) -> list[str]:
        """按类型提取 NER 实体值。"""
        return [
            e["value"]
            for e in self.entities
            if e.get("type") == entity_type and "value" in e
        ]


@dataclass
class SourceResult:
    """
    DataSource.fetch() 的返回包装。

    显式区分成功与失败，携带诊断信息，便于 ThreeTrackRetriever 做融合决策。
    """

    docs: list[RetrievedDoc] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    # 本次 fetch 的数据源名称（冗余，便于日志和 trace）
    source_name: str = ""
    # 实际 fetch 耗时（秒）
    latency_ms: float = 0.0
    # 元数据：如 API 调用次数、缓存命中状态等
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls, source_name: str = "", error: str | None = None) -> "SourceResult":
        """构造一个空结果（用于降级或缓存命中但无数据）。"""
        return cls(
            docs=[],
            success=error is None,
            error=error,
            source_name=source_name,
        )


class DataSource(ABC):
    """
    外部数据源抽象基类。

    所有金融 API、网络搜索、外部知识库必须实现此接口。
    实例由 ThreeTrackRetriever 持有，在检索阶段并发调用。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        数据源唯一标识。

        用于来源标注、成本统计、trace 记录。例如：
          - "yahoo_finance"
          - "fred_api"
          - "tavily_search"
        """
        ...

    @property
    @abstractmethod
    def data_type(self) -> DataSourceType:
        """该数据源产出的内容类型（结构化 / 非结构化 / 混合）。"""
        ...

    @property
    @abstractmethod
    def ttl_seconds(self) -> int:
        """
        数据缓存 TTL（秒）。

        -1 表示不缓存（如网络搜索，每次必须实时）。
        0 表示仅在同一次请求内缓存（默认）。
        >0 表示按秒级缓存（如股价缓存 300 秒）。
        """
        ...

    @property
    def default_timeout(self) -> float:
        """
        单次 fetch 的网络超时（秒）。

        子类可覆盖。商业级别下建议不超过 3 秒，避免拖慢整体响应。
        """
        return 3.0

    @property
    def enabled(self) -> bool:
        """
        数据源是否启用。

        子类可根据环境变量或配额状态动态控制。
        默认始终启用。
        """
        return True

    @abstractmethod
    async def fetch(
        self,
        query: QueryRoute,
        context: FetchContext,
    ) -> SourceResult:
        """
        从外部数据源获取实时数据。

        核心约束：
          1. 所有网络 IO 必须在内部设置超时（建议使用 asyncio.wait_for）
          2. 任何异常必须被捕获，返回空 SourceResult（success=False）
          3. 返回的 RetrievedDoc 必须设置 source 和 metadata 中的 source_name / timestamp
          4. 如果查询显然不需要此数据源（如问历史知识不需要实时股价），
             应尽早返回空列表，避免浪费 API 调用

        Args:
            query: 当前检索的查询（含 text / type / weight / routes）
            context: 当前对话上下文（含实体、历史、意图等）

        Returns:
            SourceResult（docs 为空列表表示无命中或降级）
        """
        ...

    def should_fetch(self, query: QueryRoute, context: FetchContext) -> bool:
        """
        前置判断：当前查询是否需要调用此数据源。

        子类可覆盖，实现更精细的成本控制。
        默认：只要 enabled 为 True 就尝试 fetch。
        """
        return self.enabled

    def _make_doc(
        self,
        doc_id: str,
        text: str,
        score: float,
        extra_metadata: dict[str, Any] | None = None,
    ) -> RetrievedDoc:
        """
        快捷方法：构造一个标准化的 RetrievedDoc，自动注入来源追溯信息。

        子类应优先使用此方法，确保 metadata 格式统一。
        """
        import time

        meta: dict[str, Any] = {
            "source_name": self.name,
            "data_type": self.data_type.value,
            "fetch_timestamp": time.time(),
            "ttl_seconds": self.ttl_seconds,
        }
        if extra_metadata:
            meta.update(extra_metadata)
        return RetrievedDoc(
            id=doc_id,
            text=text,
            score=score,
            source=self.name,
            metadata=meta,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} type={self.data_type.value}>"
