"""
ThreeTrackRetriever — 三轨融合检索器。

将本地 RAG（Track A）、金融 API（Track B）、网络搜索（Track C）
统一编排为单一路由器，输出带来源标注的融合检索结果。

核心设计：
  1. 分级触发：根据 time_sensitive 和 entities 决定是否启用 Track B/C
  2. 并发执行：asyncio.gather 并行调用所有启用的 track
  3. 权重融合：按 track 权重加权排序，截取 top_k
  4. 来源标注：每个 RetrievedDoc 携带 source_name，用于生成阶段的溯源
  5. 成本控制：配额检查 + 超时降级，绝不因外部 API 失败而中断主流程
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_framework.datasource.base import DataSource, FetchContext, SourceResult
from rag_framework.domain.base import QueryRoute
from rag_framework.retrieval.base import Retriever, RetrievedDoc, RetrievalResult

from ai_app4.core.config import WealthSettings
from ai_app4.service.quota import QuotaManager


class ThreeTrackRetriever(Retriever):
    """
    三轨融合检索器。

    Track A（本地 RAG）：始终启用，承载历史知识和结构化分析
    Track B（金融 API）：当查询包含 ticker 或宏观指标时启用
    Track C（网络搜索）：当查询标记为 time_sensitive 时启用
    """

    def __init__(
        self,
        settings: WealthSettings,
        local_retriever: Retriever,
        data_sources: list[DataSource],
        quota_manager: QuotaManager | None = None,
    ) -> None:
        """
        Args:
            settings: WealthSettings（含 track 权重配置）
            local_retriever: Track A — 本地 HybridRetriever
            data_sources: Track B + Track C — 外部数据源列表
            quota_manager: 配额管理器（可选，None 时不做配额检查）
        """
        self._settings = settings
        self._local = local_retriever
        self._sources = {s.name: s for s in data_sources}
        self._quota = quota_manager

        # 权重映射
        self._weights: dict[str, float] = {
            "local": settings.track_a_weight,
            "yahoo_finance": settings.track_b_weight,
            "fred_api": settings.track_b_weight,
            "tavily_search": settings.track_c_weight,
        }

    # ── Retriever 接口实现 ─────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str | QueryRoute | list[QueryRoute],
        top_k: int = 10,
    ) -> RetrievalResult:
        """
        执行三轨融合检索。

        当 query 为 QueryRoute 或 list[QueryRoute] 时，提取上下文信息
        用于决定启用哪些 track。

        Args:
            query: 原始查询或 QueryRoute（支持多路扩写）
            top_k: 返回文档数

        Returns:
            RetrievalResult（docs 已按融合权重排序）
        """
        t0 = time.perf_counter()

        # 归一化输入并提取上下文
        routes, fetch_context = self._normalize_query(query)
        if not routes:
            return RetrievalResult(docs=[], latency_ms=0.0)

        primary_route = routes[0]

        # 1. Track A：本地 RAG（始终执行）
        local_task = self._fetch_local(routes, top_k)
        tasks: list[asyncio.Task] = [asyncio.create_task(local_task)]
        task_names = ["local"]

        # 2. Track B / C：外部数据源（条件触发）
        for name, source in self._sources.items():
            if not source.should_fetch(primary_route, fetch_context):
                continue

            # 配额检查
            if self._quota is not None:
                allowed, quota_info = await self._quota.check_and_consume(
                    tenant_id=fetch_context.tenant_id or "default",
                    user_id=fetch_context.user_message[:32] or "default",  # fallback
                    source_name=name,
                    requested_results=source.ttl_seconds,  # 占位
                )
                if not allowed:
                    continue  # 配额不足，跳过此数据源

            task = asyncio.create_task(
                self._fetch_source(source, primary_route, fetch_context)
            )
            tasks.append(task)
            task_names.append(name)

        # 并发执行所有 track
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 融合结果
        all_docs: list[RetrievedDoc] = []
        source_metadata: dict[str, Any] = {
            "tracks_enabled": task_names,
            "tracks_succeeded": [],
            "tracks_failed": [],
        }

        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                source_metadata["tracks_failed"].append({"name": name, "error": str(result)})
                continue

            source_metadata["tracks_succeeded"].append(name)

            weight = self._weights.get(name, 1.0)
            for doc in result:
                # 应用 track 权重
                doc.score = round(doc.score * weight, 4)
                all_docs.append(doc)

        # 按分数降序排列，截取 top_k
        all_docs.sort(key=lambda d: d.score, reverse=True)
        final_docs = all_docs[:top_k]

        latency_ms = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            docs=final_docs,
            query=primary_route.text,
            latency_ms=latency_ms,
            metadata=source_metadata,
        )

    # ── 内部方法 ───────────────────────────────────────────────────────────

    def _normalize_query(
        self, query: str | QueryRoute | list[QueryRoute]
    ) -> tuple[list[QueryRoute], FetchContext]:
        """归一化查询输入，提取 QueryRoute 列表和 FetchContext。"""
        if isinstance(query, str):
            routes = [QueryRoute(text=query)]
        elif isinstance(query, QueryRoute):
            routes = [query]
        else:
            routes = [q for q in query if q.text.strip()]

        # 从 QueryRoute 的 metadata 中尝试提取上下文（预留扩展）
        context = FetchContext(
            user_message=routes[0].text if routes else "",
        )
        return routes, context

    async def _fetch_local(
        self, routes: list[QueryRoute], top_k: int
    ) -> list[RetrievedDoc]:
        """执行本地 RAG 检索。"""
        try:
            result = await self._local.retrieve(routes, top_k=top_k * 2)  # 多取一些供融合
            # 标记来源为 local
            for doc in result.docs:
                doc.source = "local"
                doc.metadata.setdefault("source_name", "local")
            return result.docs
        except Exception:
            return []

    async def _fetch_source(
        self,
        source: DataSource,
        query: QueryRoute,
        context: FetchContext,
    ) -> list[RetrievedDoc]:
        """
        执行单个外部数据源的 fetch。

        内建超时和异常捕获，失败返回空列表。
        """
        try:
            result: SourceResult = await asyncio.wait_for(
                source.fetch(query, context),
                timeout=source.default_timeout + 0.5,  # 额外 0.5s buffer
            )
            return result.docs
        except asyncio.TimeoutError:
            return []
        except Exception:
            return []
