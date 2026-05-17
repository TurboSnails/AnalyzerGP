"""
Tavily Search 数据源 — 实时网络搜索。

Tavily 是专为 AI RAG 优化的搜索引擎，返回已清洗的文本摘要，
无需额外解析网页 HTML。适用于突发新闻、政策解读、市场情绪等
本地知识库无法覆盖的时效性问题。

商业约束：
  - 成本：$0.025/次（advanced search）
  - 必须严格控制调用频率：分级触发 + 单用户日限额
  - 不缓存（搜索结果实时性要求）
  - 失败时静默降级

注册地址: https://tavily.com
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
from rag_framework.domain.base import QueryRoute


class TavilySearchSource(DataSource):
    """
    Tavily 实时搜索接入。

    支持两种搜索深度：
      - basic：快速摘要，成本低
      - advanced：深度研究，返回更详细的摘要和相关问题

    仅在查询被标记为 time_sensitive 时触发（由 classify 节点控制）。
    """

    _API_URL = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str = "",
        search_depth: str = "basic",
        max_results: int = 3,
        include_answer: bool = True,
    ) -> None:
        """
        Args:
            api_key: Tavily API Key
            search_depth: "basic" | "advanced"
            max_results: 返回结果数（控制成本，建议 3-5）
            include_answer: 是否包含 Tavily 生成的直接答案
        """
        self._api_key = api_key
        self._search_depth = search_depth
        self._max_results = max(min(max_results, 5), 1)  # 限制 1-5
        self._include_answer = include_answer
        self._client: httpx.AsyncClient | None = None

    # ── DataSource 接口实现 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "tavily_search"

    @property
    def data_type(self) -> DataSourceType:
        return DataSourceType.UNSTRUCTURED

    @property
    def ttl_seconds(self) -> int:
        return -1  # 实时搜索不缓存

    @property
    def default_timeout(self) -> float:
        return 5.0  # Tavily advanced search 可能较慢

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def should_fetch(self, query: QueryRoute, context: FetchContext) -> bool:
        """
        前置判断：仅对时效性敏感查询启用搜索。

        避免为历史知识类问题浪费搜索配额。
        """
        if not self.enabled:
            return False
        # 从 FetchContext 获取 classify 节点标记的 time_sensitive
        # 由于 context 中不直接包含 time_sensitive，我们通过意图和查询内容推断
        intent = context.intent
        if intent in ("real_time_query", "market_news", "current_event"):
            return True
        # 兜底：检测查询中是否包含时效性关键词
        text = query.text.lower()
        time_markers = (
            "今天", "今日", "现在", "当前", "最新", "实时", "即时",
            "刚才", "刚刚", "最近", "近日", "本周", "今早", "今晚",
            "昨天", "明日", "news", "today", "now", "latest", "breaking",
        )
        return any(m in text for m in time_markers)

    async def fetch(
        self,
        query: QueryRoute,
        context: FetchContext,
    ) -> SourceResult:
        if not self._api_key:
            return SourceResult.empty(self.name, error="Tavily API Key 未配置")

        start = time.monotonic()
        docs: list[Any] = []
        error: str | None = None

        # 初始化 httpx client（惰性）
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.default_timeout))

        try:
            payload = {
                "api_key": self._api_key,
                "query": query.text,
                "search_depth": self._search_depth,
                "max_results": self._max_results,
                "include_answer": self._include_answer,
                "include_raw_content": False,  # 不需要原始 HTML
            }

            response = await asyncio.wait_for(
                self._client.post(self._API_URL, json=payload),
                timeout=self.default_timeout,
            )
            response.raise_for_status()
            data = response.json()

            # Tavily 返回结构：
            # {
            #   "answer": "...",
            #   "query": "...",
            #   "results": [
            #     {"title": "...", "url": "...", "content": "...", "score": 0.95},
            #     ...
            #   ]
            # }

            # 如果 Tavily 提供了直接答案，作为第一条高置信度文档
            answer = data.get("answer", "")
            if answer and self._include_answer:
                docs.append(
                    self._make_doc(
                        doc_id=f"tavily_answer_{int(time.time())}",
                        text=f"【Tavily 综合摘要】\n{answer}",
                        score=0.95,
                        extra_metadata={
                            "result_type": "synthesized_answer",
                            "query": data.get("query", query.text),
                        },
                    )
                )

            # 逐条处理搜索结果
            for i, result in enumerate(data.get("results", [])):
                content = result.get("content", "")
                if not content or len(content) < 20:
                    continue  # 过滤空或过短的摘要

                title = result.get("title", "")
                url = result.get("url", "")
                score = result.get("score", 0.8)

                # 清洗：去除明显是广告或导航栏的内容
                cleaned = self._clean_content(content)
                if not cleaned:
                    continue

                text = f"【{title}】\n{cleaned}\n\n来源: {url}"

                docs.append(
                    self._make_doc(
                        doc_id=f"tavily_{i}_{int(time.time())}",
                        text=text,
                        score=score,
                        extra_metadata={
                            "result_type": "search_result",
                            "title": title,
                            "url": url,
                            "query": data.get("query", query.text),
                        },
                    )
                )

        except (httpx.TimeoutException, asyncio.TimeoutError):
            error = "Tavily 搜索超时"
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                error = "Tavily API 配额已用完"
            elif exc.response.status_code == 401:
                error = "Tavily API Key 无效"
            else:
                error = f"Tavily API 错误: {exc.response.status_code}"
        except Exception as exc:
            error = f"Tavily 搜索失败: {exc}"

        latency_ms = (time.monotonic() - start) * 1000
        return SourceResult(
            docs=docs,
            success=error is None,
            error=error,
            source_name=self.name,
            latency_ms=latency_ms,
            metadata={
                "search_depth": self._search_depth,
                "max_results_requested": self._max_results,
                "results_returned": len(docs),
            },
        )

    # ── 内部方法 ─────────────────────────────────────────────────────────

    @staticmethod
    def _clean_content(text: str) -> str:
        """
        清洗 Tavily 返回的摘要内容。

        去除常见的广告、导航、无关文本。
        """
        if not text:
            return ""

        # 去除常见广告/导航前缀
        noise_prefixes = (
            "Skip to content",
            "Skip to main content",
            "Sign in",
            "Subscribe",
            "Log in",
            "Home »",
            "Menu",
            "Search",
        )
        cleaned = text
        for prefix in noise_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].lstrip()

        # 去除常见广告/导航后缀
        noise_suffixes = (
            "Read more",
            "Learn more",
            "Continue reading",
            "Sign up for",
            "Subscribe to",
        )
        for suffix in noise_suffixes:
            if cleaned.endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip()

        # 去除多余的空行
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        return "\n".join(lines)
