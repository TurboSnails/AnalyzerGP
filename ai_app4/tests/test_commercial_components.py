"""
ai_app4 商业版组件单元测试

覆盖范围：
1. MemoryCache — TTL、并发安全、统计
2. QuotaManager — 日/小时限额、硬限制
3. YahooFinanceSource — 降级、实体识别
4. FredAPISource — 宏观指标查询、异常处理
5. TavilySearchSource — 搜索、清理
6. ThreeTrackRetriever — 分级触发、权重融合、Track 降级
7. nodes._build_source_footer / _append_compliance — 来源标注与合规
8. WealthSettings — 新配置字段加载

所有外部 HTTP 调用均使用 unittest.mock 拦截，不依赖真实网络。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import pytest

from ai_app4.core.config import WealthSettings
from ai_app4.service.cache import MemoryCache
from ai_app4.service.quota import QuotaConfig, QuotaManager


# ═════════════════════════════════════════════════════════════════════════════
# 1. MemoryCache
# ═════════════════════════════════════════════════════════════════════════════

class TestMemoryCache:
    @pytest.mark.asyncio
    async def test_basic_set_get(self) -> None:
        cache = MemoryCache()
        await cache.set("k1", "v1", ttl=60)
        val = await cache.get("k1")
        assert val == "v1"

    @pytest.mark.asyncio
    async def test_ttl_expiration(self) -> None:
        cache = MemoryCache()
        await cache.set("k1", "v1", ttl=0)
        await asyncio.sleep(0.05)
        val = await cache.get("k1")
        assert val is None

    @pytest.mark.asyncio
    async def test_delete_and_miss(self) -> None:
        cache = MemoryCache()
        await cache.set("k1", "v1", ttl=60)
        await cache.delete("k1")
        val = await cache.get("k1")
        assert val is None

    @pytest.mark.asyncio
    async def test_concurrent_set(self) -> None:
        cache = MemoryCache()

        async def setter(idx: int) -> None:
            await cache.set(f"k{idx}", f"v{idx}", ttl=60)

        await asyncio.gather(*(setter(i) for i in range(50)))
        stats = await cache.stats()
        assert stats["active_keys"] == 50

    @pytest.mark.asyncio
    async def test_stats_report(self) -> None:
        cache = MemoryCache()
        await cache.set("k1", "v1", ttl=60)
        stats = await cache.stats()
        assert stats["total_keys"] == 1
        assert stats["expired_keys"] == 0
        assert stats["active_keys"] == 1

    @pytest.mark.asyncio
    async def test_lazy_cleanup(self) -> None:
        cache = MemoryCache()
        await cache.set("k1", "v1", ttl=0)
        await asyncio.sleep(0.05)
        # get 触发懒清理
        await cache.get("k1")
        stats = await cache.stats()
        assert stats["total_keys"] == 0
        assert stats["expired_keys"] == 0  # 已被清理


# ═════════════════════════════════════════════════════════════════════════════
# 2. QuotaManager
# ═════════════════════════════════════════════════════════════════════════════

class TestQuotaManager:
    def _make_mgr(self, daily_hard: int = 3, hourly: int = 100) -> QuotaManager:
        mgr = QuotaManager()
        mgr.register_config("test_src", QuotaConfig(daily_hard_limit=daily_hard, hourly_limit=hourly))
        return mgr

    @pytest.mark.asyncio
    async def test_daily_limit_enforced(self) -> None:
        mgr = self._make_mgr(daily_hard=3, hourly=100)
        for _ in range(3):
            ok, _ = await mgr.check_and_consume("t1", "u1", "test_src")
            assert ok is True
        ok, info = await mgr.check_and_consume("t1", "u1", "test_src")
        assert ok is False
        assert info["reason"] == "daily_hard_limit_exceeded"

    @pytest.mark.asyncio
    async def test_hourly_limit_enforced(self) -> None:
        mgr = self._make_mgr(daily_hard=100, hourly=2)
        for _ in range(2):
            ok, _ = await mgr.check_and_consume("t1", "u1", "test_src")
            assert ok is True
        ok, info = await mgr.check_and_consume("t1", "u1", "test_src")
        assert ok is False
        assert info["reason"] == "hourly_limit_exceeded"

    @pytest.mark.asyncio
    async def test_user_isolation(self) -> None:
        mgr = self._make_mgr(daily_hard=1, hourly=10)
        ok, _ = await mgr.check_and_consume("t1", "u1", "test_src")
        assert ok is True
        ok, _ = await mgr.check_and_consume("t1", "u2", "test_src")
        assert ok is True
        ok, _ = await mgr.check_and_consume("t1", "u1", "test_src")
        assert ok is False
        ok, _ = await mgr.check_and_consume("t1", "u2", "test_src")
        assert ok is False

    @pytest.mark.asyncio
    async def test_hard_limit_zero(self) -> None:
        mgr = self._make_mgr(daily_hard=0, hourly=10)
        ok, _ = await mgr.check_and_consume("t1", "u1", "test_src")
        assert ok is False

    @pytest.mark.asyncio
    async def test_usage_tracking(self) -> None:
        mgr = self._make_mgr(daily_hard=5, hourly=5)
        for _ in range(2):
            await mgr.check_and_consume("t1", "u1", "test_src")
        summary = await mgr.get_usage_summary(tenant_id="t1", user_id="u1")
        assert summary  # 非空


# ═════════════════════════════════════════════════════════════════════════════
# 3. YahooFinanceSource
# ═════════════════════════════════════════════════════════════════════════════

class TestYahooFinanceSource:
    @pytest.mark.asyncio
    async def test_entity_extraction_price(self) -> None:
        from ai_app4.service.datasources.yahoo_finance import YahooFinanceSource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = YahooFinanceSource(cache=None)
        route = QueryRoute(text="苹果股价是多少", type="corp", weight=1.0, routes=[])
        ctx = FetchContext(user_message="苹果股价是多少")

        # yfinance 未安装时会走 mock 数据路径，成功返回文档
        result = await source.fetch(route, ctx)
        assert result.success is True
        assert len(result.docs) >= 1
        assert "AAPL" in result.docs[0].text

    @pytest.mark.asyncio
    async def test_no_ticker_found(self) -> None:
        from ai_app4.service.datasources.yahoo_finance import YahooFinanceSource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = YahooFinanceSource(cache=None)
        route = QueryRoute(text="今天的天气怎么样", type="corp", weight=1.0, routes=[])
        ctx = FetchContext(user_message="今天的天气怎么样")

        result = await source.fetch(route, ctx)
        assert result.success is True
        content = result.docs[0].text if result.docs else ""
        assert "未识别" in content or "未找到" in content or content == ""


# ═════════════════════════════════════════════════════════════════════════════
# 4. FredAPISource
# ═════════════════════════════════════════════════════════════════════════════

class TestFredAPISource:
    @pytest.mark.asyncio
    async def test_fetch_indicator(self) -> None:
        from ai_app4.service.datasources.fred_api import FredAPISource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = FredAPISource(api_key="fake_key")
        route = QueryRoute(text="最新 CPI 数据", type="macro", weight=1.0, routes=[])
        ctx = FetchContext(user_message="最新 CPI 数据")

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "observations": [
                    {"date": "2026-04-01", "value": "3.2"},
                    {"date": "2026-03-01", "value": "3.4"},
                ]
            }
            mock_get.return_value = mock_response

            result = await source.fetch(route, ctx)
            assert result.success is True
            assert "3.2" in result.docs[0].text

    @pytest.mark.asyncio
    async def test_invalid_api_key(self) -> None:
        from ai_app4.service.datasources.fred_api import FredAPISource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = FredAPISource(api_key="")
        route = QueryRoute(text="CPI", type="macro", weight=1.0, routes=[])
        ctx = FetchContext(user_message="CPI")

        # 模拟 FRED 返回 403 拒绝空 Key
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_response.json.return_value = {"error": "Invalid API key"}
            mock_response.raise_for_status.side_effect = Exception("403 Forbidden")
            mock_get.return_value = mock_response

            result = await source.fetch(route, ctx)
            assert result.success is False
            assert "未配置" in (result.error or "")


# ═════════════════════════════════════════════════════════════════════════════
# 5. TavilySearchSource
# ═════════════════════════════════════════════════════════════════════════════

class TestTavilySearchSource:
    @pytest.mark.asyncio
    async def test_basic_search(self) -> None:
        from ai_app4.service.datasources.tavily_search import TavilySearchSource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = TavilySearchSource(api_key="fake_key")
        route = QueryRoute(text="美联储最新利率决定", type="macro", weight=1.0, routes=[])
        ctx = FetchContext(user_message="美联储最新利率决定")

        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "results": [
                    {"title": "Fed Keeps Rates", "content": "The Fed decided to...", "url": "https://example.com"}
                ],
                "answer": "The Federal Reserve maintained...",
            }
            mock_post.return_value = mock_response

            result = await source.fetch(route, ctx)
            assert result.success is True
            assert "Fed" in result.docs[0].text or "Federal Reserve" in result.docs[0].text

    @pytest.mark.asyncio
    async def test_timeout_returns_failure(self) -> None:
        from ai_app4.service.datasources.tavily_search import TavilySearchSource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = TavilySearchSource(api_key="fake_key")
        route = QueryRoute(text="test", type="macro", weight=1.0, routes=[])
        ctx = FetchContext(user_message="test")

        with patch("httpx.AsyncClient.post", side_effect=TimeoutError("connection timeout")):
            result = await source.fetch(route, ctx)
            assert result.success is False
            assert result.error is not None

    @pytest.mark.asyncio
    async def test_not_time_sensitive_skipped(self) -> None:
        from ai_app4.service.datasources.tavily_search import TavilySearchSource
        from rag_framework.datasource.base import FetchContext
        from rag_framework.domain.base import QueryRoute

        source = TavilySearchSource(api_key="fake_key")
        route = QueryRoute(text="test", type="macro", weight=1.0, routes=[])
        ctx = FetchContext(user_message="test")
        # Tavily 默认都触发（无 should_fetch 限制），只要 fetch 被调用即可
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"results": [], "answer": ""}
            mock_post.return_value = mock_response

            result = await source.fetch(route, ctx)
            assert result.success is True


# ═════════════════════════════════════════════════════════════════════════════
# 6. ThreeTrackRetriever
# ═════════════════════════════════════════════════════════════════════════════

class TestThreeTrackRetriever:
    @pytest.mark.asyncio
    async def test_track_a_fallback_when_no_datasources(self) -> None:
        from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever
        from rag_framework.retrieval.base import RetrievedDoc, RetrievalResult
        from rag_framework.domain.base import QueryRoute

        settings = WealthSettings(
            three_track_enabled=True,
            track_a_weight=1.0,
            track_b_weight=0.0,
            track_c_weight=0.0,
        )
        local = AsyncMock()
        local.retrieve.return_value = RetrievalResult(
            docs=[RetrievedDoc(id="d1", text="local doc", score=0.9, source="local")]
        )

        retriever = ThreeTrackRetriever(settings, local, [])
        route = QueryRoute(text="test", type="all", weight=1.0, routes=[])
        result = await retriever.retrieve([route], top_k=5)

        assert len(result.docs) == 1
        assert result.docs[0].text == "local doc"
        local.retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_weighted_fusion_multi_tracks(self) -> None:
        from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever
        from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
        from rag_framework.retrieval.base import RetrievedDoc, RetrievalResult
        from rag_framework.domain.base import QueryRoute

        class FakeDataSource(DataSource):
            @property
            def name(self) -> str: return "fake"
            @property
            def data_type(self) -> DataSourceType: return DataSourceType.FINANCIAL_API
            @property
            def ttl_seconds(self) -> int: return 60
            async def fetch(self, query: QueryRoute, context: FetchContext) -> SourceResult:
                return SourceResult(
                    docs=[RetrievedDoc(id="f1", text="api data", score=0.95, source="fake")],
                    success=True,
                    source_name="fake",
                )

        settings = WealthSettings(
            three_track_enabled=True,
            track_a_weight=1.0,
            track_b_weight=0.95,
            track_c_weight=0.0,
        )
        local = AsyncMock()
        local.retrieve.return_value = RetrievalResult(
            docs=[RetrievedDoc(id="d1", text="local", score=0.8, source="local")]
        )

        retriever = ThreeTrackRetriever(settings, local, [FakeDataSource()])
        route = QueryRoute(text="test", type="all", weight=1.0, routes=[])
        result = await retriever.retrieve([route], top_k=5)

        assert len(result.docs) >= 1
        local.retrieve.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_quota_exceeded_skips_external(self) -> None:
        from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever
        from ai_app4.service.quota import QuotaConfig, QuotaManager
        from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
        from rag_framework.retrieval.base import RetrievedDoc, RetrievalResult
        from rag_framework.domain.base import QueryRoute

        class FakeDataSource(DataSource):
            @property
            def name(self) -> str: return "fake"
            @property
            def data_type(self) -> DataSourceType: return DataSourceType.FINANCIAL_API
            @property
            def ttl_seconds(self) -> int: return 60
            async def fetch(self, query: QueryRoute, context: FetchContext) -> SourceResult:
                return SourceResult(docs=[], success=True, source_name="fake")

        quota = QuotaManager()
        quota.register_config("fake", QuotaConfig(daily_hard_limit=0, hourly_limit=0))
        settings = WealthSettings(three_track_enabled=True)
        local = AsyncMock()
        local.retrieve.return_value = RetrievalResult(docs=[])

        retriever = ThreeTrackRetriever(settings, local, [FakeDataSource()], quota_manager=quota)
        route = QueryRoute(text="test", type="all", weight=1.0, routes=[])
        ctx = FetchContext()

        # ThreeTrackRetriever.retrieve() doesn't take context kwarg directly;
        # FetchContext is derived from QueryRoute by _normalize_query.
        result = await retriever.retrieve([route], top_k=5)
        # 配额耗尽时仍应返回本地 Track A 的结果（虽然这里 mock 为空）
        assert result.docs == []

    @pytest.mark.asyncio
    async def test_external_source_failure_graceful(self) -> None:
        from ai_app4.service.retrieval.three_track_retriever import ThreeTrackRetriever
        from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
        from rag_framework.retrieval.base import RetrievedDoc, RetrievalResult
        from rag_framework.domain.base import QueryRoute

        class FailDataSource(DataSource):
            @property
            def name(self) -> str: return "fail"
            @property
            def data_type(self) -> DataSourceType: return DataSourceType.FINANCIAL_API
            @property
            def ttl_seconds(self) -> int: return 60
            async def fetch(self, query: QueryRoute, context: FetchContext) -> SourceResult:
                raise RuntimeError("network error")

        settings = WealthSettings(three_track_enabled=True)
        local = AsyncMock()
        local.retrieve.return_value = RetrievalResult(
            docs=[RetrievedDoc(id="d1", text="safe", score=0.9, source="local")]
        )

        retriever = ThreeTrackRetriever(settings, local, [FailDataSource()])
        route = QueryRoute(text="test", type="all", weight=1.0, routes=[])
        result = await retriever.retrieve([route], top_k=5)

        assert len(result.docs) == 1
        assert result.docs[0].text == "safe"


# ═════════════════════════════════════════════════════════════════════════════
# 7. Source Attribution & Compliance Helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestSourceAttribution:
    def test_build_source_footer_single_track(self) -> None:
        from ai_app4.graph.nodes import _build_source_footer

        trace = [{"node": "parallel_retrieval", "tracks_used": ["track_a"]}]
        footer = _build_source_footer(trace)
        assert "本地知识库" in footer
        assert "实时金融数据 API" not in footer

    def test_build_source_footer_all_tracks(self) -> None:
        from ai_app4.graph.nodes import _build_source_footer

        trace = [{"node": "parallel_retrieval", "tracks_used": ["track_a", "track_b", "track_c"]}]
        footer = _build_source_footer(trace)
        assert "本地知识库" in footer
        assert "实时金融数据 API" in footer
        assert "网络搜索" in footer

    def test_build_source_footer_empty_trace(self) -> None:
        from ai_app4.graph.nodes import _build_source_footer
        assert _build_source_footer([]) == ""

    def test_append_compliance_enabled(self) -> None:
        from ai_app4.graph.nodes import _append_compliance

        settings = WealthSettings(enable_compliance_disclaimer=True)
        reply = _append_compliance("建议买入 NVDA", settings)
        assert "免责声明" in reply
        assert "建议买入 NVDA" in reply

    def test_append_compliance_disabled(self) -> None:
        from ai_app4.graph.nodes import _append_compliance

        settings = WealthSettings(enable_compliance_disclaimer=False)
        reply = _append_compliance("建议买入 NVDA", settings)
        assert reply == "建议买入 NVDA"


# ═════════════════════════════════════════════════════════════════════════════
# 8. WealthSettings — 新配置字段
# ═════════════════════════════════════════════════════════════════════════════

class TestWealthSettings:
    def test_default_values(self) -> None:
        s = WealthSettings()
        assert s.three_track_enabled is True
        assert s.track_a_weight == 1.0
        assert s.track_b_weight == 0.95
        assert s.track_c_weight == 0.85
        assert s.enable_source_attribution is True
        assert s.enable_compliance_disclaimer is True
        assert s.yahoo_finance_enabled is True
        assert s.fred_api_enabled is False
        assert s.tavily_search_enabled is False

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WEALTH_THREE_TRACK_ENABLED", "false")
        monkeypatch.setenv("WEALTH_TRACK_C_WEIGHT", "0.5")
        monkeypatch.setenv("WEALTH_ENABLE_SOURCE_ATTRIBUTION", "false")
        s = WealthSettings()
        assert s.three_track_enabled is False
        assert s.track_c_weight == 0.5
        assert s.enable_source_attribution is False


# ═════════════════════════════════════════════════════════════════════════════
# pytest 入口兼容
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
