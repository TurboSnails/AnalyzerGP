"""
FRED (Federal Reserve Economic Data) 数据源 — 美国宏观经济指标。

获取 CPI、非农就业、联邦基金利率、GDP、失业率等关键宏观数据。
FRED API 免费注册即可使用，是商业级金融 AI 的理想数据源。

注册地址: https://fred.stlouisfed.org/docs/api/api_key.html
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
from rag_framework.domain.base import QueryRoute


class FredAPISource(DataSource):
    """
    FRED 宏观经济数据接入。

    支持根据查询中的指标名称自动匹配 FRED Series ID，
    获取最新值和历史趋势。
    """

    _BASE_URL = "https://api.stlouisfed.org/fred"

    # 常用指标映射：中文/英文关键词 → FRED Series ID
    _SERIES_MAP: dict[str, str] = {
        # 通胀
        "cpi": "CPIAUCSL",
        "消费者物价指数": "CPIAUCSL",
        "消费者价格指数": "CPIAUCSL",
        "core cpi": "CPILFESL",
        "核心cpi": "CPILFESL",
        "ppi": "PPIFID",
        "生产者物价指数": "PPIFID",
        # 就业
        "nonfarm payrolls": "PAYEMS",
        "非农就业": "PAYEMS",
        "unemployment rate": "UNRATE",
        "失业率": "UNRATE",
        "initial claims": "ICSA",
        "初请失业金": "ICSA",
        # 利率
        "fed funds rate": "DFF",
        "联邦基金利率": "DFF",
        "美联储利率": "DFF",
        "interest rate": "DFF",
        "利率": "DFF",
        "10 year treasury": "DGS10",
        "十年期国债": "DGS10",
        "国债收益率": "DGS10",
        # 增长
        "gdp": "GDP",
        "国内生产总值": "GDP",
        "real gdp": "GDPC1",
        "实际gdp": "GDPC1",
        "pmi": "NAPM",
        "制造业pmi": "NAPM",
        # 其他
        "m2": "M2SL",
        "m2货币": "M2SL",
        "consumer sentiment": "UMCSENT",
        "消费者信心": "UMCSENT",
        "dxy": "DTWEXBGS",
        "美元指数": "DTWEXBGS",
        "vix": "VIXCLS",
        "恐慌指数": "VIXCLS",
    }

    def __init__(self, api_key: str = "", cache: Any | None = None) -> None:
        self._api_key = api_key or ""
        self._cache = cache
        self._client: httpx.AsyncClient | None = None

    # ── DataSource 接口实现 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "fred_api"

    @property
    def data_type(self) -> DataSourceType:
        return DataSourceType.STRUCTURED

    @property
    def ttl_seconds(self) -> int:
        return 3600  # 宏观指标缓存 1 小时（更新频率低）

    @property
    def default_timeout(self) -> float:
        return 3.0

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def should_fetch(self, query: QueryRoute, context: FetchContext) -> bool:
        """查询中是否包含宏观经济指标关键词。"""
        series_ids = self._extract_series_ids(query.text)
        return len(series_ids) > 0 and self.enabled

    async def fetch(
        self,
        query: QueryRoute,
        context: FetchContext,
    ) -> SourceResult:
        if not self._api_key:
            return SourceResult.empty(
                self.name, error="FRED API Key 未配置"
            )

        series_ids = self._extract_series_ids(query.text)
        if not series_ids:
            return SourceResult.empty(self.name)

        start = time.monotonic()
        docs: list[Any] = []
        error: str | None = None

        # 初始化 httpx client（惰性）
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._BASE_URL,
                timeout=httpx.Timeout(self.default_timeout),
            )

        try:
            for sid in list(series_ids)[:3]:  # 最多查 3 个指标
                doc = await self._fetch_series(sid)
                if doc is not None:
                    docs.append(doc)
        except Exception as exc:
            error = f"FRED API 获取失败: {exc}"

        latency_ms = (time.monotonic() - start) * 1000
        return SourceResult(
            docs=docs,
            success=error is None,
            error=error,
            source_name=self.name,
            latency_ms=latency_ms,
            metadata={"series_queried": list(series_ids)[:3]},
        )

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _extract_series_ids(self, text: str) -> set[str]:
        """从查询文本中提取 FRED Series ID。"""
        series_ids: set[str] = set()
        lower_text = text.lower()

        for keyword, sid in self._SERIES_MAP.items():
            if keyword in lower_text:
                series_ids.add(sid)

        return series_ids

    async def _fetch_series(self, series_id: str) -> Any | None:
        """获取单个 Series 的最新数据和元信息。"""
        cache_key = f"fred:{series_id}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            # 1. 获取最新数据点
            obs_resp = await self._client.get(
                "/series/observations",
                params={
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                    "sort_order": "desc",
                    "limit": 3,  # 最近 3 个数据点
                },
            )
            obs_resp.raise_for_status()
            obs_data = obs_resp.json()
            observations = obs_data.get("observations", [])

            if not observations:
                return None

            latest = observations[0]
            prev = observations[1] if len(observations) > 1 else None

            # 2. 获取 Series 元信息
            info_resp = await self._client.get(
                "/series",
                params={
                    "series_id": series_id,
                    "api_key": self._api_key,
                    "file_type": "json",
                },
            )
            info_resp.raise_for_status()
            info_data = info_resp.json()
            series_info = info_data.get("seriess", [{}])[0]

            title = series_info.get("title", series_id)
            units = series_info.get("units", "")
            frequency = series_info.get("frequency", "")

            # 3. 计算环比变化
            change_text = ""
            if prev and latest.get("value") != "." and prev.get("value") != ".":
                try:
                    latest_val = float(latest["value"])
                    prev_val = float(prev["value"])
                    change = latest_val - prev_val
                    change_pct = (change / prev_val * 100) if prev_val != 0 else 0
                    direction = "↑" if change > 0 else "↓"
                    change_text = (
                        f"环比变化: {direction} {abs(change):.2f} ({abs(change_pct):.2f}%)\n"
                        f"上期数据 ({prev['date']}): {prev['value']}"
                    )
                except (ValueError, TypeError):
                    pass

            text = (
                f"【{title} ({series_id})】\n"
                f"最新数据 ({latest['date']}): {latest['value']} {units}\n"
                f"发布频率: {frequency}\n"
                f"{change_text}"
            ).strip()

            doc = self._make_doc(
                doc_id=f"fred_{series_id}_{int(time.time())}",
                text=text,
                score=0.92,
                extra_metadata={
                    "series_id": series_id,
                    "title": title,
                    "latest_date": latest.get("date"),
                    "latest_value": latest.get("value"),
                    "units": units,
                    "frequency": frequency,
                },
            )

            if self._cache is not None:
                self._cache.set(cache_key, doc, ttl=self.ttl_seconds)

            return doc

        except (httpx.TimeoutException, asyncio.TimeoutError):
            return None
        except httpx.HTTPStatusError as exc:
            # FRED API 限流或 Key 无效时静默降级
            if exc.response.status_code == 429:
                return None
            return None
        except Exception:
            return None
