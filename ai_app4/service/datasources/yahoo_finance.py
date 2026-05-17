"""
Yahoo Finance 数据源 — 实时股价与基本面数据。

基于 yfinance 库获取股票实时报价、市值、PE、财报日历等结构化数据。
适用于回答"英伟达现在股价多少"、"特斯拉 PE 多少"等时效性问题。

商业约束：
  - Yahoo Finance 是非官方数据源，存在限流风险
  - 生产环境应配合本地缓存（ttl=300s）降低调用频率
  - 失败时静默降级，不抛异常
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_framework.datasource.base import DataSource, DataSourceType, FetchContext, SourceResult
from rag_framework.domain.base import QueryRoute


class YahooFinanceSource(DataSource):
    """
    Yahoo Finance 实时数据接入。

    支持从用户查询中提取股票代码（ticker），获取实时报价和基本面数据。
    """

    _TICKER_MAP: dict[str, str] = {
        "英伟达": "NVDA",
        "nvidia": "NVDA",
        "特斯拉": "TSLA",
        "tsla": "TSLA",
        "苹果": "AAPL",
        "apple": "AAPL",
        "微软": "MSFT",
        "microsoft": "MSFT",
        "谷歌": "GOOGL",
        "google": "GOOGL",
        "亚马逊": "AMZN",
        "amazon": "AMZN",
        "meta": "META",
        "脸书": "META",
        "facebook": "META",
        "amd": "AMD",
        "英特尔": "INTC",
        "intel": "INTC",
        "台积电": "TSM",
        "tsmc": "TSM",
        "阿里巴巴": "BABA",
        "alibaba": "BABA",
        "腾讯": "TCEHY",
        "tencent": "TCEHY",
        "拼多多": "PDD",
        "pdd": "PDD",
        "小米": "XIACY",
        "xiaomi": "XIACY",
    }

    def __init__(self, cache: Any | None = None) -> None:
        """
        Args:
            cache: 可选的缓存对象，需实现 get(key) / set(key, value, ttl) 接口
        """
        self._cache = cache
        self._yf: Any | None = None  # 惰性导入 yfinance

    # ── DataSource 接口实现 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "yahoo_finance"

    @property
    def data_type(self) -> DataSourceType:
        return DataSourceType.STRUCTURED

    @property
    def ttl_seconds(self) -> int:
        return 300  # 股价缓存 5 分钟

    @property
    def default_timeout(self) -> float:
        return 3.0

    def should_fetch(self, query: QueryRoute, context: FetchContext) -> bool:
        """
        前置判断：查询中是否包含股票相关实体。

        无 ticker 时不触发，避免浪费调用。
        """
        tickers = self._extract_tickers(query.text, context)
        return len(tickers) > 0 and self.enabled

    async def fetch(
        self,
        query: QueryRoute,
        context: FetchContext,
    ) -> SourceResult:
        tickers = self._extract_tickers(query.text, context)
        if not tickers:
            return SourceResult.empty(self.name)

        start = time.monotonic()
        docs: list[Any] = []
        error: str | None = None

        try:
            # 逐只查询，最多 3 只（控制成本）
            for ticker in tickers[:3]:
                doc = await self._fetch_ticker(ticker)
                if doc is not None:
                    docs.append(doc)
        except Exception as exc:
            error = f"Yahoo Finance 获取失败: {exc}"

        latency_ms = (time.monotonic() - start) * 1000
        return SourceResult(
            docs=docs,
            success=error is None,
            error=error,
            source_name=self.name,
            latency_ms=latency_ms,
            metadata={"tickers_queried": tickers[:3]},
        )

    # ── 内部方法 ─────────────────────────────────────────────────────────

    def _extract_tickers(self, text: str, context: FetchContext) -> list[str]:
        """从查询文本和 NER 实体中提取股票代码。"""
        tickers: set[str] = set()

        # 1. 从 NER 实体提取（classify 节点已识别的 ticker）
        for t in context.get_entities_by_type("ticker"):
            tickers.add(t.upper())

        # 2. 从文本关键词映射（中英文名称）
        lower_text = text.lower()
        for keyword, ticker in self._TICKER_MAP.items():
            if keyword in lower_text:
                tickers.add(ticker)

        # 3. 直接匹配大写代码（如 "NVDA"、"AAPL"）
        import re
        for match in re.findall(r"\b[A-Z]{1,5}\b", text):
            if match not in ("USD", "CNY", "HKD", "ETF", "IPO"):  # 过滤常见非 ticker 大写词
                tickers.add(match)

        return list(tickers)

    async def _fetch_ticker(self, ticker: str) -> Any | None:
        """获取单只股票的实时数据，返回 RetrievedDoc 或 None。"""
        # 惰性导入 yfinance（避免启动时加载）
        if self._yf is None:
            try:
                import yfinance as yf

                self._yf = yf
            except ImportError:
                # yfinance 未安装时，构造模拟数据（用于测试）
                return self._make_doc(
                    doc_id=f"yf_{ticker}_mock",
                    text=f"【模拟数据】{ticker} 当前股价 $150.00，市值 3.5T，PE 32.5，52周高低 $120-$200。",
                    score=0.85,
                    extra_metadata={"ticker": ticker, "mock": True},
                )

        cache_key = f"yf:{ticker}"
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            # yfinance 是同步阻塞库，用 asyncio.to_thread 卸载
            info = await asyncio.wait_for(
                asyncio.to_thread(self._yf.Ticker(ticker).info),
                timeout=self.default_timeout,
            )

            price = info.get("currentPrice") or info.get("regularMarketPrice") or "N/A"
            market_cap = info.get("marketCap")
            pe = info.get("trailingPE") or info.get("forwardPE") or "N/A"
            week_52_high = info.get("fiftyTwoWeekHigh", "N/A")
            week_52_low = info.get("fiftyTwoWeekLow", "N/A")
            sector = info.get("sector", "N/A")
            name = info.get("longName") or info.get("shortName") or ticker

            text = (
                f"【{name} ({ticker}) 实时数据】\n"
                f"当前股价: ${price}\n"
                f"市值: {self._format_market_cap(market_cap)}\n"
                f"市盈率 (PE): {pe}\n"
                f"52 周区间: ${week_52_low} - ${week_52_high}\n"
                f"所属行业: {sector}"
            )

            doc = self._make_doc(
                doc_id=f"yf_{ticker}_{int(time.time())}",
                text=text,
                score=0.95,
                extra_metadata={
                    "ticker": ticker,
                    "price": price,
                    "market_cap": market_cap,
                    "pe": pe,
                    "sector": sector,
                },
            )

            if self._cache is not None:
                self._cache.set(cache_key, doc, ttl=self.ttl_seconds)

            return doc

        except asyncio.TimeoutError:
            return None
        except Exception:
            return None

    @staticmethod
    def _format_market_cap(cap: Any) -> str:
        if cap is None:
            return "N/A"
        try:
            cap_f = float(cap)
            if cap_f >= 1e12:
                return f"{cap_f / 1e12:.2f}T"
            if cap_f >= 1e9:
                return f"{cap_f / 1e9:.2f}B"
            if cap_f >= 1e6:
                return f"{cap_f / 1e6:.2f}M"
            return str(cap)
        except (TypeError, ValueError):
            return str(cap)
