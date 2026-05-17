"""
ai_app4 外部数据源集合。

Track B：金融 API
  - YahooFinanceSource：股价、市值、PE、财报日历
  - FredAPISource：宏观指标（CPI、非农、利率、GDP）

Track C：网络搜索
  - TavilySearchSource：通用实时搜索

所有数据源均继承 rag_framework.datasource.DataSource，
统一返回 RetrievedDoc 格式，由 ThreeTrackRetriever 编排融合。
"""

from ai_app4.service.datasources.yahoo_finance import YahooFinanceSource
from ai_app4.service.datasources.fred_api import FredAPISource

__all__ = [
    "YahooFinanceSource",
    "FredAPISource",
]
