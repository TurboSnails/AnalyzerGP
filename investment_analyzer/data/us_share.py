"""
美股 & 港股数据获取模块 - 基于 yfinance
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any

try:
    from data.cache import cached
except ImportError:
    from cache import cached


class YFinanceData:
    """
    美股和港股数据获取器
    美股: 直接使用 ticker (如 AAPL)
    港股: 使用 ticker.HK 格式 (如 0700.HK)
    """

    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            raise ImportError("请安装 yfinance: pip install yfinance")

    def _get_ticker(self, symbol: str, market: str = "US") -> str:
        """转换为yfinance的ticker格式"""
        if market == "HK":
            # 港股: 确保是4-5位数字 + .HK
            num = symbol.lstrip("0") or "0"
            return f"{symbol}.HK"
        return symbol  # 美股直接使用

    @cached("us_stock_info")
    def get_stock_info(self, symbol: str, market: str = "US") -> Dict[str, Any]:
        """获取股票基本信息"""
        ticker_str = self._get_ticker(symbol, market)
        try:
            ticker = self.yf.Ticker(ticker_str)
            info = ticker.info
            return {
                "name": info.get("longName", info.get("shortName", "")),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "market_cap": info.get("marketCap"),
                "pe_trailing": info.get("trailingPE"),
                "pe_forward": info.get("forwardPE"),
                "pb": info.get("priceToBook"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "52w_high": info.get("fiftyTwoWeekHigh"),
                "52w_low": info.get("fiftyTwoWeekLow"),
                "currency": info.get("currency", ""),
                "exchange": info.get("exchange", ""),
            }
        except Exception as e:
            return {"error": str(e)}

    @cached("us_price_hist")
    def get_price_history(self, symbol: str, market: str = "US",
                          period: str = "5y") -> pd.DataFrame:
        """获取历史行情数据"""
        ticker_str = self._get_ticker(symbol, market)
        try:
            ticker = self.yf.Ticker(ticker_str)
            df = ticker.history(period=period)
            return df
        except Exception as e:
            print(f"获取行情数据失败: {e}")
            return pd.DataFrame()

    @cached("us_financials")
    def get_financials(self, symbol: str, market: str = "US") -> Dict[str, pd.DataFrame]:
        """获取财务数据"""
        ticker_str = self._get_ticker(symbol, market)
        try:
            ticker = self.yf.Ticker(ticker_str)
            return {
                "income": ticker.income_stmt,
                "balance": ticker.balance_sheet,
                "cashflow": ticker.cashflow,
                "quarterly_income": ticker.quarterly_income_stmt,
                "quarterly_balance": ticker.quarterly_balance_sheet,
                "quarterly_cashflow": ticker.quarterly_cashflow,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_insider_trades(self, symbol: str, market: str = "US") -> pd.DataFrame:
        """获取内部人交易（仅美股有效）"""
        ticker_str = self._get_ticker(symbol, market)
        try:
            ticker = self.yf.Ticker(ticker_str)
            return ticker.insider_transactions
        except Exception:
            return pd.DataFrame()
