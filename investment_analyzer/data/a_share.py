"""
A股数据获取模块 - 基于 akshare
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

try:
    from data.cache import cached
except ImportError:
    from cache import cached


class AShareData:
    """A股数据获取器"""

    def __init__(self):
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ImportError("请安装 akshare: pip install akshare")

    @cached("a_stock_info")
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息"""
        try:
            df = self.ak.stock_individual_info_em(symbol=symbol)
            info = {}
            for _, row in df.iterrows():
                info[row["item"]] = row["value"]
            return info
        except Exception as e:
            return {"error": str(e)}

    @cached("a_price_hist")
    def get_price_history(self, symbol: str, period: str = "daily",
                          start: str = None, end: str = None) -> pd.DataFrame:
        """获取历史行情数据"""
        if not start:
            start = (datetime.now() - timedelta(days=365 * 5)).strftime("%Y%m%d")
        if not end:
            end = datetime.now().strftime("%Y%m%d")
        try:
            df = self.ak.stock_zh_a_hist(
                symbol=symbol, period=period,
                start_date=start, end_date=end, adjust="qfq"
            )
            return df
        except Exception as e:
            print(f"获取行情数据失败: {e}")
            return pd.DataFrame()

    @cached("a_financials")
    def get_financial_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """获取财务数据（利润表、资产负债表、现金流量表）"""
        result = {}
        try:
            # 主要财务指标
            result["indicators"] = self.ak.stock_financial_abstract_ths(
                symbol=symbol, indicator="按报告期"
            )
        except Exception:
            result["indicators"] = pd.DataFrame()

        try:
            # 利润表
            result["income"] = self.ak.stock_profit_sheet_by_report_em(symbol=f"SH{symbol}" if symbol.startswith("6") else f"SZ{symbol}")
        except Exception:
            result["income"] = pd.DataFrame()

        try:
            # 资产负债表
            result["balance"] = self.ak.stock_balance_sheet_by_report_em(symbol=f"SH{symbol}" if symbol.startswith("6") else f"SZ{symbol}")
        except Exception:
            result["balance"] = pd.DataFrame()

        return result

    @cached("a_valuation")
    def get_valuation(self, symbol: str) -> Dict[str, Any]:
        """获取估值数据（PE/PB/市值等）"""
        try:
            df = self.ak.stock_a_indicator_lg(symbol=symbol)
            if df.empty:
                return {}
            latest = df.iloc[-1]
            return {
                "pe": latest.get("pe"),
                "pe_ttm": latest.get("pe_ttm"),
                "pb": latest.get("pb"),
                "ps_ttm": latest.get("ps_ttm"),
                "dv_ttm": latest.get("dv_ttm"),
                "total_mv": latest.get("total_mv"),
                "history": df,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_sector_info(self, symbol: str) -> Dict[str, Any]:
        """获取行业分类信息"""
        try:
            df = self.ak.stock_board_industry_name_em()
            # 这里需要反查个股所属行业
            return {"sector_list": df}
        except Exception as e:
            return {"error": str(e)}

    @cached("a_insider")
    def get_holder_changes(self, symbol: str) -> pd.DataFrame:
        """获取股东/管理层持股变动"""
        try:
            df = self.ak.stock_inner_trade_xq(symbol=symbol)
            return df
        except Exception:
            return pd.DataFrame()

    def search_by_name(self, name: str) -> Optional[str]:
        """通过公司名搜索股票代码"""
        try:
            df = self.ak.stock_info_a_code_name()
            match = df[df["name"].str.contains(name)]
            if not match.empty:
                return match.iloc[0]["code"]
            return None
        except Exception:
            return None


class AShareMacro:
    """A股宏观数据"""

    def __init__(self):
        import akshare as ak
        self.ak = ak

    def get_pmi(self) -> pd.DataFrame:
        """获取PMI数据"""
        try:
            return self.ak.macro_china_pmi_yearly()
        except Exception:
            return pd.DataFrame()

    def get_m2(self) -> pd.DataFrame:
        """获取M2货币供应量"""
        try:
            return self.ak.macro_china_money_supply()
        except Exception:
            return pd.DataFrame()

    def get_bond_yield(self) -> pd.DataFrame:
        """获取10年期国债收益率"""
        try:
            return self.ak.bond_china_yield(start_date="2020-01-01")
        except Exception:
            return pd.DataFrame()

    def get_index_history(self, symbol: str = "sh000300") -> pd.DataFrame:
        """
        获取宽基指数历史行情（用于计算200日均线）
        默认: 沪深300 (sh000300)
        """
        try:
            df = self.ak.stock_zh_index_daily(symbol=symbol)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower() for c in df.columns]
            if "date" in df.columns:
                df = df.sort_values("date")
            return df
        except Exception:
            return pd.DataFrame()

    def get_index_pe_history(self, index_name: str = "沪深300") -> pd.DataFrame:
        """
        获取宽基指数 PE 历史数据（用于分位数计算）
        """
        try:
            # 方法1: 使用指数估值历史
            df = self.ak.stock_index_pe_lg(symbol=index_name)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        try:
            # 方法2: 使用 A 股整体 PE
            df = self.ak.stock_a_pe_and_dividend()
            if df is not None and not df.empty:
                return df
        except Exception:
            pass

        return pd.DataFrame()
