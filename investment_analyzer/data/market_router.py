"""
市场路由器 - 自动识别股票代码所属市场，并路由到对应数据源
"""
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Market(Enum):
    A_SHARE = "a_share"      # 沪深A股
    HK_SHARE = "hk_share"    # 港股
    US_SHARE = "us_share"    # 美股


@dataclass
class StockInfo:
    """股票基础信息"""
    code: str           # 原始代码
    symbol: str         # 标准化代码 (如 600519, 00700, AAPL)
    market: Market      # 所属市场
    exchange: str       # 交易所 (SH/SZ/HK/NASDAQ/NYSE)
    name: str = ""      # 公司名称（后续填充）


class MarketRouter:
    """
    根据输入自动识别市场

    支持的输入格式:
      A股: 600519, 000001, sh600519, sz000001, 贵州茅台
      港股: 00700, 00700.HK, 9988.HK
      美股: AAPL, MSFT, TSLA
    """

    # A股代码规则
    A_SHARE_PATTERNS = {
        "SH": re.compile(r"^(sh)?6\d{5}$", re.IGNORECASE),       # 上证: 6开头
        "SZ": re.compile(r"^(sz)?[03]\d{5}$", re.IGNORECASE),    # 深证: 0/3开头
        "SZ_CYB": re.compile(r"^(sz)?30\d{4}$", re.IGNORECASE),  # 创业板
    }

    # 港股代码规则
    HK_PATTERN = re.compile(r"^(\d{4,5})(\.HK)?$", re.IGNORECASE)

    # 美股代码规则 (1-5个大写字母)
    US_PATTERN = re.compile(r"^[A-Z]{1,5}$")

    @classmethod
    def identify(cls, input_code: str) -> StockInfo:
        """识别股票代码所属市场"""
        code = input_code.strip()

        # 1. 检查是否为A股代码
        a_share = cls._check_a_share(code)
        if a_share:
            return a_share

        # 2. 检查是否为港股代码 (带 .HK 后缀)
        if code.upper().endswith(".HK"):
            num = code.replace(".HK", "").replace(".hk", "")
            return StockInfo(
                code=input_code,
                symbol=num.zfill(5),
                market=Market.HK_SHARE,
                exchange="HK",
            )

        # 3. 检查是否为纯数字（区分A股和港股）
        if code.isdigit():
            if len(code) == 6:
                # 6位数字 -> A股
                return cls._parse_a_share_number(code, input_code)
            elif len(code) <= 5:
                # 4-5位数字 -> 港股
                return StockInfo(
                    code=input_code,
                    symbol=code.zfill(5),
                    market=Market.HK_SHARE,
                    exchange="HK",
                )

        # 4. 检查是否为美股代码 (纯字母)
        if cls.US_PATTERN.match(code.upper()):
            return StockInfo(
                code=input_code,
                symbol=code.upper(),
                market=Market.US_SHARE,
                exchange="US",
            )

        # 5. 可能是中文名称，后续通过搜索解析
        return StockInfo(
            code=input_code,
            symbol=input_code,
            market=Market.A_SHARE,  # 默认按A股处理
            exchange="",
            name=input_code,
        )

    @classmethod
    def _check_a_share(cls, code: str) -> Optional[StockInfo]:
        """检查是否为带前缀的A股代码"""
        lower = code.lower()
        if lower.startswith("sh") and len(lower) == 8:
            return StockInfo(
                code=code, symbol=lower[2:],
                market=Market.A_SHARE, exchange="SH",
            )
        if lower.startswith("sz") and len(lower) == 8:
            return StockInfo(
                code=code, symbol=lower[2:],
                market=Market.A_SHARE, exchange="SZ",
            )
        return None

    @classmethod
    def _parse_a_share_number(cls, code: str, raw: str) -> StockInfo:
        """解析6位纯数字A股代码"""
        if code.startswith("6"):
            exchange = "SH"
        elif code.startswith(("0", "3")):
            exchange = "SZ"
        else:
            exchange = "SZ"  # 默认
        return StockInfo(
            code=raw, symbol=code,
            market=Market.A_SHARE, exchange=exchange,
        )
