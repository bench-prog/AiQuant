"""
AiQuant 训练数据工具。

训练脚本共享的数据加载和外部数据合并函数。
用法:
    from research.data_utils import load_training_data, merge_external_data
    df = load_training_data()
    df = merge_external_data(df)
"""

import logging
from typing import Optional

import pandas as pd

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from data.market_data import fetch_ohlcv_ccxt
from data.service import query, merge_into
import data.service_defaults  # registers built-in data sources

from research.training_config import (
    SYMBOL,
    TIMEFRAME,
    TRAIN_START,
    FULL_END,
    EXCHANGE,
)

logger = logging.getLogger(__name__)


def load_training_data(
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    start: str = TRAIN_START,
    end: str = FULL_END,
    exchange: str = EXCHANGE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """通过 ccxt 加载历史 OHLCV 数据。

    Args:
        symbol: 交易对，如 "BTC/USDT"
        timeframe: K线周期，如 "1h"
        start: 起始日期，如 "2022-01-01"
        end: 结束日期，如 "2024-12-31"
        exchange: 交易所名，如 "binance"
        use_cache: 是否使用本地 Parquet 缓存

    Returns:
        OHLCV DataFrame
    """
    df = fetch_ohlcv_ccxt(
        symbol=symbol,
        timeframe=timeframe,
        start_date=start,
        end_date=end,
        exchange_name=exchange,
        use_cache=use_cache,
    )
    logger.info("Loaded %d rows from %s.", len(df), exchange)
    return df


def merge_external_data(
    df: pd.DataFrame,
    symbol: str = SYMBOL,
    start: str = TRAIN_START,
    end: str = FULL_END,
    exchange: str = EXCHANGE,
    timeframe: str = TIMEFRAME,
) -> pd.DataFrame:
    """将资金费率和持仓量数据合并进 OHLCV DataFrame。

    资金费率和持仓量通过统一数据服务层查询，失败时降级（记录 warning，返回原 df）。

    Args:
        df: OHLCV DataFrame
        symbol: 交易对
        start: 起始日期
        end: 结束日期
        exchange: 交易所名
        timeframe: K线周期（持仓量查询用）

    Returns:
        合并后的 DataFrame
    """
    df = df.copy()

    try:
        fr_df = query(
            "funding_rate", symbol, since=start, until=end,
            exchange_name=exchange, use_cache=True,
        )
        df = merge_into(df, fr_df, "fundingRate")
        logger.info("Merged funding rate data (%d records).", len(fr_df))
    except Exception as e:
        logger.warning("Failed to fetch/merge funding rate: %s", e)

    try:
        oi_df = query(
            "open_interest", symbol, since=start, until=end,
            exchange_name=exchange, timeframe=timeframe, use_cache=True,
        )
        df = merge_into(df, oi_df, "openInterest")
        logger.info("Merged open interest data (%d records).", len(oi_df))
    except Exception as e:
        logger.warning("Failed to fetch/merge open interest: %s", e)

    return df
