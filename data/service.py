"""
AiQuant 统一外部数据服务层。

通过注册表（registry）管理外部数据源，策略和训练脚本通过 key 查询数据，
无需关心底层实现（ccxt fetch / parquet 缓存 / 网络重试等）。

用法:
    from data.service import query, merge_into
    import data.service_defaults  # 触发内置数据源注册

    funding = query("funding_rate", "BTC/USDT", since="2022-01-01", until="2024-12-31",
                    exchange_name="binance")
    df = merge_into(df, funding, "fundingRate")
"""

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 注册表与缓存
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable] = {}
_CACHE: dict[str, pd.DataFrame] = {}


def register(key: str, fetcher: Callable):
    """注册数据源。

    fetcher 签名: (symbol, start_date, end_date, **kwargs) -> DataFrame
    返回的 DataFrame 必须包含名为 'date' 的 datetime 列。
    """
    _REGISTRY[key] = fetcher
    logger.debug("Registered data source: %s", key)


def query(key: str, symbol: str, since: str, until: str, **kwargs) -> pd.DataFrame:
    """通过已注册的 key 查询外部数据。模块级缓存避免重复拉取。"""
    cache_key = f"{key}:{symbol}:{since}:{until}:{hash(tuple(sorted(kwargs.items())))}"
    if cache_key not in _CACHE:
        if key not in _REGISTRY:
            raise KeyError(f"未注册数据源 key: '{key}'。可用: {list(_REGISTRY.keys())}")
        logger.info("Querying data: key=%s symbol=%s since=%s until=%s", key, symbol, since, until)
        _CACHE[cache_key] = _REGISTRY[key](symbol, start_date=since, end_date=until, **kwargs)
    return _CACHE[cache_key]


def merge_into(df: pd.DataFrame, ext_df: pd.DataFrame, col: str, fillna=0) -> pd.DataFrame:
    """将外部数据合并进 OHLCV DataFrame（asof merge + 前向填充）。

    Args:
        df: OHLCV DataFrame，含 'date' 列或 DatetimeIndex。
        ext_df: 外部 DataFrame，含 'date' 列和目标列 `col`。
        col: 要合并的目标列名。
        fillna: ffill 后剩余 NaN 的填充值。

    Returns:
        合并后的 enriched DataFrame。
    """
    if ext_df.empty or col not in ext_df.columns:
        return df

    df = df.copy()

    # 确保两边都有可排序的 date 列
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("df 必须有 'date' 列或 DatetimeIndex")

    # 统一 datetime 精度，避免 merge_asof 类型不匹配
    df["date"] = pd.to_datetime(df["date"], utc=True).astype("datetime64[ms, UTC]")
    ext_df = ext_df.copy()
    ext_df["date"] = pd.to_datetime(ext_df["date"], utc=True).astype("datetime64[ms, UTC]")

    df = df.sort_values("date").reset_index(drop=True)
    ext_df = ext_df.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(df, ext_df[["date", col]], on="date", direction="backward")
    df[col] = df[col].ffill().fillna(fillna)
    return df
