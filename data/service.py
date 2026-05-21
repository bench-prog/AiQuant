"""
Unified data service layer for AiQuant.

Registry-based external data access. Strategies and training scripts query data
by key without caring about the underlying implementation.

Usage:
    from data.service import query, merge_into
    import data.service_defaults  # triggers registration

    funding = query("funding_rate", "BTC/USDT", since="2022-01-01", until="2024-12-31",
                    exchange_name="binance", cache_dir="data/cache")
    df = merge_into(df, funding, "fundingRate")
"""

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, Callable] = {}
_CACHE: dict[str, pd.DataFrame] = {}


def register(key: str, fetcher: Callable):
    """Register a data source.

    fetcher signature: (symbol, start_date, end_date, **kwargs) -> DataFrame
    The returned DataFrame must have a datetime column named 'date'.
    """
    _REGISTRY[key] = fetcher
    logger.debug("Registered data source: %s", key)


def query(key: str, symbol: str, since: str, until: str, **kwargs) -> pd.DataFrame:
    """Query external data by registered key. Module-level cache avoids re-fetching."""
    cache_key = f"{key}:{symbol}:{since}:{until}:{hash(tuple(sorted(kwargs.items())))}"
    if cache_key not in _CACHE:
        if key not in _REGISTRY:
            raise KeyError(f"Unregistered data key: '{key}'. Available: {list(_REGISTRY.keys())}")
        logger.info("Querying data: key=%s symbol=%s since=%s until=%s", key, symbol, since, until)
        _CACHE[cache_key] = _REGISTRY[key](symbol, start_date=since, end_date=until, **kwargs)
    return _CACHE[cache_key]


def merge_into(df: pd.DataFrame, ext_df: pd.DataFrame, col: str, fillna=0) -> pd.DataFrame:
    """Merge external data into OHLCV dataframe using asof merge + forward fill.

    Args:
        df: OHLCV dataframe with a 'date' column or datetime index.
        ext_df: External dataframe with a 'date' column and `col` target column.
        col: Target column name to merge.
        fillna: Value to fill remaining NaNs after ffill.

    Returns:
        Enriched dataframe.
    """
    if ext_df.empty or col not in ext_df.columns:
        return df

    df = df.copy()

    # Ensure both have a sortable date column
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        else:
            raise ValueError("df must have a 'date' column or a DatetimeIndex")

    # Normalize datetime types for merge_asof compatibility (force ms precision)
    df["date"] = pd.to_datetime(df["date"], utc=True).astype("datetime64[ms, UTC]")
    ext_df = ext_df.copy()
    ext_df["date"] = pd.to_datetime(ext_df["date"], utc=True).astype("datetime64[ms, UTC]")

    df = df.sort_values("date").reset_index(drop=True)
    ext_df = ext_df.sort_values("date").reset_index(drop=True)

    df = pd.merge_asof(df, ext_df[["date", col]], on="date", direction="backward")
    df[col] = df[col].ffill().fillna(fillna)
    return df
