"""
CCXT-based crypto data fetcher for AiQuant.

Fetches OHLCV historical data from Binance (or any ccxt-supported exchange)
with automatic pagination, rate-limit handling, and local caching.

Usage:
    from data_fetcher import fetch_ohlcv_ccxt
    df = fetch_ohlcv_ccxt("BTC/USDT", "1h", "2022-01-01", "2024-12-31")
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _to_ms(date_str: str) -> int:
    """Convert 'YYYY-MM-DD' string to Unix timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _cache_path(symbol: str, timeframe: str, start: str, end: str, exchange_name: str) -> Path:
    """Generate a cache file path based on query parameters."""
    safe_symbol = symbol.replace("/", "_")
    filename = f"{exchange_name}_{safe_symbol}_{timeframe}_{start}_{end}.parquet"
    return CACHE_DIR / filename


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from a crypto exchange via ccxt.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT", "ETH/USDT:USDT" (perpetual)
        timeframe: K-line interval, e.g. "1m", "5m", "15m", "1h", "4h", "1d"
        start_date: Start date in "YYYY-MM-DD"
        end_date: End date in "YYYY-MM-DD"
        exchange_name: ccxt exchange id, e.g. "binance", "okx", "bybit"
        use_cache: If True, load from local cache if available

    Returns:
        DataFrame with columns: [date, open, high, low, close, volume]
    """
    cache_file = _cache_path(symbol, timeframe, start_date, end_date, exchange_name)

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(
        f"Fetching {symbol} ({timeframe}) from {exchange_name} "
        f"between {start_date} and {end_date}..."
    )

    # Initialize exchange with rate limiting
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({
        "enableRateLimit": True,
        "options": {
            "defaultType": "spot",  # Change to "future" or "swap" for perpetuals
        },
    })

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_ohlcv = []

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        except ccxt.NetworkError as e:
            logger.warning(f"Network error: {e}. Retrying in 5s...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}")
            raise

        if not ohlcv:
            logger.info("No more data returned from exchange.")
            break

        all_ohlcv.extend(ohlcv)

        # Update 'since' to the timestamp after the last candle
        last_ts = ohlcv[-1][0]
        since = last_ts + 1

        # Stop if we've reached the end date
        if last_ts >= end_ms:
            break

        logger.info(f"Fetched {len(ohlcv)} candles. Latest: {pd.to_datetime(last_ts, unit='ms')}")

    if not all_ohlcv:
        raise ValueError("No data fetched. Check symbol, timeframe, and date range.")

    # Convert to DataFrame
    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # Filter to exact date range
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    logger.info(f"Total rows fetched: {len(df)}. Date range: {df['date'].min()} ~ {df['date'].max()}")

    # Save to cache
    df.to_parquet(cache_file)
    logger.info(f"Data cached to {cache_file}")

    return df


def fetch_funding_rate(
    symbol: str = "BTC/USDT:USDT",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
) -> pd.DataFrame:
    """
    Fetch historical funding rate for perpetual contracts.
    Note: Not all exchanges support this via ccxt.
    """
    logger.info(f"Fetching funding rate for {symbol}...")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_rates = []

    while since < end_ms:
        try:
            # Binance-specific funding rate fetch
            if exchange_name == "binance":
                rates = exchange.fetchFundingRateHistory(symbol, since=since, limit=1000)
            else:
                rates = exchange.fetchFundingRateHistory(symbol, since=since, limit=1000)
        except Exception as e:
            logger.warning(f"Error fetching funding rate: {e}")
            break

        if not rates:
            break

        all_rates.extend(rates)
        last_ts = exchange.parse8601(rates[-1]["datetime"])
        since = last_ts + 1

    if not all_rates:
        logger.warning("No funding rate data available.")
        return pd.DataFrame(columns=["date", "fundingRate"])

    df = pd.DataFrame(all_rates)
    df["date"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[["date", "fundingRate"]].sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick test
    df = fetch_ohlcv_ccxt("BTC/USDT", "1h", "2024-01-01", "2024-02-01")
    print(df.head())
    print(df.tail())
