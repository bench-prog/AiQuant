"""
CCXT-based crypto data fetcher for AiQuant.

Fetches OHLCV historical data from Binance (or any ccxt-supported exchange)
with automatic pagination, rate-limit handling, and local caching.

Usage:
    from market_data import fetch_ohlcv_ccxt
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
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    filename = f"{exchange_name}_{safe_symbol}_{timeframe}_{start}_{end}.parquet"
    return CACHE_DIR / filename


def _cache_path_data_type(
    symbol: str,
    data_type: str,
    start: str,
    end: str,
    exchange_name: str,
    timeframe: str = "",
) -> Path:
    """Generate a cache file path for non-OHLCV data (funding rate, open interest, etc.)."""
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    tf_suffix = f"_{timeframe}" if timeframe else ""
    filename = f"{exchange_name}_{safe_symbol}_{data_type}{tf_suffix}_{start}_{end}.parquet"
    return CACHE_DIR / filename


def _to_perpetual_symbol(symbol: str) -> str:
    """Convert spot symbol (e.g. BTC/USDT) to perpetual format (BTC/USDT:USDT) if needed."""
    if ":" not in symbol:
        return f"{symbol}:USDT"
    return symbol


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
    symbol: str = "BTC/USDT",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical funding rate for perpetual contracts.

    Args:
        symbol: Trading pair. Spot format (e.g. "BTC/USDT") is auto-converted
                to perpetual format ("BTC/USDT:USDT").
        start_date: Start date in "YYYY-MM-DD"
        end_date: End date in "YYYY-MM-DD"
        exchange_name: ccxt exchange id
        use_cache: If True, load from local cache if available

    Returns:
        DataFrame with columns: [date, fundingRate]
        Suitable for merge_asof with OHLCV data on the 'date' column.
    """
    perp_symbol = _to_perpetual_symbol(symbol)
    cache_file = _cache_path_data_type(
        perp_symbol, "funding_rate", start_date, end_date, exchange_name
    )

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached funding rate from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Fetching funding rate for {perp_symbol}...")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_rates = []

    while since < end_ms:
        try:
            rates = exchange.fetchFundingRateHistory(perp_symbol, since=since, limit=1000)
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

    # Filter to exact date range
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]

    if len(df) == 0:
        logger.warning("No funding rate data in requested date range.")
        return pd.DataFrame(columns=["date", "fundingRate"])

    # Save to cache
    df.to_parquet(cache_file)
    logger.info(f"Funding rate cached to {cache_file} ({len(df)} rows)")
    return df


def fetch_open_interest(
    symbol: str = "BTC/USDT",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
    timeframe: str = "1h",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical open interest for perpetual contracts.

    Note:
        Binance only retains ~30 days of 1h open interest history via public API.
        For older historical periods the returned DataFrame may be empty.
        The caller should handle missing data gracefully (ffill + fillna(0)).

    Args:
        symbol: Trading pair. Spot format is auto-converted to perpetual format.
        start_date: Start date in "YYYY-MM-DD"
        end_date: End date in "YYYY-MM-DD"
        exchange_name: ccxt exchange id
        timeframe: OI snapshot interval, e.g. "1h", "4h", "1d"
        use_cache: If True, load from local cache if available

    Returns:
        DataFrame with columns: [date, openInterest]
        Suitable for merge_asof with OHLCV data on the 'date' column.
    """
    perp_symbol = _to_perpetual_symbol(symbol)
    cache_file = _cache_path_data_type(
        perp_symbol, "open_interest", start_date, end_date, exchange_name, timeframe
    )

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached open interest from {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"Fetching open interest for {perp_symbol} ({timeframe})...")
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_oi = []

    # Some exchanges (e.g. Binance) restrict historical OI to ~30 days.
    # We attempt pagination; if the exchange rejects old 'since', we fall back.
    while since < end_ms:
        try:
            oi_batch = exchange.fetchOpenInterestHistory(
                perp_symbol, timeframe=timeframe, since=since, limit=500
            )
        except ccxt.ExchangeError as e:
            err_msg = str(e).lower()
            if "starttime" in err_msg or "parameter" in err_msg:
                logger.warning(
                    f"Exchange rejected old startTime for open interest "
                    f"({pd.to_datetime(since, unit='ms')}): {e}"
                )
                # Attempt one-shot fetch without 'since' (most recent window)
                try:
                    oi_batch = exchange.fetchOpenInterestHistory(
                        perp_symbol, timeframe=timeframe, limit=500
                    )
                except Exception as e2:
                    logger.warning(f"Fallback OI fetch also failed: {e2}")
                    oi_batch = []
                # Filter to requested range locally, then break
                if oi_batch:
                    all_oi.extend(oi_batch)
                break
            else:
                logger.warning(f"Error fetching open interest: {e}")
                break
        except Exception as e:
            logger.warning(f"Error fetching open interest: {e}")
            break

        if not oi_batch:
            break

        all_oi.extend(oi_batch)
        last_ts = exchange.parse8601(oi_batch[-1]["datetime"])
        since = last_ts + 1

    if not all_oi:
        logger.warning("No open interest data available.")
        return pd.DataFrame(columns=["date", "openInterest"])

    # Normalize ccxt open-interest structure
    df = pd.DataFrame(all_oi)
    df["date"] = pd.to_datetime(df["datetime"], utc=True)

    # ccxt may return 'baseVolume' or 'openInterestAmount' as the OI value
    if "openInterestAmount" in df.columns:
        df["openInterest"] = pd.to_numeric(df["openInterestAmount"], errors="coerce")
    elif "baseVolume" in df.columns:
        df["openInterest"] = pd.to_numeric(df["baseVolume"], errors="coerce")
    else:
        logger.warning("Unexpected open interest schema; available columns: %s", df.columns.tolist())
        return pd.DataFrame(columns=["date", "openInterest"])

    df = df[["date", "openInterest"]].sort_values("date").reset_index(drop=True)

    # Filter to exact date range
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]

    if len(df) == 0:
        logger.warning("No open interest data in requested date range.")
        return pd.DataFrame(columns=["date", "openInterest"])

    # Save to cache
    df.to_parquet(cache_file)
    logger.info(f"Open interest cached to {cache_file} ({len(df)} rows)")
    return df


def fetch_coinpaprika_tickers(use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch all tickers from CoinPaprika API.

    Returns:
        DataFrame with columns: id, name, symbol, rank, first_data_at,
        price, volume_24h, market_cap, percent_change_7d, percent_change_24h
    """
    today_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / f"coinpaprika_tickers_{today_date}.parquet"

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached CoinPaprika tickers from {cache_file}")
        return pd.read_parquet(cache_file)

    url = "https://api.coinpaprika.com/v1/tickers"
    logger.info(f"Fetching CoinPaprika tickers from {url}...")

    try:
        try:
            import requests

            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except ImportError:
            import json
            import urllib.request

            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.warning(f"Failed to fetch CoinPaprika tickers: {e}")
        return pd.DataFrame()

    if not isinstance(data, list):
        logger.warning(f"Unexpected CoinPaprika response format: {type(data)}")
        return pd.DataFrame()

    records = []
    for item in data:
        quotes = item.get("quotes", {})
        usd_quote = quotes.get("USD", {})
        records.append(
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "symbol": str(item.get("symbol", "")).upper(),
                "rank": item.get("rank"),
                "first_data_at": item.get("first_data_at"),
                "price": usd_quote.get("price"),
                "volume_24h": usd_quote.get("volume_24h"),
                "market_cap": usd_quote.get("market_cap"),
                "percent_change_7d": usd_quote.get("percent_change_7d"),
                "percent_change_24h": usd_quote.get("percent_change_24h"),
            }
        )

    df = pd.DataFrame(records)

    # Ensure numeric columns are numeric
    for col in ["price", "volume_24h", "market_cap", "percent_change_7d", "percent_change_24h"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save to cache
    df.to_parquet(cache_file)
    logger.info(f"CoinPaprika tickers cached to {cache_file} ({len(df)} rows)")
    return df


def map_to_binance_pairs(df: pd.DataFrame, exchange_name: str = "binance") -> pd.DataFrame:
    """
    Map CoinPaprika symbols to Binance spot USDT pairs.

    Args:
        df: DataFrame with a 'symbol' column (uppercase coin ticker).
        exchange_name: ccxt exchange id.

    Returns:
        DataFrame with an added 'binance_pair' column, filtered to mappable rows.
    """
    logger.info(f"Loading markets from {exchange_name}...")
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({"enableRateLimit": True})
        markets = exchange.load_markets()
    except Exception as e:
        logger.warning(f"Failed to load markets from {exchange_name}: {e}")
        return df.iloc[0:0].copy()  # empty df with same columns

    mapping = {}
    for pair_symbol, market in markets.items():
        is_spot = market.get("spot", False) or market.get("type") == "spot"
        quote = market.get("quote", "")
        base = market.get("base", "")
        if is_spot and quote == "USDT" and base:
            mapping[base.lower()] = pair_symbol

    df = df.copy()
    df["binance_pair"] = df["symbol"].str.lower().map(mapping)
    before = len(df)
    df = df.dropna(subset=["binance_pair"]).reset_index(drop=True)
    after = len(df)
    logger.info(f"Mapped {after}/{before} symbols to {exchange_name} USDT spot pairs.")
    return df


def filter_smallcap_candidates(
    df: pd.DataFrame,
    max_market_cap: float = 500_000_000,
    min_volume: float = 10_000_000,
    min_turnover: float = 100.0,
    max_turnover: float = 500.0,
    min_age_days: int = 30,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Filter CoinPaprika tickers to small-cap high-momentum candidates traded on Binance.

    Steps:
        1. percent_change_7d > 0, sort descending.
        2. Compute turnover_rate = volume_24h / market_cap * 100, filter range.
        3. market_cap <= max_market_cap, volume_24h >= min_volume,
           first_data_at >= min_age_days ago.
        4. Map to Binance spot USDT pairs.
        5. Return top_n.
    """
    df = df.copy()

    # Step 1: positive 7d momentum
    df = df[df["percent_change_7d"] > 0].sort_values("percent_change_7d", ascending=False)

    # Step 2: turnover rate filter
    df["turnover_rate"] = df["volume_24h"] / df["market_cap"] * 100
    df = df[(df["turnover_rate"] >= min_turnover) & (df["turnover_rate"] <= max_turnover)]

    # Step 3: market cap, volume, age filters
    df = df[df["market_cap"] <= max_market_cap]
    df = df[df["volume_24h"] >= min_volume]

    cutoff_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=min_age_days)
    df["first_data_at_dt"] = pd.to_datetime(df["first_data_at"], errors="coerce", utc=True)
    df = df[df["first_data_at_dt"] <= cutoff_date]
    df = df.drop(columns=["first_data_at_dt"])

    # Step 4: map to Binance
    df = map_to_binance_pairs(df)

    # Step 5: top N
    df = df.head(top_n).reset_index(drop=True)
    logger.info(
        f"Smallcap universe: {len(df)} coins (cap≤{max_market_cap:,.0f}, "
        f"vol≥{min_volume:,.0f}, turnover {min_turnover:.0f}-{max_turnover:.0f}%, age≥{min_age_days}d)"
    )
    return df


if __name__ == "__main__":
    # Quick test
    df = fetch_ohlcv_ccxt("BTC/USDT", "1h", "2024-01-01", "2024-02-01")
    print(df.head())
    print(df.tail())
