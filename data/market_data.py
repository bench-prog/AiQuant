"""
AiQuant CCXT 加密货币数据下载器。

从 Binance（或任何 ccxt 支持的交易所）获取历史 OHLCV 数据，
支持自动分页、速率限制处理、本地 Parquet 缓存。

用法:
    from data.market_data import fetch_ohlcv_ccxt
    df = fetch_ohlcv_ccxt("BTC/USDT", "1h", "2022-01-01", "2024-12-31")
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import ccxt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def _to_ms(date_str: str) -> int:
    """将 'YYYY-MM-DD' 字符串转为毫秒级 Unix 时间戳。"""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _cache_path(symbol: str, timeframe: str, start: str, end: str, exchange_name: str) -> Path:
    """根据查询参数生成 OHLCV 缓存文件路径。"""
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
    """生成非 OHLCV 数据（资金费率、持仓量等）的缓存文件路径。"""
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    tf_suffix = f"_{timeframe}" if timeframe else ""
    filename = f"{exchange_name}_{safe_symbol}_{data_type}{tf_suffix}_{start}_{end}.parquet"
    return CACHE_DIR / filename


def _to_perpetual_symbol(symbol: str) -> str:
    """将现货 symbol（如 BTC/USDT）自动转为永续格式（BTC/USDT:USDT）。"""
    if ":" not in symbol:
        return f"{symbol}:USDT"
    return symbol


def _init_exchange(exchange_name: str, **options) -> ccxt.Exchange:
    """初始化 ccxt exchange，启用速率限制。"""
    exchange_class = getattr(ccxt, exchange_name)
    return exchange_class({"enableRateLimit": True, **options})


def _fetch_paginated(
    fetch_page: Callable[[int, int], list],
    since: int,
    end_ms: int,
    parse_last_ts: Callable[[list], int],
    limit: int = 1000,
) -> list:
    """
    通用分页拉取循环。

    Args:
        fetch_page: 单页获取函数，签名 (since, limit) -> list。
        since: 起始时间戳（毫秒）。
        end_ms: 结束时间戳（毫秒）。
        parse_last_ts: 从 batch 中解析最后一条时间戳的函数。
        limit: 每页条数。

    Returns:
        所有拉取到的数据列表。
    """
    all_data = []
    while since < end_ms:
        try:
            batch = fetch_page(since=since, limit=limit)
        except ccxt.NetworkError as e:
            logger.warning(f"网络错误: {e}，5秒后重试...")
            time.sleep(5)
            continue
        except ccxt.ExchangeError as e:
            logger.error(f"交易所错误: {e}")
            raise

        if not batch:
            logger.info("交易所无更多数据返回。")
            break

        all_data.extend(batch)
        last_ts = parse_last_ts(batch)
        since = last_ts + 1

        if last_ts >= end_ms:
            break

        logger.info(f"已拉取 {len(batch)} 条。最新: {pd.to_datetime(last_ts, unit='ms')}")

    return all_data


def fetch_ohlcv_ccxt(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    通过 ccxt 从交易所获取 OHLCV 历史数据。

    Args:
        symbol: 交易对，如 "BTC/USDT" 或 "ETH/USDT:USDT"（永续）。
        timeframe: K 线周期，如 "1m", "5m", "15m", "1h", "4h", "1d"。
        start_date: 开始日期，格式 "YYYY-MM-DD"。
        end_date: 结束日期，格式 "YYYY-MM-DD"。
        exchange_name: ccxt 交易所 ID，如 "binance", "okx", "bybit"。
        use_cache: 为 True 时优先读取本地缓存。

    Returns:
        DataFrame，列: [date, open, high, low, close, volume]。
    """
    cache_file = _cache_path(symbol, timeframe, start_date, end_date, exchange_name)

    if use_cache and cache_file.exists():
        logger.info(f"从缓存加载数据: {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(
        f"拉取 {symbol} ({timeframe}) from {exchange_name}，"
        f"时间范围: {start_date} ~ {end_date}..."
    )

    exchange = _init_exchange(
        exchange_name,
        options={"defaultType": "spot"},  # 如需永续可改为 "future" 或 "swap"
    )

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)

    def _fetch_page(since, limit):
        return exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)

    all_ohlcv = _fetch_paginated(
        _fetch_page,
        since=since,
        end_ms=end_ms,
        parse_last_ts=lambda batch: batch[-1][0],
        limit=1000,
    )

    if not all_ohlcv:
        raise ValueError("未获取到数据，请检查交易对、周期和日期范围。")

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df[["date", "open", "high", "low", "close", "volume"]]

    # 过滤到精确日期范围
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    logger.info(f"共拉取 {len(df)} 行。时间范围: {df['date'].min()} ~ {df['date'].max()}")

    # 写入缓存
    df.to_parquet(cache_file)
    logger.info(f"数据已缓存至 {cache_file}")

    return df


def fetch_funding_rate(
    symbol: str = "BTC/USDT",
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    exchange_name: str = "binance",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    获取永续合约历史资金费率。

    Args:
        symbol: 交易对。现货格式（如 "BTC/USDT"）会自动转为永续格式。
        start_date: 开始日期，格式 "YYYY-MM-DD"。
        end_date: 结束日期，格式 "YYYY-MM-DD"。
        exchange_name: ccxt 交易所 ID。
        use_cache: 为 True 时优先读取本地缓存。

    Returns:
        DataFrame，列: [date, fundingRate]。
        可直接用 merge_asof 与 OHLCV 数据按 'date' 合并。
    """
    perp_symbol = _to_perpetual_symbol(symbol)
    cache_file = _cache_path_data_type(
        perp_symbol, "funding_rate", start_date, end_date, exchange_name
    )

    if use_cache and cache_file.exists():
        logger.info(f"从缓存加载资金费率: {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"拉取资金费率: {perp_symbol}...")
    exchange = _init_exchange(exchange_name)

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_rates = []

    while since < end_ms:
        try:
            rates = exchange.fetchFundingRateHistory(perp_symbol, since=since, limit=1000)
        except Exception as e:
            logger.warning(f"拉取资金费率出错: {e}")
            break

        if not rates:
            break

        all_rates.extend(rates)
        last_ts = exchange.parse8601(rates[-1]["datetime"])
        since = last_ts + 1

    if not all_rates:
        logger.warning("无资金费率数据。")
        return pd.DataFrame(columns=["date", "fundingRate"])

    df = pd.DataFrame(all_rates)
    df["date"] = pd.to_datetime(df["datetime"], utc=True)
    df = df[["date", "fundingRate"]].sort_values("date").reset_index(drop=True)

    # 过滤到精确日期范围
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]

    if len(df) == 0:
        logger.warning("请求时间范围内无资金费率数据。")
        return pd.DataFrame(columns=["date", "fundingRate"])

    # 写入缓存
    df.to_parquet(cache_file)
    logger.info(f"资金费率已缓存至 {cache_file} ({len(df)} 行)")
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
    获取永续合约历史持仓量（Open Interest）。

    Note:
        Binance 公开 API 仅保留约 30 天的 1h 持仓量历史。
        超过 30 天的请求可能返回空 DataFrame，调用方需优雅处理（ffill + fillna(0)）。

    Args:
        symbol: 交易对。现货格式会自动转为永续格式。
        start_date: 开始日期，格式 "YYYY-MM-DD"。
        end_date: 结束日期，格式 "YYYY-MM-DD"。
        exchange_name: ccxt 交易所 ID。
        timeframe: OI 快照周期，如 "1h", "4h", "1d"。
        use_cache: 为 True 时优先读取本地缓存。

    Returns:
        DataFrame，列: [date, openInterest]。
        可直接用 merge_asof 与 OHLCV 数据按 'date' 合并。
    """
    perp_symbol = _to_perpetual_symbol(symbol)
    cache_file = _cache_path_data_type(
        perp_symbol, "open_interest", start_date, end_date, exchange_name, timeframe
    )

    if use_cache and cache_file.exists():
        logger.info(f"从缓存加载持仓量: {cache_file}")
        return pd.read_parquet(cache_file)

    logger.info(f"拉取持仓量: {perp_symbol} ({timeframe})...")
    exchange = _init_exchange(exchange_name)

    since = _to_ms(start_date)
    end_ms = _to_ms(end_date)
    all_oi = []

    # Binance 等交易所限制历史 OI 约为 30 天。
    # 若交易所拒绝旧的 since，则回退到无 since 的单次抓取（最近窗口）。
    while since < end_ms:
        try:
            oi_batch = exchange.fetchOpenInterestHistory(
                perp_symbol, timeframe=timeframe, since=since, limit=500
            )
        except ccxt.ExchangeError as e:
            err_msg = str(e).lower()
            if "starttime" in err_msg or "parameter" in err_msg:
                logger.warning(
                    f"交易所拒绝旧 startTime ({pd.to_datetime(since, unit='ms')}): {e}"
                )
                # 回退：不带 since 的单次抓取（最近窗口）
                try:
                    oi_batch = exchange.fetchOpenInterestHistory(
                        perp_symbol, timeframe=timeframe, limit=500
                    )
                except Exception as e2:
                    logger.warning(f"回退 OI 抓取也失败: {e2}")
                    oi_batch = []
                # 本地过滤到请求范围后退出
                if oi_batch:
                    all_oi.extend(oi_batch)
                break
            else:
                logger.warning(f"拉取持仓量出错: {e}")
                break
        except Exception as e:
            logger.warning(f"拉取持仓量出错: {e}")
            break

        if not oi_batch:
            break

        all_oi.extend(oi_batch)
        last_ts = exchange.parse8601(oi_batch[-1]["datetime"])
        since = last_ts + 1

    if not all_oi:
        logger.warning("无持仓量数据。")
        return pd.DataFrame(columns=["date", "openInterest"])

    # 标准化 ccxt 返回的持仓量结构
    df = pd.DataFrame(all_oi)
    df["date"] = pd.to_datetime(df["datetime"], utc=True)

    # ccxt 可能返回 'openInterestAmount' 或 'baseVolume' 作为 OI 值
    if "openInterestAmount" in df.columns:
        df["openInterest"] = pd.to_numeric(df["openInterestAmount"], errors="coerce")
    elif "baseVolume" in df.columns:
        df["openInterest"] = pd.to_numeric(df["baseVolume"], errors="coerce")
    else:
        logger.warning("持仓量 schema 异常; 可用列: %s", df.columns.tolist())
        return pd.DataFrame(columns=["date", "openInterest"])

    df = df[["date", "openInterest"]].sort_values("date").reset_index(drop=True)

    # 过滤到精确日期范围
    df = df[(df["date"] >= pd.Timestamp(start_date, tz="UTC")) & (df["date"] < pd.Timestamp(end_date, tz="UTC"))]

    if len(df) == 0:
        logger.warning("请求时间范围内无持仓量数据。")
        return pd.DataFrame(columns=["date", "openInterest"])

    # 写入缓存
    df.to_parquet(cache_file)
    logger.info(f"持仓量已缓存至 {cache_file} ({len(df)} 行)")
    return df


def fetch_coinpaprika_tickers(use_cache: bool = True) -> pd.DataFrame:
    """
    从 CoinPaprika API 获取全部币种行情。

    Returns:
        DataFrame，列: id, name, symbol, rank, first_data_at,
        price, volume_24h, market_cap, percent_change_7d, percent_change_24h
    """
    today_date = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    cache_file = CACHE_DIR / f"coinpaprika_tickers_{today_date}.parquet"

    if use_cache and cache_file.exists():
        logger.info(f"从缓存加载 CoinPaprika: {cache_file}")
        return pd.read_parquet(cache_file)

    url = "https://api.coinpaprika.com/v1/tickers"
    logger.info(f"拉取 CoinPaprika tickers: {url}...")

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
        logger.warning(f"CoinPaprika 拉取失败: {e}")
        return pd.DataFrame()

    if not isinstance(data, list):
        logger.warning(f"CoinPaprika 返回格式异常: {type(data)}")
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
                "total_supply": item.get("total_supply"),
                "market_cap": usd_quote.get("market_cap"),
                "percent_change_7d": usd_quote.get("percent_change_7d"),
                "percent_change_24h": usd_quote.get("percent_change_24h"),
            }
        )

    df = pd.DataFrame(records)

    # 确保数值列为数值类型
    for col in ["price", "volume_24h", "market_cap", "percent_change_7d", "percent_change_24h", "total_supply"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 写入缓存
    df.to_parquet(cache_file)
    logger.info(f"CoinPaprika 已缓存至 {cache_file} ({len(df)} 行)")
    return df


def map_to_binance_pairs(df: pd.DataFrame, exchange_name: str = "binance") -> pd.DataFrame:
    """
    将 CoinPaprika symbol 映射到 Binance 现货 USDT 交易对。

    Args:
        df: 含 'symbol' 列（大写币种代码）的 DataFrame。
        exchange_name: ccxt 交易所 ID。

    Returns:
        增加 'binance_pair' 列并过滤掉无法映射的行。
    """
    logger.info(f"从 {exchange_name} 加载交易对...")
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class({"enableRateLimit": True})
        markets = exchange.load_markets()
    except Exception as e:
        logger.warning(f"加载 {exchange_name} 交易对失败: {e}")
        return df.iloc[0:0].copy()  # 保持列结构的空 DataFrame

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
    logger.info(f"映射成功 {after}/{before} 个 symbol 到 {exchange_name} USDT 现货对。")
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
    从 CoinPaprika tickers 中筛选小市值高动量候选币（仅保留在 Binance 有现货的）。

    NOTE: 市值和换手率会在策略内用实时 K 线价格 × total_supply 精确计算。
    本函数只做粗筛，控制白名单规模。

    步骤:
        1. 7d 涨幅 > 0，按涨幅降序。
        2. 粗筛市值 <= max_market_cap * 2（使用 CoinPaprika 静态值）。
        3. 24h 成交量 >= min_volume，上线时间 >= min_age_days。
        4. 必须有 total_supply > 0。
        5. 映射到 Binance 现货 USDT 对。
        6. 取 top_n。
    """
    df = df.copy()

    # 步骤 1: 7d 正动量
    df = df[df["percent_change_7d"] > 0].sort_values("percent_change_7d", ascending=False)

    # 步骤 2-4: 粗筛（静态数据，策略用实时价格精算）
    df = df[df["market_cap"] <= max_market_cap * 2]
    df = df[df["volume_24h"] >= min_volume]
    df = df[df["total_supply"] > 0]

    cutoff_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=min_age_days)
    df["first_data_at_dt"] = pd.to_datetime(df["first_data_at"], errors="coerce", utc=True)
    df = df[df["first_data_at_dt"] <= cutoff_date]
    df = df.drop(columns=["first_data_at_dt"])

    # 步骤 5: 映射到 Binance
    df = map_to_binance_pairs(df)

    # 步骤 6: 取前 N
    df = df.head(top_n).reset_index(drop=True)
    logger.info(
        f"小市值候选池: {len(df)} 个 (粗筛市值≤{max_market_cap * 2:,.0f}, "
        f"成交量≥{min_volume:,.0f}, 上线≥{min_age_days}天)"
    )
    return df


if __name__ == "__main__":
    # 快速测试
    df = fetch_ohlcv_ccxt("BTC/USDT", "1h", "2024-01-01", "2024-02-01")
    print(df.head())
    print(df.tail())
