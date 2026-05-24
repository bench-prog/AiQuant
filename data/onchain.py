"""
BTC 链上数据获取模块。

数据源: Blockchain.com Charts API (免费，无需 API Key)
指标: 交易笔数、活跃地址数、链上交易量

用法:
    from data.onchain import fetch_btc_onchain
    df = fetch_btc_onchain()
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Blockchain.com chart IDs
CHARTS = {
    "n_transactions": "n-transactions",
    "active_addresses": "n-unique-addresses",
    "txn_volume_usd": "estimated-transaction-volume-usd",
}

BASE_URL = "https://api.blockchain.info/charts"


def _fetch_chart(chart_id: str, timespan: str = "5years") -> list[dict]:
    """从 Blockchain.com API 拉取单个指标的时间序列。"""
    url = f"{BASE_URL}/{chart_id}"
    params = {"timespan": timespan, "format": "json"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("values", [])


def fetch_btc_onchain(
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取 BTC 链上数据（日频）。

    Returns:
        DataFrame with columns: date, n_transactions, active_addresses, txn_volume_usd
    """
    cache_path = CACHE_DIR / "btc_onchain_daily.parquet"

    if use_cache and cache_path.exists():
        logger.info("Loading BTC on-chain data from cache.")
        return pd.read_parquet(cache_path)

    logger.info("Fetching BTC on-chain data from blockchain.com...")
    df = None

    for col, chart_id in CHARTS.items():
        try:
            values = _fetch_chart(chart_id)
            ts = pd.DataFrame(values)
            ts = ts.rename(columns={"x": "date", "y": col})
            ts["date"] = pd.to_datetime(ts["date"], unit="s")
            ts = ts.set_index("date")[col]

            if df is None:
                df = ts.to_frame()
            else:
                df[col] = ts
        except Exception as e:
            logger.warning(f"Failed to fetch {chart_id}: {e}")

    if df is None or df.empty:
        raise RuntimeError("Failed to fetch any BTC on-chain data.")

    df = df.reset_index()
    df.to_parquet(cache_path, index=False)
    logger.info(f"BTC on-chain data cached: {len(df)} daily rows.")
    return df


def merge_onchain_into_ohlcv(
    ohlcv: pd.DataFrame,
    onchain: pd.DataFrame,
) -> pd.DataFrame:
    """将日频链上数据合并到 OHLCV DataFrame（1h）。

    使用前向填充 + 1 天偏移（避免未来信息泄漏）。
    今天的 OHLCV 数据只能使用昨天的链上数据。
    """
    df = ohlcv.copy()
    if "date" not in df.columns or "date" not in onchain.columns:
        return df

    onchain = onchain.copy()
    onchain["date"] = pd.to_datetime(onchain["date"], utc=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["_date_only"] = df["date"].dt.normalize()

    # 将链上数据偏移 1 天（今天只用昨天的数据）
    onchain["_merge_date"] = onchain["date"] + pd.Timedelta(days=1)

    # 重命名指标列
    rename_map = {}
    for col in onchain.columns:
        if col not in ("date", "_merge_date"):
            rename_map[col] = f"btc_{col}"
    onchain = onchain.rename(columns=rename_map)

    # 合并：每个 OHLCV 行匹配对应日期偏移后的链上数据
    for col in rename_map.values():
        df = pd.merge(
            df, onchain[["_merge_date", col]],
            left_on="_date_only", right_on="_merge_date",
            how="left",
        )
        df = df.drop(columns=["_merge_date"], errors="ignore")
        # 前向填充（日内所有小时用同一个日频值）
        df[col] = df[col].ffill()

    df = df.drop(columns=["_date_only"], errors="ignore")

    # 计算衍生特征
    for col in ["btc_n_transactions", "btc_active_addresses", "btc_txn_volume_usd"]:
        if col in df.columns:
            df[f"{col}_chg"] = df[col].pct_change()
            df[f"{col}_ma7"] = df[col].rolling(7).mean()

    return df
