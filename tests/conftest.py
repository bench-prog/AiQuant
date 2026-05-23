"""
AiQuant 测试共享 fixture。

提供合成市场数据，用于 features.py 的单元测试和集成测试。
所有测试使用固定随机种子，确保可复现。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """合成 OHLCV 数据，100 条 1 小时 K 线。"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    close = 100.0 + np.random.randn(100).cumsum()
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close + np.random.randn(100) * 0.5,
            "high": close + np.abs(np.random.randn(100)) * 2 + 0.1,
            "low": close - np.abs(np.random.randn(100)) * 2 - 0.1,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = df[["open", "close", "high"]].max(axis=1)
    df["low"] = df[["open", "close", "low"]].min(axis=1)
    return df


@pytest.fixture
def sample_ohlcv_with_funding(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """在合成数据上添加 fundingRate 列。"""
    df = sample_ohlcv.copy()
    df["fundingRate"] = np.random.randn(len(df)) * 0.0001
    return df


@pytest.fixture
def sample_ohlcv_with_oi(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """在合成数据上添加 openInterest 列。"""
    df = sample_ohlcv.copy()
    df["openInterest"] = np.random.randint(1_000_000, 10_000_000, len(df))
    return df
