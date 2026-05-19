"""Momentum filters for SmallCapMomentumStrategy. Pure pandas/numpy."""

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def add_trend_filter(df: pd.DataFrame, length: int = 30) -> pd.DataFrame:
    """Add EMA trend filter."""
    df = df.copy()
    df["ema_30"] = ema(df["close"], length)
    df["above_ema30"] = (df["close"] > df["ema_30"]).astype(int)
    return df


def add_momentum_filter(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Add RSI filter."""
    df = df.copy()
    df["rsi_14"] = rsi(df["close"], length)
    return df


def add_consecutive_down_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate consecutive down days (close < previous close).
    Returns the number of consecutive down candles ending at the current candle.
    """
    df = df.copy()
    down = (df["close"] < df["close"].shift(1)).astype(int)
    arr = down.to_numpy(dtype=int)
    result = np.zeros_like(arr, dtype=int)
    count = 0
    for i in range(len(arr)):
        if arr[i] == 1:
            count += 1
        else:
            count = 0
        result[i] = count
    df["consecutive_down_days"] = result
    return df


def apply_all_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = add_trend_filter(df)
    df = add_momentum_filter(df)
    df = add_consecutive_down_days(df)
    return df
