"""
Shared feature engineering for AiQuant.
Used by both training scripts (via path insertion) and Freqtrade strategy.
Pure pandas/numpy — no external TA libraries.
"""

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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def bbands(series: pd.Series, length: int = 20, std: float = 2.0):
    middle = series.rolling(window=length).mean()
    sigma = series.rolling(window=length).std()
    upper = middle + std * sigma
    lower = middle - std * sigma
    return lower, middle, upper


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3):
    lowest_low = low.rolling(window=k).min()
    highest_high = high.rolling(window=k).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d).mean()
    return stoch_k, stoch_d


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=length).mean()
    mean_dev = tp.rolling(window=length).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mean_dev)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["ema_50"] = ema(df["close"], 50)
    macd_line, macd_sig, macd_hist = macd(df["close"], 12, 26, 9)
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi_14"] = rsi(df["close"], 14)
    df["rsi_6"] = rsi(df["close"], 6)
    stoch_k, stoch_d = stoch(df["high"], df["low"], df["close"], 14, 3)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    df["cci_20"] = cci(df["high"], df["low"], df["close"], 20)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)
    lower, middle, upper = bbands(df["close"], 20, 2.0)
    df["bb_lower"] = lower
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["obv"] = obv(df["close"], df["volume"])
    df["vwap"] = vwap(df)
    return df


def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["close_above_ema12"] = (df["close"] > ema(df["close"], 12)).astype(int)
    df["close_above_ema26"] = (df["close"] > ema(df["close"], 26)).astype(int)
    df["body_pct"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_wick_pct"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick_pct"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    df = df.copy()
    for lag in lags:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_crypto_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_6h"] = df["close"].pct_change(6)
    df["return_24h"] = df["close"].pct_change(24)
    df["volatility_12h"] = df["close"].pct_change().rolling(12).std()
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_structure(df)
    df = add_lag_features(df)
    df = add_time_features(df)
    df = add_crypto_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    base_cols = {"open", "high", "low", "close", "volume", "date"}
    return [c for c in df.columns if c not in base_cols]
