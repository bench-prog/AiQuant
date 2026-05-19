"""
Feature Engineering Module for AiQuant

Provides reusable feature transformers for crypto OHLCV data.
All functions accept a pandas DataFrame and return a DataFrame with added columns.
"""

import pandas as pd
import pandas_ta as pta


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """EMA, MACD trend indicators."""
    df = df.copy()
    df["ema_12"] = pta.ema(df["close"], length=12)
    df["ema_26"] = pta.ema(df["close"], length=26)
    df["ema_50"] = pta.ema(df["close"], length=50)
    macd_df = pta.macd(df["close"], fast=12, slow=26, signal=9)
    df["macd"] = macd_df["MACD_12_26_9"]
    df["macd_signal"] = macd_df["MACDs_12_26_9"]
    df["macd_hist"] = macd_df["MACDh_12_26_9"]
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, Stochastic, CCI."""
    df = df.copy()
    df["rsi_14"] = pta.rsi(df["close"], length=14)
    df["rsi_6"] = pta.rsi(df["close"], length=6)
    stoch_df = pta.stoch(df["high"], df["low"], df["close"], k=14, d=3)
    df["stoch_k"] = stoch_df["STOCHk_14_3_3"]
    df["stoch_d"] = stoch_df["STOCHd_14_3_3"]
    df["cci_20"] = pta.cci(df["high"], df["low"], df["close"], length=20)
    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """ATR, Bollinger Bands, Keltner Channels."""
    df = df.copy()
    df["atr_14"] = pta.atr(df["high"], df["low"], df["close"], length=14)
    bbands_df = pta.bbands(df["close"], length=20, std=2)
    df["bb_upper"] = bbands_df["BBU_20_2.0"]
    df["bb_middle"] = bbands_df["BBM_20_2.0"]
    df["bb_lower"] = bbands_df["BBL_20_2.0"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume SMA, OBV, VWAP."""
    df = df.copy()
    df["volume_sma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["obv"] = pta.obv(df["close"], df["volume"])
    df["vwap"] = pta.vwap(df["high"], df["low"], df["close"], df["volume"])
    return df


def add_price_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Price relative to moving averages, body size, wicks."""
    df = df.copy()
    df["close_above_ema12"] = (df["close"] > pta.ema(df["close"], length=12)).astype(int)
    df["close_above_ema26"] = (df["close"] > pta.ema(df["close"], length=26)).astype(int)
    df["body_pct"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_wick_pct"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick_pct"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = None) -> pd.DataFrame:
    """Lagged returns and volume."""
    if lags is None:
        lags = [1, 2, 3, 5, 10]
    df = df.copy()
    for lag in lags:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature pipelines."""
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_price_structure(df)
    df = add_lag_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return list of feature columns (exclude OHLCV and target)."""
    base_cols = {"open", "high", "low", "close", "volume", "date"}
    return [c for c in df.columns if c not in base_cols]
