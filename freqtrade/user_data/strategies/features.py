"""
AiQuant 共享特征工程模块。

训练脚本和 Freqtrade 策略共用本模块，确保训练/回测特征一致。
纯 pandas/numpy 实现，不依赖外部 TA 库。
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


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    平均趋向指数 (ADX) 与 +DI / -DI。

    Returns:
        adx: 趋势强度 (0-100)
        plus_di: 正向趋向指标
        minus_di: 负向趋向指标
    """
    # 真实波幅 TR
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # +DM 与 -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    # Wilder 平滑 (RMA)
    alpha = 1.0 / length
    tr_smooth = tr.ewm(alpha=alpha, min_periods=length).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, min_periods=length).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, min_periods=length).mean()

    # +DI / -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth

    # DX 与 ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_series = dx.ewm(alpha=alpha, min_periods=length).mean()

    return adx_series, plus_di, minus_di


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


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
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


def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_6h"] = df["close"].pct_change(6)
    df["return_24h"] = df["close"].pct_change(24)
    df["volatility_12h"] = df["close"].pct_change().rolling(12).std()
    return df


def add_funding_rate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    资金费率相关特征。

    要求 df 包含 'fundingRate' 列（已通过 funding rate 数据 merge）。
    若列不存在则原样返回 df，策略仍可正常运行。
    """
    if "fundingRate" not in df.columns:
        return df

    df = df.copy()
    df["funding_rate"] = df["fundingRate"]
    df["funding_rate_ema_8"] = ema(df["funding_rate"], 8)
    df["funding_rate_sign"] = np.sign(df["funding_rate"])
    df["funding_rate_change"] = df["funding_rate"].diff()
    return df


def add_open_interest_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    持仓量相关特征。

    要求 df 包含 'openInterest' 列（已通过 OI 数据 merge）。
    若列不存在则原样返回 df，策略仍可正常运行。
    """
    if "openInterest" not in df.columns:
        return df

    df = df.copy()
    # 对数缩放处理绝对值过大的问题
    df["open_interest"] = np.log1p(df["openInterest"])
    df["oi_ema_12"] = ema(df["open_interest"], 12)
    df["oi_ema_24"] = ema(df["open_interest"], 24)
    df["oi_change_1h"] = df["open_interest"].diff(1)
    df["oi_change_6h"] = df["open_interest"].diff(6)
    df["oi_change_24h"] = df["open_interest"].diff(24)

    # OI 速度: OI 变化率除以成交量，反映持仓建立/清算的强度
    volume_safe = df["volume"].replace(0, np.nan)
    df["oi_velocity"] = df["oi_change_1h"] / volume_safe
    return df


def build_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """按顺序构建全部特征。"""
    df = add_trend_features(df)
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_candle_features(df)
    df = add_lag_features(df)
    df = add_time_features(df)
    df = add_return_features(df)
    df = add_funding_rate_features(df)
    df = add_open_interest_features(df)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """从 DataFrame 中剔除基础 OHLCV 列，返回特征列名列表。"""
    base_cols = {"open", "high", "low", "close", "volume", "date"}
    return [c for c in df.columns if c not in base_cols]
