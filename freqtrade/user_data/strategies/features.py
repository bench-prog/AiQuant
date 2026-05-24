"""
AiQuant 共享特征工程模块。

训练脚本和 Freqtrade 策略共用本模块，确保训练/回测特征一致。
纯 pandas/numpy 实现，不依赖外部 TA 库。
"""

import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# 默认特征参数配置
# ---------------------------------------------------------------------------
# 所有 add_*_features() 的默认参数来源。
# 训练脚本和策略可以通过 build_all_features(df, config=custom_config) 传入自定义配置。
FEATURE_PARAMS = {
    "ema_lengths": [12, 26, 50],
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "adx_length": 14,
    "rsi_lengths": [14, 6],
    "stoch": {"k": 14, "d": 3},
    "cci_length": 20,
    "williams_r_length": 14,
    "mom_length": 10,
    "atr_length": 14,
    "bbands": {"length": 20, "std": 2.0},
    "volume_sma_length": 20,
    "lag_periods": [1, 2, 3, 5, 10],
    "return_windows": {"short": 6, "long": 24},
    "volatility_window": 12,
    "funding_rate_ema_length": 8,
    "oi_ema_lengths": [12, 24],
    "oi_change_periods": [1, 6, 24],
}

# 配置目录：支持本地开发和 Docker 环境
_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "features"

_DEFAULT_GROUPS = [
    "trend", "momentum", "volatility", "volume",
    "candle", "lag", "time", "return",
    "funding_rate", "open_interest",
]


def load_feature_config(path: str | Path | dict | None = None) -> dict:
    """加载特征配置。

    Args:
        path: 配置来源。
            - None: 返回 FEATURE_PARAMS。
            - dict: 直接使用。
            - str/Path: YAML 文件路径（相对路径在 configs/features/ 下查找）。

    Returns:
        完整配置字典（含 parameters + enabled_groups）。
    """
    if path is None:
        return {"parameters": FEATURE_PARAMS.copy(), "enabled_groups": _DEFAULT_GROUPS.copy()}
    if isinstance(path, dict):
        return path.copy()

    p = Path(path)
    if not p.is_absolute():
        # 尝试多个候选路径（本地开发 + Docker）
        candidates = [
            _CONFIG_DIR / p,
            _CONFIG_DIR / f"{p}.yml",
            _CONFIG_DIR / f"{p}.yaml",
        ]
        for cand in candidates:
            if cand.exists():
                p = cand
                break
        else:
            raise FileNotFoundError(f"Feature config not found: {path}")

    with open(p, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


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


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
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


def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = series.rolling(window=length).mean()
    sigma = series.rolling(window=length).std()
    upper = middle + std * sigma
    lower = middle - std * sigma
    return lower, middle, upper


def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
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


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Williams %R — 与 Stochastic 类似但刻度为 [-100, 0]。"""
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)


def mom(series: pd.Series, length: int = 10) -> pd.Series:
    """价格动量 — 当前价格与 N 周期前价格的差值。"""
    return series - series.shift(length)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff()) * volume).cumsum()


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    return (tp * df["volume"]).cumsum() / df["volume"].cumsum()


def add_trend_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = params or FEATURE_PARAMS
    df = df.copy()
    for length in params.get("ema_lengths", [12, 26, 50]):
        df[f"ema_{length}"] = ema(df["close"], length)
    macd_cfg = params.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    macd_line, macd_sig, macd_hist = macd(
        df["close"], macd_cfg["fast"], macd_cfg["slow"], macd_cfg["signal"]
    )
    df["macd"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist
    adx_len = params.get("adx_length", 14)
    adx_val, plus_di, minus_di = adx(df["high"], df["low"], df["close"], adx_len)
    df[f"adx_{adx_len}"] = adx_val
    df[f"plus_di_{adx_len}"] = plus_di
    df[f"minus_di_{adx_len}"] = minus_di
    return df


def add_momentum_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = params or FEATURE_PARAMS
    df = df.copy()
    for length in params.get("rsi_lengths", [14, 6]):
        df[f"rsi_{length}"] = rsi(df["close"], length)
    stoch_cfg = params.get("stoch", {"k": 14, "d": 3})
    stoch_k, stoch_d = stoch(
        df["high"], df["low"], df["close"], stoch_cfg["k"], stoch_cfg["d"]
    )
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    cci_len = params.get("cci_length", 20)
    df[f"cci_{cci_len}"] = cci(df["high"], df["low"], df["close"], cci_len)
    wr_len = params.get("williams_r_length", 14)
    df[f"williams_r_{wr_len}"] = williams_r(df["high"], df["low"], df["close"], wr_len)
    mom_len = params.get("mom_length", 10)
    df[f"mom_{mom_len}"] = mom(df["close"], mom_len)
    return df


def add_volatility_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = params or FEATURE_PARAMS
    df = df.copy()
    atr_len = params.get("atr_length", 14)
    df[f"atr_{atr_len}"] = atr(df["high"], df["low"], df["close"], atr_len)
    bb_cfg = params.get("bbands", {"length": 20, "std": 2.0})
    lower, middle, upper = bbands(df["close"], bb_cfg["length"], bb_cfg["std"])
    df["bb_lower"] = lower
    df["bb_middle"] = middle
    df["bb_upper"] = upper
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    bb_range = df["bb_upper"] - df["bb_lower"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
    return df


def add_volume_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = params or FEATURE_PARAMS
    df = df.copy()
    vol_sma = params.get("volume_sma_length", 20)
    df[f"volume_sma_{vol_sma}"] = df["volume"].rolling(window=vol_sma).mean()
    df["volume_ratio"] = df["volume"] / df[f"volume_sma_{vol_sma}"]
    df["obv"] = obv(df["close"], df["volume"])
    df["vwap"] = vwap(df)
    df["obv_change_1h"] = df["obv"].diff(1)
    vwap_safe = df["vwap"].replace(0, np.nan)
    df["vwap_distance"] = (df["close"] - df["vwap"]) / vwap_safe
    return df


def add_candle_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    df = df.copy()
    # 复用 add_trend_features 已生成的 ema 列，避免重复计算
    ema_12 = df["ema_12"] if "ema_12" in df.columns else ema(df["close"], 12)
    ema_26 = df["ema_26"] if "ema_26" in df.columns else ema(df["close"], 26)
    df["close_above_ema12"] = (df["close"] > ema_12).astype(int)
    df["close_above_ema26"] = (df["close"] > ema_26).astype(int)
    df["body_pct"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-9)
    df["upper_wick_pct"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"] + 1e-9)
    df["lower_wick_pct"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"] + 1e-9)
    return df


def add_lag_features(df: pd.DataFrame, params: dict = None, lags: list[int] = None) -> pd.DataFrame:
    if lags is not None:
        lag_periods = lags
    else:
        params = params or FEATURE_PARAMS
        lag_periods = params.get("lag_periods", [1, 2, 3, 5, 10])
    df = df.copy()
    for lag in lag_periods:
        df[f"return_lag_{lag}"] = df["close"].pct_change(lag)
        df[f"volume_lag_{lag}"] = df["volume"].shift(lag)
    return df


def add_time_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_return_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    params = params or FEATURE_PARAMS
    df = df.copy()
    ret_cfg = params.get("return_windows", {"short": 6, "long": 24})
    df[f"return_{ret_cfg['short']}h"] = df["close"].pct_change(ret_cfg["short"])
    df[f"return_{ret_cfg['long']}h"] = df["close"].pct_change(ret_cfg["long"])
    vol_win = params.get("volatility_window", 12)
    df[f"volatility_{vol_win}h"] = df["close"].pct_change().rolling(vol_win).std()
    return df


def add_funding_rate_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    资金费率相关特征。

    要求 df 包含 'fundingRate' 列（已通过 funding rate 数据 merge）。
    若列不存在则原样返回 df，策略仍可正常运行。
    """
    if "fundingRate" not in df.columns:
        return df

    params = params or FEATURE_PARAMS
    df = df.copy()
    df["funding_rate"] = df["fundingRate"]
    ema_len = params.get("funding_rate_ema_length", 8)
    df[f"funding_rate_ema_{ema_len}"] = ema(df["funding_rate"], ema_len)
    df["funding_rate_sign"] = np.sign(df["funding_rate"])
    df["funding_rate_change"] = df["funding_rate"].diff()
    return df


def add_open_interest_features(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    持仓量相关特征。

    要求 df 包含 'openInterest' 列（已通过 OI 数据 merge）。
    若列不存在则原样返回 df，策略仍可正常运行。
    """
    if "openInterest" not in df.columns:
        return df

    params = params or FEATURE_PARAMS
    df = df.copy()
    # 对数缩放处理绝对值过大的问题
    df["open_interest"] = np.log1p(df["openInterest"])
    for length in params.get("oi_ema_lengths", [12, 24]):
        df[f"oi_ema_{length}"] = ema(df["open_interest"], length)
    for period in params.get("oi_change_periods", [1, 6, 24]):
        df[f"oi_change_{period}h"] = df["open_interest"].diff(period)

    # OI 速度: OI 变化率除以成交量，反映持仓建立/清算的强度
    volume_safe = df["volume"].replace(0, np.nan)
    df["oi_velocity"] = df["oi_change_1h"] / volume_safe
    return df


def build_all_features(df: pd.DataFrame, config: str | Path | dict | None = None) -> pd.DataFrame:
    """按顺序构建全部特征。

    Args:
        config: 配置来源。
            - None: 使用 FEATURE_PARAMS 默认配置。
            - dict: 直接传入参数字典。
            - str/Path: YAML 配置文件路径（如 "default"、"minimal"）。
    """
    cfg = load_feature_config(config)
    params = cfg.get("parameters", {})
    groups = cfg.get("enabled_groups", _DEFAULT_GROUPS)
    return DEFAULT_REGISTRY.compute(df, feature_names=groups, config=params)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """从 DataFrame 中剔除基础 OHLCV 列，返回特征列名列表。"""
    base_cols = {"open", "high", "low", "close", "volume", "date"}
    return [c for c in df.columns if c not in base_cols]


def add_higher_timeframe_features(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame | None = None,
    df_1d: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """将大时间框架特征合并到主时间框架（1h）。

    大时间框架只计算核心趋势/动量/波动率指标，通过前向填充对齐到 1h。
    关键：高层蜡烛的时间戳是周期起点，数据到周期结尾才已知。
    因此将时间戳偏移一个周期长度，确保 ffill 只用已完成蜡烛的数据，
    避免未来信息泄漏。
    新增列名格式: {original_col}_{tf}，如 ema_12_4h、rsi_14_1d。
    """
    _base_cols = {"open", "high", "low", "close", "volume", "date"}
    df = df_1h.copy()

    if df_4h is not None and len(df_4h) > 0:
        # 4h: 趋势 + 动量 + 波动率
        df_4h = add_trend_features(df_4h)
        df_4h = add_momentum_features(df_4h)
        df_4h = add_volatility_features(df_4h)
        df_4h = df_4h.set_index("date")
        # 将时间戳从周期起点偏移到周期终点（+4h），确保只用已完成蜡烛
        df_4h.index = df_4h.index + pd.Timedelta(hours=4)
        for col in df_4h.columns:
            if col not in _base_cols:
                df[f"{col}_4h"] = (
                    df_4h[col]
                    .reindex(df["date"], method="ffill")
                    .values
                )

    if df_1d is not None and len(df_1d) > 0:
        # 1d: 趋势 + 动量
        df_1d = add_trend_features(df_1d)
        df_1d = add_momentum_features(df_1d)
        df_1d = df_1d.set_index("date")
        # 将时间戳从周期起点偏移到周期终点（+24h），确保只用已完成蜡烛
        df_1d.index = df_1d.index + pd.Timedelta(days=1)
        for col in df_1d.columns:
            if col not in _base_cols:
                df[f"{col}_1d"] = (
                    df_1d[col]
                    .reindex(df["date"], method="ffill")
                    .values
                )

    return df


# ---------------------------------------------------------------------------
# 特征注册表（阶段 2）
# ---------------------------------------------------------------------------

class FeatureRegistry:
    """特征注册表 — 按需计算 + 元数据管理。"""

    def __init__(self):
        self._features: dict[str, dict] = {}

    def register(
        self,
        name: str,
        func: callable,
        category: str,
        params: dict | None = None,
    ) -> None:
        """注册一个特征生成函数。"""
        self._features[name] = {
            "func": func,
            "category": category,
            "params": params,
        }

    def compute(
        self,
        df: pd.DataFrame,
        feature_names: list[str] | None = None,
        config: dict | None = None,
    ) -> pd.DataFrame:
        """按需计算指定特征组。"""
        names = feature_names or list(self._features.keys())
        for name in names:
            meta = self._features[name]
            cfg = config or meta.get("params")
            if cfg is not None:
                df = meta["func"](df, cfg)
            else:
                df = meta["func"](df)
        return df

    def list_features(self, category: str | None = None) -> list[str]:
        """列出已注册的特征名。"""
        if category is None:
            return list(self._features.keys())
        return [k for k, v in self._features.items() if v["category"] == category]

    def get_categories(self) -> list[str]:
        """列出所有类别。"""
        return sorted(set(v["category"] for v in self._features.values()))

    def __contains__(self, name: str) -> bool:
        return name in self._features


# 预构建默认注册表
DEFAULT_REGISTRY = FeatureRegistry()
DEFAULT_REGISTRY.register("trend", add_trend_features, "trend")
DEFAULT_REGISTRY.register("momentum", add_momentum_features, "momentum")
DEFAULT_REGISTRY.register("volatility", add_volatility_features, "volatility")
DEFAULT_REGISTRY.register("volume", add_volume_features, "volume")
DEFAULT_REGISTRY.register("candle", add_candle_features, "candle")
DEFAULT_REGISTRY.register("lag", add_lag_features, "lag")
DEFAULT_REGISTRY.register("time", add_time_features, "time")
DEFAULT_REGISTRY.register("return", add_return_features, "return")
DEFAULT_REGISTRY.register("funding_rate", add_funding_rate_features, "funding_rate")
DEFAULT_REGISTRY.register("open_interest", add_open_interest_features, "open_interest")
