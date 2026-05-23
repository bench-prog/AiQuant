"""
AiQuant features.py 单元测试。

覆盖核心指标函数和特征组合函数。
使用合成数据，不依赖外部 API。
"""

import numpy as np
import pandas as pd

from features import (
    DEFAULT_REGISTRY,
    add_candle_features,
    load_feature_config,
    add_funding_rate_features,
    add_lag_features,
    add_momentum_features,
    add_open_interest_features,
    add_return_features,
    add_time_features,
    add_trend_features,
    add_volatility_features,
    add_volume_features,
    adx,
    atr,
    bbands,
    build_all_features,
    cci,
    ema,
    get_feature_columns,
    macd,
    mom,
    obv,
    rsi,
    stoch,
    vwap,
    williams_r,
)


# ---------------------------------------------------------------------------
# 核心指标函数测试
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA 长度参数影响平滑程度。"""
        ema_short = ema(sample_ohlcv["close"], 5)
        ema_long = ema(sample_ohlcv["close"], 50)
        # 短期 EMA 波动更大（对价格变化更敏感）
        assert ema_short.std() > ema_long.std()

    def test_ema_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        assert isinstance(ema(sample_ohlcv["close"], 12), pd.Series)


class TestRSI:
    def test_rsi_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """RSI 必须在 [0, 100] 范围内。"""
        result = rsi(sample_ohlcv["close"], 14)
        assert result.min() >= 0
        assert result.max() <= 100

    def test_rsi_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """不同 length 产生不同结果。"""
        rsi14 = rsi(sample_ohlcv["close"], 14)
        rsi6 = rsi(sample_ohlcv["close"], 6)
        # 短期 RSI 波动更大
        assert rsi6.std() > rsi14.std()


class TestMACD:
    def test_macd_returns_three_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """MACD 返回 macd_line, signal, hist 三个 Series。"""
        line, signal, hist = macd(sample_ohlcv["close"])
        assert isinstance(line, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(hist, pd.Series)

    def test_macd_hist_equals_line_minus_signal(self, sample_ohlcv: pd.DataFrame) -> None:
        """hist = macd_line - signal。"""
        line, signal, hist = macd(sample_ohlcv["close"])
        pd.testing.assert_series_equal(hist, line - signal, check_names=False)


class TestATR:
    def test_atr_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """ATR 必须为正数。"""
        result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert (result.dropna() > 0).all()

    def test_atr_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """不同 length 产生不同平滑程度。"""
        atr5 = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 5)
        atr20 = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 20)
        assert atr5.std() > atr20.std()


class TestADX:
    def test_adx_returns_three_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX 返回 adx, plus_di, minus_di 三个 Series。"""
        adx_val, plus_di, minus_di = adx(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        assert isinstance(adx_val, pd.Series)
        assert isinstance(plus_di, pd.Series)
        assert isinstance(minus_di, pd.Series)

    def test_adx_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX 在 [0, 100] 范围内。"""
        adx_val, _, _ = adx(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert adx_val.min() >= 0
        assert adx_val.max() <= 100

    def test_di_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """+DI 和 -DI 在 [0, 100] 范围内。"""
        _, plus_di, minus_di = adx(
            sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"]
        )
        assert plus_di.min() >= 0
        assert plus_di.max() <= 100
        assert minus_di.min() >= 0
        assert minus_di.max() <= 100


class TestBBands:
    def test_bbands_returns_three_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """布林带返回 lower, middle, upper 三个 Series。"""
        lower, middle, upper = bbands(sample_ohlcv["close"])
        assert isinstance(lower, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(upper, pd.Series)

    def test_bbands_order(self, sample_ohlcv: pd.DataFrame) -> None:
        """lower <= middle <= upper。"""
        lower, middle, upper = bbands(sample_ohlcv["close"])
        valid = lower.notna() & middle.notna() & upper.notna()
        assert (lower[valid] <= middle[valid]).all()
        assert (middle[valid] <= upper[valid]).all()


class TestStoch:
    def test_stoch_returns_two_series(self, sample_ohlcv: pd.DataFrame) -> None:
        """随机指标返回 %K 和 %D 两个 Series。"""
        k, d = stoch(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert isinstance(k, pd.Series)
        assert isinstance(d, pd.Series)

    def test_stoch_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """%K 和 %D 在 [0, 100] 范围内。"""
        k, d = stoch(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert k.min() >= 0
        assert k.max() <= 100
        assert d.min() >= 0
        assert d.max() <= 100


class TestWilliamsR:
    def test_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """Williams %R 必须在 [-100, 0] 范围内。"""
        result = williams_r(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert result.min() >= -100
        assert result.max() <= 0

    def test_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        result = williams_r(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert isinstance(result, pd.Series)


class TestMOM:
    def test_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        result = mom(sample_ohlcv["close"])
        assert isinstance(result, pd.Series)

    def test_length(self, sample_ohlcv: pd.DataFrame) -> None:
        """不同 length 产生不同动量值。"""
        mom5 = mom(sample_ohlcv["close"], 5)
        mom20 = mom(sample_ohlcv["close"], 20)
        assert mom5.std() != mom20.std()


class TestCCI:
    def test_cci_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        result = cci(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert isinstance(result, pd.Series)


class TestOBV:
    def test_obv_cumulative(self, sample_ohlcv: pd.DataFrame) -> None:
        """OBV 是累积值，应该单调或稳定变化。"""
        result = obv(sample_ohlcv["close"], sample_ohlcv["volume"])
        assert isinstance(result, pd.Series)
        # OBV 应该非零（有成交量）
        assert result.abs().sum() > 0


class TestVWAP:
    def test_vwap_is_series(self, sample_ohlcv: pd.DataFrame) -> None:
        result = vwap(sample_ohlcv)
        assert isinstance(result, pd.Series)

    def test_vwap_positive(self, sample_ohlcv: pd.DataFrame) -> None:
        """VWAP 应该为正数。"""
        result = vwap(sample_ohlcv)
        assert (result.dropna() > 0).all()


# ---------------------------------------------------------------------------
# 特征组合函数集成测试
# ---------------------------------------------------------------------------


class TestAddTrendFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_trend_features(sample_ohlcv)
        assert "ema_12" in df.columns
        assert "ema_26" in df.columns
        assert "ema_50" in df.columns
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_hist" in df.columns
        assert "adx_14" in df.columns
        assert "plus_di_14" in df.columns
        assert "minus_di_14" in df.columns

    def test_adx_range(self, sample_ohlcv: pd.DataFrame) -> None:
        """ADX 特征列值域验证。"""
        df = add_trend_features(sample_ohlcv)
        assert df["adx_14"].min() >= 0
        assert df["adx_14"].max() <= 100
        assert df["plus_di_14"].min() >= 0
        assert df["plus_di_14"].max() <= 100
        assert df["minus_di_14"].min() >= 0
        assert df["minus_di_14"].max() <= 100


class TestAddMomentumFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_momentum_features(sample_ohlcv)
        assert "rsi_14" in df.columns
        assert "rsi_6" in df.columns
        assert "stoch_k" in df.columns
        assert "stoch_d" in df.columns
        assert "cci_20" in df.columns
        assert "williams_r_14" in df.columns
        assert "mom_10" in df.columns

    def test_williams_r_range(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_momentum_features(sample_ohlcv)
        assert df["williams_r_14"].min() >= -100
        assert df["williams_r_14"].max() <= 0


class TestAddVolatilityFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_volatility_features(sample_ohlcv)
        assert "atr_14" in df.columns
        assert "bb_lower" in df.columns
        assert "bb_middle" in df.columns
        assert "bb_upper" in df.columns
        assert "bb_width" in df.columns
        assert "bb_position" in df.columns

    def test_bb_position_finite(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_volatility_features(sample_ohlcv)
        valid = df["bb_position"].dropna()
        # close 可能突破布林带，所以 bb_position 不一定在 [0, 1]，但必须在有限范围
        assert np.isfinite(valid).all()


class TestAddVolumeFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_volume_features(sample_ohlcv)
        assert "volume_sma_20" in df.columns
        assert "volume_ratio" in df.columns
        assert "obv" in df.columns
        assert "vwap" in df.columns
        assert "obv_change_1h" in df.columns
        assert "vwap_distance" in df.columns


class TestAddCandleFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_candle_features(sample_ohlcv)
        assert "close_above_ema12" in df.columns
        assert "close_above_ema26" in df.columns
        assert "body_pct" in df.columns
        assert "upper_wick_pct" in df.columns
        assert "lower_wick_pct" in df.columns

    def test_binary_values(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_candle_features(sample_ohlcv)
        assert set(df["close_above_ema12"].unique()).issubset({0, 1})
        assert set(df["close_above_ema26"].unique()).issubset({0, 1})

    def test_pct_range(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_candle_features(sample_ohlcv)
        assert (df["body_pct"].dropna() >= 0).all() and (df["body_pct"].dropna() <= 1).all()
        assert (df["upper_wick_pct"].dropna() >= 0).all() and (df["upper_wick_pct"].dropna() <= 1).all()
        assert (df["lower_wick_pct"].dropna() >= 0).all() and (df["lower_wick_pct"].dropna() <= 1).all()


class TestAddLagFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_lag_features(sample_ohlcv)
        expected = [f"return_lag_{i}" for i in [1, 2, 3, 5, 10]]
        expected += [f"volume_lag_{i}" for i in [1, 2, 3, 5, 10]]
        for col in expected:
            assert col in df.columns

    def test_custom_lags(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_lag_features(sample_ohlcv, lags=[1, 3])
        assert "return_lag_1" in df.columns
        assert "return_lag_3" in df.columns
        assert "return_lag_2" not in df.columns


class TestAddTimeFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_time_features(sample_ohlcv)
        assert "hour" in df.columns
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns

    def test_hour_range(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_time_features(sample_ohlcv)
        assert (df["hour"] >= 0).all() and (df["hour"] <= 23).all()


class TestAddReturnFeatures:
    def test_columns(self, sample_ohlcv: pd.DataFrame) -> None:
        df = add_return_features(sample_ohlcv)
        assert "return_6h" in df.columns
        assert "return_24h" in df.columns
        assert "volatility_12h" in df.columns


class TestAddFundingRateFeatures:
    def test_with_funding(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df = add_funding_rate_features(sample_ohlcv_with_funding)
        assert "funding_rate" in df.columns
        assert "funding_rate_ema_8" in df.columns
        assert "funding_rate_sign" in df.columns
        assert "funding_rate_change" in df.columns

    def test_without_funding(self, sample_ohlcv: pd.DataFrame) -> None:
        """无 fundingRate 列时原样返回。"""
        original_cols = set(sample_ohlcv.columns)
        df = add_funding_rate_features(sample_ohlcv)
        assert set(df.columns) == original_cols


class TestAddOpenInterestFeatures:
    def test_with_oi(self, sample_ohlcv_with_oi: pd.DataFrame) -> None:
        df = add_open_interest_features(sample_ohlcv_with_oi)
        assert "open_interest" in df.columns
        assert "oi_ema_12" in df.columns
        assert "oi_velocity" in df.columns

    def test_without_oi(self, sample_ohlcv: pd.DataFrame) -> None:
        """无 openInterest 列时原样返回。"""
        original_cols = set(sample_ohlcv.columns)
        df = add_open_interest_features(sample_ohlcv)
        assert set(df.columns) == original_cols


class TestBuildAllFeatures:
    def test_all_columns_present(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df = build_all_features(sample_ohlcv_with_funding)
        # 核心特征列应该存在
        expected = [
            "ema_12", "rsi_14", "macd", "atr_14",
            "bb_lower", "volume_ratio", "obv", "vwap",
            "return_lag_1", "hour_sin", "return_6h",
            "funding_rate", "funding_rate_sign",
        ]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_inf(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df = build_all_features(sample_ohlcv_with_funding)
        numeric = df.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.values).any()


class TestGetFeatureColumns:
    def test_excludes_base_cols(self, sample_ohlcv: pd.DataFrame) -> None:
        cols = get_feature_columns(sample_ohlcv)
        base = {"open", "high", "low", "close", "volume", "date"}
        assert not any(c in base for c in cols)

    def test_includes_feature_cols(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df = build_all_features(sample_ohlcv_with_funding)
        cols = get_feature_columns(df)
        assert "rsi_14" in cols
        assert "macd" in cols


class TestFeatureRegistry:
    def test_default_registry_has_all_features(self) -> None:
        assert "trend" in DEFAULT_REGISTRY
        assert "momentum" in DEFAULT_REGISTRY
        assert "volatility" in DEFAULT_REGISTRY
        assert "volume" in DEFAULT_REGISTRY

    def test_list_features(self) -> None:
        features = DEFAULT_REGISTRY.list_features()
        assert "trend" in features
        assert "momentum" in features

    def test_list_features_by_category(self) -> None:
        trend_features = DEFAULT_REGISTRY.list_features(category="trend")
        assert "trend" in trend_features
        assert "momentum" not in trend_features

    def test_get_categories(self) -> None:
        cats = DEFAULT_REGISTRY.get_categories()
        assert "trend" in cats
        assert "momentum" in cats

    def test_compute_single(self, sample_ohlcv: pd.DataFrame) -> None:
        df = DEFAULT_REGISTRY.compute(sample_ohlcv.copy(), feature_names=["trend"])
        assert "ema_12" in df.columns
        assert "rsi_14" not in df.columns

    def test_compute_multiple(self, sample_ohlcv: pd.DataFrame) -> None:
        df = DEFAULT_REGISTRY.compute(
            sample_ohlcv.copy(), feature_names=["trend", "momentum"]
        )
        assert "ema_12" in df.columns
        assert "rsi_14" in df.columns
        assert "atr_14" not in df.columns

    def test_compute_all(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df = DEFAULT_REGISTRY.compute(sample_ohlcv_with_funding.copy())
        assert "ema_12" in df.columns
        assert "rsi_14" in df.columns
        assert "funding_rate" in df.columns

    def test_compute_equivalent_to_build_all(
        self, sample_ohlcv_with_funding: pd.DataFrame
    ) -> None:
        df_registry = DEFAULT_REGISTRY.compute(sample_ohlcv_with_funding.copy())
        df_build = build_all_features(sample_ohlcv_with_funding.copy())
        assert set(df_registry.columns) == set(df_build.columns)


class TestLoadFeatureConfig:
    def test_load_none_returns_default(self) -> None:
        cfg = load_feature_config(None)
        assert "parameters" in cfg
        assert "enabled_groups" in cfg
        assert "ema_lengths" in cfg["parameters"]

    def test_load_dict(self) -> None:
        custom = {"parameters": {"adx_length": 20}, "enabled_groups": ["trend"]}
        cfg = load_feature_config(custom)
        assert cfg["parameters"]["adx_length"] == 20

    def test_load_yaml_default(self) -> None:
        cfg = load_feature_config("default")
        assert "parameters" in cfg
        assert "enabled_groups" in cfg
        assert cfg["parameters"]["adx_length"] == 14

    def test_load_yaml_minimal(self) -> None:
        cfg = load_feature_config("minimal")
        assert "parameters" in cfg
        # minimal 配置 EMA 只有 12/26
        assert cfg["parameters"]["ema_lengths"] == [12, 26]

    def test_build_from_yaml(self, sample_ohlcv_with_funding: pd.DataFrame) -> None:
        df_default = build_all_features(sample_ohlcv_with_funding.copy(), config="default")
        df_none = build_all_features(sample_ohlcv_with_funding.copy())
        assert set(df_default.columns) == set(df_none.columns)

    def test_build_minimal_has_fewer_columns(
        self, sample_ohlcv_with_funding: pd.DataFrame
    ) -> None:
        df_minimal = build_all_features(sample_ohlcv_with_funding.copy(), config="minimal")
        df_default = build_all_features(sample_ohlcv_with_funding.copy(), config="default")
        # minimal 只有 2 个 EMA，default 有 3 个
        assert "ema_50" in df_default.columns
        assert "ema_50" not in df_minimal.columns
