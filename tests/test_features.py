"""
AiQuant features.py 单元测试。

覆盖核心指标函数和特征组合函数。
使用合成数据，不依赖外部 API。
"""

import numpy as np
import pandas as pd
import pytest

from features import (
    adx,
    atr,
    bbands,
    build_all_features,
    cci,
    ema,
    get_feature_columns,
    macd,
    obv,
    rsi,
    stoch,
    vwap,
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
