"""
StrategyEnsemble 单元测试。

验证信号归一化、加权融合、入场/出场逻辑。
不依赖 Freqtrade 运行时，纯公式验证。
"""

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Trend signal computation (pure function, matching strategy logic)
# ---------------------------------------------------------------------------

def compute_trend_signal(df: pd.DataFrame) -> pd.Series:
    """纯函数：趋势跟踪信号计算（与策略实现一致）。"""
    ema_s = df["close"].ewm(span=10).mean()
    ema_m = df["close"].ewm(span=30).mean()
    ema_l = df["close"].ewm(span=100).mean()

    # Use pre-computed ADX if available
    if "adx_14" in df.columns:
        adx = df["adx_14"]
    else:
        high, low, close = df["high"], df["low"], df["close"]
        tr = pd.DataFrame({
            "hl": high - low,
            "hc": (high - close.shift()).abs(),
            "lc": (low - close.shift()).abs(),
        }).max(axis=1)
        atr = tr.ewm(span=14).mean()
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > 0) & (up > down), 0.0)
        minus_dm = down.where((down > 0) & (down > up), 0.0)
        plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=14).mean()

    trend_up = (ema_s > ema_m) & (ema_m > ema_l)
    price_ok = df["close"] > ema_s
    trending = adx > 20

    score = pd.Series(0.0, index=df.index)
    score.loc[trend_up & price_ok] = 0.5
    score.loc[trend_up & trending & price_ok] = 1.0
    return score


# ---------------------------------------------------------------------------
# Ensemble score computation
# ---------------------------------------------------------------------------

def compute_ensemble_score(ai_signal: pd.Series, trend_signal: pd.Series) -> pd.Series:
    """纯函数：加权融合（与策略实现一致）。"""
    w_ai = 0.60
    w_trend = 0.40
    ai = ai_signal.clip(0.0, 1.0).fillna(0.5)
    trend = trend_signal.clip(0.0, 1.0).fillna(0.0)
    return w_ai * ai + w_trend * trend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """生成带明显趋势和震荡的合成数据。"""
    np.random.seed(42)
    n = 200
    # 前 100 根：上涨趋势
    trend = np.linspace(100, 150, 100)
    # 后 100 根：震荡
    osc = 150 + np.sin(np.linspace(0, 4 * np.pi, 100)) * 10
    close = np.concatenate([trend, osc])
    df = pd.DataFrame({
        "open": close - np.abs(np.random.randn(n) * 2),
        "high": close + np.abs(np.random.randn(n) * 3),
        "low": close - np.abs(np.random.randn(n) * 3),
        "close": close,
        "volume": np.abs(np.random.randn(n) * 1000 + 500),
    })
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = np.maximum(df["high"], df[["open", "close"]].max(axis=1))
    df["low"] = np.minimum(df["low"], df[["open", "close"]].min(axis=1))
    return df


# ---------------------------------------------------------------------------
# Trend signal tests
# ---------------------------------------------------------------------------

class TestTrendSignal:
    def test_full_bull(self, sample_ohlcv: pd.DataFrame) -> None:
        """强趋势 + ADX > 20 → trend_signal = 1.0。"""
        df = sample_ohlcv.copy()
        signal = compute_trend_signal(df)
        # In the first 100 bars (strong uptrend), signal should reach 1.0
        assert signal.iloc[50:90].max() == pytest.approx(1.0, abs=1e-6)

    def test_ema_bull_weak_adx(self, sample_ohlcv: pd.DataFrame) -> None:
        """EMA 多头但 ADX 弱 → trend_signal = 0.5。"""
        df = sample_ohlcv.copy()
        signal = compute_trend_signal(df)
        # There should be some 0.5 values (EMA bull but not strong ADX)
        assert 0.5 in signal.values

    def test_no_trend(self, sample_ohlcv: pd.DataFrame) -> None:
        """无趋势 → trend_signal = 0.0。"""
        df = sample_ohlcv.copy()
        signal = compute_trend_signal(df)
        assert signal.min() == pytest.approx(0.0, abs=1e-6)

    def test_range_0_to_1(self, sample_ohlcv: pd.DataFrame) -> None:
        """trend_signal 必须在 [0, 1] 范围内。"""
        df = sample_ohlcv.copy()
        signal = compute_trend_signal(df)
        assert signal.min() >= 0.0
        assert signal.max() <= 1.0


# ---------------------------------------------------------------------------
# AI signal tests
# ---------------------------------------------------------------------------

class TestAISignal:
    def test_strong_signal(self) -> None:
        """ai_prediction = 0.9 → ai_signal = 0.9。"""
        ai = pd.Series([0.9])
        assert ai.clip(0, 1).iloc[0] == pytest.approx(0.9, abs=1e-6)

    def test_weak_signal(self) -> None:
        """ai_prediction = 0.3 → ai_signal = 0.3。"""
        ai = pd.Series([0.3])
        assert ai.clip(0, 1).iloc[0] == pytest.approx(0.3, abs=1e-6)

    def test_nan_fallback(self) -> None:
        """ai_prediction = NaN → fallback to 0.5。"""
        ai = pd.Series([np.nan])
        result = ai.clip(0.0, 1.0).fillna(0.5)
        assert result.iloc[0] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Ensemble score tests
# ---------------------------------------------------------------------------

class TestEnsembleScore:
    def test_max_score(self) -> None:
        """ai=1.0, trend=1.0 → ensemble = 1.0。"""
        ai = pd.Series([1.0])
        trend = pd.Series([1.0])
        score = compute_ensemble_score(ai, trend)
        assert score.iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_min_score(self) -> None:
        """ai=0.0, trend=0.0 → ensemble = 0.0。"""
        ai = pd.Series([0.0])
        trend = pd.Series([0.0])
        score = compute_ensemble_score(ai, trend)
        assert score.iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_mid_score(self) -> None:
        """ai=0.75, trend=0.5 → ensemble = 0.65。"""
        ai = pd.Series([0.75])
        trend = pd.Series([0.5])
        score = compute_ensemble_score(ai, trend)
        # 0.6 * 0.75 + 0.4 * 0.5 = 0.45 + 0.20 = 0.65
        assert score.iloc[0] == pytest.approx(0.65, abs=1e-6)

    def test_weight_sum(self) -> None:
        """权重和为 1.0。"""
        ai = pd.Series([0.5])
        trend = pd.Series([0.5])
        score = compute_ensemble_score(ai, trend)
        assert score.iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_clips_out_of_range(self) -> None:
        """超出范围的信号被 clip 到 [0, 1]。"""
        ai = pd.Series([1.5])
        trend = pd.Series([-0.3])
        score = compute_ensemble_score(ai, trend)
        # ai clipped to 1.0, trend clipped to 0.0
        assert score.iloc[0] == pytest.approx(0.6, abs=1e-6)

    def test_nan_handling(self) -> None:
        """NaN 信号正确 fallback。"""
        ai = pd.Series([np.nan])
        trend = pd.Series([np.nan])
        score = compute_ensemble_score(ai, trend)
        # ai fallback 0.5, trend fallback 0.0
        # 0.6 * 0.5 + 0.4 * 0.0 = 0.3
        assert score.iloc[0] == pytest.approx(0.3, abs=1e-6)


# ---------------------------------------------------------------------------
# Entry / Exit threshold tests
# ---------------------------------------------------------------------------

class TestEntryExit:
    def test_entry_above_threshold(self) -> None:
        """ensemble_score > 0.55 → enter_long = 1。"""
        scores = pd.Series([0.56, 0.60, 0.90])
        enter = (scores > 0.55).astype(int)
        assert enter.sum() == 3

    def test_entry_below_threshold(self) -> None:
        """ensemble_score < 0.55 → enter_long = 0。"""
        scores = pd.Series([0.50, 0.40, 0.30])
        enter = (scores > 0.55).astype(int)
        assert enter.sum() == 0

    def test_exit_below_threshold(self) -> None:
        """ensemble_score < 0.45 → exit_long = 1。"""
        scores = pd.Series([0.40, 0.30, 0.44])
        exit_long = (scores < 0.45).astype(int)
        assert exit_long.sum() == 3

    def test_exit_above_threshold(self) -> None:
        """ensemble_score > 0.45 → exit_long = 0。"""
        scores = pd.Series([0.50, 0.60, 0.70])
        exit_long = (scores < 0.45).astype(int)
        assert exit_long.sum() == 0

    def test_neutral_zone(self) -> None:
        """0.45 <= score <= 0.55 → 不入场也不出场。"""
        scores = pd.Series([0.50])
        enter = (scores > 0.55).astype(int)
        exit_long = (scores < 0.45).astype(int)
        assert enter.sum() == 0
        assert exit_long.sum() == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_ai_dominates_when_trend_weak(self) -> None:
        """AI 强信号 + Trend 弱 → ensemble 主要由 AI 贡献。"""
        ai = pd.Series([0.90])
        trend = pd.Series([0.0])
        score = compute_ensemble_score(ai, trend)
        # 0.6 * 0.9 + 0.4 * 0.0 = 0.54
        assert score.iloc[0] == pytest.approx(0.54, abs=1e-6)
        # Should trigger entry (0.54 > 0.55? No, borderline)

    def test_trend_dominates_when_ai_weak(self) -> None:
        """Trend 强信号 + AI 弱 → ensemble 主要由 Trend 贡献。"""
        ai = pd.Series([0.40])
        trend = pd.Series([1.0])
        score = compute_ensemble_score(ai, trend)
        # 0.6 * 0.4 + 0.4 * 1.0 = 0.24 + 0.40 = 0.64
        assert score.iloc[0] == pytest.approx(0.64, abs=1e-6)
        # Should trigger entry (0.64 > 0.55)

    def test_both_agree(self) -> None:
        """AI 和 Trend 都强 → ensemble 最强。"""
        ai = pd.Series([0.90])
        trend = pd.Series([1.0])
        score = compute_ensemble_score(ai, trend)
        # 0.6 * 0.9 + 0.4 * 1.0 = 0.54 + 0.40 = 0.94
        assert score.iloc[0] == pytest.approx(0.94, abs=1e-6)
