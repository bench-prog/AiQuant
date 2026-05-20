"""
SmallCapRegimeStrategy (v3)

Regime Switching Hybrid framework for small-cap crypto trading.

Theory:
- Small-cap assets cycle through Accumulation -> Momentum -> Mean Reversion -> Accumulation.
- Fixed rules (v1/v2) underperform when applied uniformly across all regimes.
- v3 detects the current regime via ADX + ATR ratio + volume trend, then applies
  regime-specific entry/exit rules.

Regimes:
  1. Accumulation: Low vol, low volume, range-bound -> No entry.
  2. Momentum: Vol expanding, volume sustained, strong trend -> Trend-following breakout.
  3. Mean Reversion: High vol but no trend, volume fading, oversold -> Counter-trend dip buy.

Detection method (pure technical indicators, no ML):
  - ATR ratio (atr_14 / atr_14_sma_20) > 1.2 = volatility expanding
  - Volume trend (volume_sma_6 / volume_sma_20) > 1.3 = volume surging
  - ADX > 25 = strong trend
  Combined: all three = Momentum; high vol + fading volume + weak trend = Mean Reversion.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

from features import adx, atr

logger = logging.getLogger(__name__)

_UNIVERSE_CANDIDATES = [
    Path("/freqtrade/user_data/data/smallcap_universe.json"),
    Path(__file__).parent.parent / "data" / "smallcap_universe.json",
]
UNIVERSE_PATH = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), _UNIVERSE_CANDIDATES[0])

# ---------------------------------------------------------------------------
# Regime detection parameters
# ---------------------------------------------------------------------------
ATR_EXPANSION = 1.2
ATR_CONTRACTION_THRESHOLD = 1.0
VOLUME_EXPANSION = 1.3
VOLUME_FADE = 0.9
ADX_TREND = 25
ADX_WEAK = 20

# Turtle parameters
TURTLE_ENTRY_PERIOD = 24
TURTLE_EXIT_PERIOD = 12
DUAL_THRUST_PERIOD = 12
DUAL_THRUST_AVG = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

# Risk
RISK_PER_TRADE = 0.02
MAX_STAKE_PCT = 0.05

# Mean reversion
RSI_OVERSOLD = 35
BB_PERIOD = 20
MR_HARD_STOP = -0.10
MR_TIME_STOP_HOURS = 48

# Profit protection
PROFIT_PULLBACK_THRESHOLD = 0.15
PROFIT_PULLBACK_RATIO = 0.50


class SmallCapRegimeStrategy(IStrategy):
    """
    Regime Switching Hybrid small-cap strategy.
    """

    timeframe = "4h"
    stoploss = -0.50          # Ultra-wide emergency fallback only
    max_open_trades = 5
    can_short = False
    startup_candle_count = max(TURTLE_ENTRY_PERIOD, DUAL_THRUST_AVG, ATR_PERIOD, BB_PERIOD) + 5

    # ROI kept as ultimate profit target; regime-specific exits handle most exits
    minimal_roi = {"0": 0.30}

    trailing_stop = False
    use_custom_stoploss = False

    universe_coins: list[str] = []
    universe_data: dict[str, dict] = {}
    trade_max_profit: dict = {}
    trade_entry_atr: dict = {}

    def bot_start(self, **kwargs) -> None:
        """Load smallcap universe at bot startup."""
        logger.info("[SmallCapV3] Loading universe...")
        universe_path = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), UNIVERSE_PATH)
        if universe_path.exists():
            with open(universe_path, "r") as f:
                data = json.load(f)
            coins = data.get("coins", [])
            self.universe_coins = [c["binance_pair"] for c in coins if "binance_pair" in c]
            self.universe_data = {c["binance_pair"]: c for c in coins if "binance_pair" in c}
            logger.info(f"[SmallCapV3] Loaded {len(self.universe_coins)} coins.")
        else:
            logger.warning(f"[SmallCapV3] Universe not found at {universe_path}.")
            self.universe_coins = []
            self.universe_data = {}
        self.trade_max_profit = {}
        self.trade_entry_atr = {}

    # ------------------------------------------------------------------
    # Regime-aware position sizing
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self, pair: str, current_time, current_rate: float,
        proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
        entry_tag: Optional[str], side: str, **kwargs,
    ) -> float:
        total = self.wallets.get_total_stake_amount()
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not dataframe.empty:
            last = dataframe.iloc[-1]
            regime = last.get("regime", "accumulation")

            if regime == "momentum":
                # ATR-adaptive sizing (Turtle-style)
                atr_val = last["atr_14"]
                if atr_val > 0 and current_rate > 0:
                    atr_pct = atr_val / current_rate
                    stake = (total * RISK_PER_TRADE) / (ATR_STOP_MULTIPLIER * atr_pct)
                    stake = min(stake, total * MAX_STAKE_PCT)
                    if max_stake is not None:
                        stake = min(stake, max_stake)
                    if min_stake is not None:
                        stake = max(stake, min_stake)
                    return stake
            elif regime == "mean_reversion":
                # Fixed 2% for mean reversion (shorter hold, tighter stop)
                stake = total * RISK_PER_TRADE
                if max_stake is not None:
                    stake = min(stake, max_stake)
                if min_stake is not None:
                    stake = max(stake, min_stake)
                return stake

        return total * 0.03

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")

        # Supply injection
        if pair in self.universe_data:
            supply = self.universe_data[pair].get("total_supply", 0)
            dataframe["supply"] = float(supply) if supply else 0.0
        else:
            dataframe["supply"] = 0.0

        # Volume metrics
        dataframe["volume_sma_6"] = dataframe["volume"].rolling(window=6).mean()
        dataframe["volume_sma_20"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_sma_20"] + 1e-9)
        dataframe["volume_trend"] = dataframe["volume_sma_6"] / (dataframe["volume_sma_20"] + 1e-9)
        dataframe["rolling_vol_24h"] = dataframe["volume"].rolling(window=6).sum()

        # Real-time market cap & turnover
        dataframe["market_cap"] = dataframe["close"] * dataframe["supply"]
        dataframe["turnover_24h"] = (
            dataframe["rolling_vol_24h"] / (dataframe["supply"] + 1e-9) * 100.0
        )

        # -- Turtle Donchian Channels --
        dataframe[f"hh_{TURTLE_ENTRY_PERIOD}"] = (
            dataframe["high"].rolling(window=TURTLE_ENTRY_PERIOD).max().shift(1)
        )
        dataframe[f"ll_{TURTLE_EXIT_PERIOD}"] = (
            dataframe["low"].rolling(window=TURTLE_EXIT_PERIOD).min().shift(1)
        )

        # -- Dual Thrust Range & Chop Filter --
        hh_dt = dataframe["high"].rolling(window=DUAL_THRUST_PERIOD).max()
        ll_dt = dataframe["low"].rolling(window=DUAL_THRUST_PERIOD).min()
        hc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).max()
        lc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).min()
        dt_range = pd.concat([hh_dt - lc_dt, hc_dt - ll_dt], axis=1).max(axis=1)
        dt_range_avg = dt_range.rolling(window=DUAL_THRUST_AVG).mean()
        dataframe["dt_range"] = dt_range
        dataframe["dt_range_avg"] = dt_range_avg
        dataframe["range_expanding"] = dt_range > dt_range_avg

        # -- Volatility --
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], ATR_PERIOD)
        dataframe["atr_14_sma_20"] = dataframe["atr_14"].rolling(window=20).mean()
        dataframe["atr_ratio"] = dataframe["atr_14"] / (dataframe["atr_14_sma_20"] + 1e-9)
        dataframe["candle_return"] = dataframe["close"] / (dataframe["open"] + 1e-9) - 1.0

        # -- ADX --
        adx_series, plus_di, minus_di = adx(
            dataframe["high"], dataframe["low"], dataframe["close"], 14
        )
        dataframe["adx_14"] = adx_series
        dataframe["plus_di_14"] = plus_di
        dataframe["minus_di_14"] = minus_di

        # -- Bollinger Bands --
        bb_middle = dataframe["close"].rolling(window=BB_PERIOD).mean()
        bb_std = dataframe["close"].rolling(window=BB_PERIOD).std()
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_middle - 2.0 * bb_std
        dataframe["bb_upper"] = bb_middle + 2.0 * bb_std

        # -- RSI --
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14).mean()
        rs = avg_gain / avg_loss
        dataframe["rsi_14"] = 100 - (100 / (1 + rs))

        # -- Regime Detection --
        # Momentum: expanding volatility + surging volume + strong trend
        momentum_cond = (
            (dataframe["atr_ratio"] > ATR_EXPANSION)
            & (dataframe["volume_trend"] > VOLUME_EXPANSION)
            & (dataframe["adx_14"] > ADX_TREND)
        )
        # Mean Reversion: high vol + fading volume + weak trend
        mean_reversion_cond = (
            (dataframe["atr_ratio"] > ATR_CONTRACTION_THRESHOLD)
            & (dataframe["volume_trend"] < VOLUME_FADE)
            & (dataframe["adx_14"] < ADX_WEAK)
        )

        dataframe["regime"] = "accumulation"
        dataframe.loc[momentum_cond, "regime"] = "momentum"
        dataframe.loc[mean_reversion_cond, "regime"] = "mean_reversion"

        return dataframe

    # ------------------------------------------------------------------
    # Entry signal (regime-specific)
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        pair = metadata.get("pair", "")

        if pair not in self.universe_coins or dataframe["supply"].iloc[-1] <= 0:
            return dataframe

        # Shared filters
        small_cap = dataframe["market_cap"] <= 1_000_000_000
        active = dataframe["turnover_24h"] >= 1.0
        not_parabolic = dataframe["close"].pct_change(6) < 0.20

        # -- Momentum regime entry --
        momentum_regime = dataframe["regime"] == "momentum"
        volume_spike = dataframe["volume_ratio"] > 1.5
        price_breakout = dataframe["close"] > dataframe[f"hh_{TURTLE_ENTRY_PERIOD}"]
        vol_expanding = dataframe["range_expanding"]
        bullish = dataframe["close"] > dataframe["open"]

        momentum_entry = (
            momentum_regime & small_cap & active & volume_spike
            & price_breakout & vol_expanding & bullish & not_parabolic
        )

        # -- Mean Reversion regime entry --
        mr_regime = dataframe["regime"] == "mean_reversion"
        oversold = dataframe["rsi_14"] < RSI_OVERSOLD
        bb_below = dataframe["close"] < dataframe["bb_lower"]
        volume_fade = dataframe["volume_ratio"] < 0.8
        not_collapsing = dataframe["close"] > dataframe[f"ll_{TURTLE_EXIT_PERIOD}"]

        mr_entry = (
            mr_regime & small_cap & active & oversold
            & bb_below & volume_fade & not_collapsing & not_parabolic
        )

        dataframe.loc[momentum_entry | mr_entry, "enter_long"] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Exit signal (dataframe-based minimal)
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # Custom exit -- regime-aware
    # ------------------------------------------------------------------
    def custom_exit(
        self, pair: str, trade, current_time, current_rate: float,
        current_profit: float, **kwargs,
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        hold_seconds = (current_time - trade.open_date_utc).total_seconds()
        trade_key = trade.id
        regime = last_candle.get("regime", "accumulation")

        # -- Shared crash exit --
        candle_ret = last_candle.get("candle_return", 0)
        if pd.notna(candle_ret) and candle_ret < -0.12:
            return "crash_exit"

        if regime == "momentum":
            # 1. Fixed 2xATR stop (Turtle)
            if trade_key not in self.trade_entry_atr:
                entry_candle = dataframe[dataframe["date"] <= pd.Timestamp(trade.open_date_utc)].iloc[-1]
                self.trade_entry_atr[trade_key] = float(entry_candle["atr_14"])

            entry_atr = self.trade_entry_atr[trade_key]
            atr_stop_pct = (ATR_STOP_MULTIPLIER * entry_atr) / trade.open_rate
            if current_profit < -atr_stop_pct:
                return "atr_stop_2x"

            # 2. ADX trend exhaustion: ADX drops below weak threshold
            if hold_seconds > 4 * 3600:
                adx_val = last_candle["adx_14"]
                if pd.notna(adx_val) and adx_val < ADX_WEAK:
                    return "adx_exhaustion"

            # 3. Profit pullback
            self.trade_max_profit[trade_key] = max(
                self.trade_max_profit.get(trade_key, current_profit), current_profit
            )
            max_profit = self.trade_max_profit[trade_key]
            if max_profit > PROFIT_PULLBACK_THRESHOLD and current_profit < max_profit * PROFIT_PULLBACK_RATIO:
                return "profit_pullback_50pct"

        elif regime == "mean_reversion":
            # 1. Hard stop -10% (tighter than momentum)
            if current_profit < MR_HARD_STOP:
                return "mr_hard_stop"

            # 2. Target: price back to BB middle
            bb_middle = last_candle["bb_middle"]
            if pd.notna(bb_middle) and current_rate >= bb_middle:
                return "mr_target_bb_middle"

            # 3. Time stop: 48h max
            if hold_seconds > MR_TIME_STOP_HOURS * 3600:
                return "mr_time_stop"

        return None
