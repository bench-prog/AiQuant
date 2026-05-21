"""
SmallCapRegimeStrategy (v3) — Hyperopt-enabled

Regime Switching Hybrid framework for small-cap crypto trading.
Hyperopt parameters allow optimization of regime detection thresholds
and entry conditions.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import adx, atr, rsi

# Ensure data/ package is importable both locally and in Docker
_data_dir = None
for candidate in [
    Path(__file__).parent.parent.parent.parent / "data",
    Path("/freqtrade/data"),
]:
    if candidate.exists():
        _data_dir = candidate
        break

if _data_dir is None:
    raise ImportError("Could not find data/ directory.")

if str(_data_dir.parent) not in sys.path:
    sys.path.insert(0, str(_data_dir.parent))

from data.service import query, merge_into
import data.service_defaults  # registers built-in data sources

logger = logging.getLogger(__name__)

_UNIVERSE_CANDIDATES = [
    Path("/freqtrade/user_data/data/smallcap_universe.json"),
    Path(__file__).parent.parent / "data" / "smallcap_universe.json",
]
UNIVERSE_PATH = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), _UNIVERSE_CANDIDATES[0])

# Non-hyperopt constants
TURTLE_EXIT_PERIOD = 12
DUAL_THRUST_PERIOD = 12
DUAL_THRUST_AVG = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
RISK_PER_TRADE = 0.02
MAX_STAKE_PCT = 0.05
MR_HARD_STOP = -0.10
MR_TIME_STOP_HOURS = 48
BB_PERIOD = 20


class SmallCapRegimeStrategy(IStrategy):
    """
    Regime Switching Hybrid small-cap strategy with hyperopt support.
    """

    timeframe = "4h"
    stoploss = -0.50
    max_open_trades = 5
    can_short = False
    startup_candle_count = max(24, DUAL_THRUST_AVG, ATR_PERIOD, BB_PERIOD) + 5

    minimal_roi = {"0": 0.30}
    trailing_stop = False
    use_custom_stoploss = False

    # --- Buy space (regime detection + entry) ---
    # Relaxed defaults for more trades while keeping win rate
    buy_adx_trend = IntParameter(15, 35, default=20, space="buy")
    buy_adx_weak = IntParameter(10, 25, default=20, space="buy")
    buy_atr_expansion = DecimalParameter(1.0, 1.5, decimals=1, default=1.1, space="buy")
    buy_atr_contract = DecimalParameter(0.8, 1.2, decimals=1, default=1.0, space="buy")
    buy_volume_expansion = DecimalParameter(1.0, 2.0, decimals=1, default=1.2, space="buy")
    buy_volume_fade = DecimalParameter(0.5, 1.0, decimals=1, default=0.9, space="buy")
    buy_volume_ratio = DecimalParameter(1.0, 2.5, decimals=1, default=1.3, space="buy")
    buy_not_para_pct = DecimalParameter(0.10, 0.30, decimals=2, default=0.25, space="buy")
    buy_enable_mr = BooleanParameter(default=True, space="buy")
    buy_rsi_oversold = IntParameter(20, 45, default=35, space="buy")

    # --- Sell space (exit thresholds) ---
    sell_profit_pullback_threshold = DecimalParameter(0.05, 0.30, decimals=2, default=0.15, space="sell")
    sell_profit_pullback_ratio = DecimalParameter(0.20, 0.80, decimals=2, default=0.50, space="sell")

    # --- BTC market filter ---
    buy_btc_filter = BooleanParameter(default=True, space="buy")
    buy_btc_ema_period = IntParameter(10, 50, default=20, space="buy")

    universe_coins: list[str] = []
    universe_data: dict[str, dict] = {}
    trade_max_profit: dict = {}
    trade_entry_atr: dict = {}
    _btc_data: pd.DataFrame | None = None  # Cached BTC data for market filter

    def _load_btc_data(self) -> None:
        """Load BTC data for market filter. Called lazily to support backtest."""
        if not self.buy_btc_filter.value:
            return
        if SmallCapRegimeStrategy._btc_data is not None:
            return

        btc_path = Path("/freqtrade/user_data/data/binance/BTC_USDT-4h.feather")
        if not btc_path.exists():
            btc_path = Path(__file__).parent.parent / "data" / "binance" / "BTC_USDT-4h.feather"
        if btc_path.exists():
            try:
                import pyarrow.feather as feather
                btc_df = feather.read_feather(btc_path)
                btc_df["btc_ema"] = btc_df["close"].rolling(window=self.buy_btc_ema_period.value).mean()
                SmallCapRegimeStrategy._btc_data = btc_df[["date", "close", "btc_ema"]].copy()
                logger.info(f"[SmallCapV3] Loaded BTC data for market filter ({len(btc_df)} rows).")
            except Exception as e:
                logger.warning(f"[SmallCapV3] Failed to load BTC data: {e}")
        else:
            logger.warning(f"[SmallCapV3] BTC data not found at {btc_path}")

    def bot_start(self, **kwargs) -> None:
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

        self._load_btc_data()

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
                stake = total * RISK_PER_TRADE
                if max_stake is not None:
                    stake = min(stake, max_stake)
                if min_stake is not None:
                    stake = max(stake, min_stake)
                return stake
        return total * 0.03

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Lazy-load BTC data for backtest compatibility (bot_start is not called in backtest)
        self._load_btc_data()

        pair = metadata.get("pair", "")
        exchange_name = self.config.get("exchange", {}).get("name", "binance")

        # Merge external data (funding rate / open interest) via unified data service
        if pair and "date" in dataframe.columns:
            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                fr_df = query("funding_rate", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, fr_df, "fundingRate")
            except Exception as e:
                logger.warning(f"[SmallCapV3] Failed to merge funding rate: {e}")

            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                oi_df = query("open_interest", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, oi_df, "openInterest")
            except Exception as e:
                logger.warning(f"[SmallCapV3] Failed to merge open interest: {e}")

        if pair in self.universe_data:
            supply = self.universe_data[pair].get("total_supply", 0)
            dataframe["supply"] = float(supply) if supply else 0.0
        else:
            dataframe["supply"] = 0.0

        # Volume
        dataframe["volume_sma_6"] = dataframe["volume"].rolling(window=6).mean()
        dataframe["volume_sma_20"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_sma_20"] + 1e-9)
        dataframe["volume_trend"] = dataframe["volume_sma_6"] / (dataframe["volume_sma_20"] + 1e-9)
        dataframe["rolling_vol_24h"] = dataframe["volume"].rolling(window=6).sum()

        # Market cap & turnover
        dataframe["market_cap"] = dataframe["close"] * dataframe["supply"]
        dataframe["turnover_24h"] = (
            dataframe["rolling_vol_24h"] / (dataframe["supply"] + 1e-9) * 100.0
        )

        # Donchian
        dataframe["hh_24"] = dataframe["high"].rolling(window=24).max().shift(1)
        dataframe["ll_12"] = dataframe["low"].rolling(window=12).min().shift(1)

        # Dual Thrust
        hh_dt = dataframe["high"].rolling(window=DUAL_THRUST_PERIOD).max()
        ll_dt = dataframe["low"].rolling(window=DUAL_THRUST_PERIOD).min()
        hc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).max()
        lc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).min()
        dt_range = pd.concat([hh_dt - lc_dt, hc_dt - ll_dt], axis=1).max(axis=1)
        dt_range_avg = dt_range.rolling(window=DUAL_THRUST_AVG).mean()
        dataframe["range_expanding"] = dt_range > dt_range_avg

        # Volatility
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], ATR_PERIOD)
        dataframe["atr_14_sma_20"] = dataframe["atr_14"].rolling(window=20).mean()
        dataframe["atr_ratio"] = dataframe["atr_14"] / (dataframe["atr_14_sma_20"] + 1e-9)
        dataframe["candle_return"] = dataframe["close"] / (dataframe["open"] + 1e-9) - 1.0

        # ADX
        adx_series, plus_di, minus_di = adx(
            dataframe["high"], dataframe["low"], dataframe["close"], 14
        )
        dataframe["adx_14"] = adx_series

        # Bollinger Bands
        bb_middle = dataframe["close"].rolling(window=BB_PERIOD).mean()
        bb_std = dataframe["close"].rolling(window=BB_PERIOD).std()
        dataframe["bb_middle"] = bb_middle
        dataframe["bb_lower"] = bb_middle - 2.0 * bb_std

        # RSI
        dataframe["rsi_14"] = rsi(dataframe["close"], 14)

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        pair = metadata.get("pair", "")

        if pair not in self.universe_coins or dataframe["supply"].iloc[-1] <= 0:
            return dataframe

        # Regime detection (using hyperopt parameters)
        momentum_cond = (
            (dataframe["atr_ratio"] > self.buy_atr_expansion.value)
            & (dataframe["volume_trend"] > self.buy_volume_expansion.value)
            & (dataframe["adx_14"] > self.buy_adx_trend.value)
        )
        mean_reversion_cond = (
            (dataframe["atr_ratio"] > self.buy_atr_contract.value)
            & (dataframe["volume_trend"] < self.buy_volume_fade.value)
            & (dataframe["adx_14"] < self.buy_adx_weak.value)
        )

        dataframe["regime"] = "accumulation"
        dataframe.loc[momentum_cond, "regime"] = "momentum"
        dataframe.loc[mean_reversion_cond, "regime"] = "mean_reversion"

        # Shared filters
        small_cap = dataframe["market_cap"] <= 1_000_000_000
        active = dataframe["turnover_24h"] >= 1.0
        not_parabolic = dataframe["close"].pct_change(6) < self.buy_not_para_pct.value

        # Momentum entry
        momentum_regime = dataframe["regime"] == "momentum"
        volume_spike = dataframe["volume_ratio"] > self.buy_volume_ratio.value
        price_breakout = dataframe["close"] > dataframe["hh_24"]
        vol_expanding = dataframe["range_expanding"]
        bullish = dataframe["close"] > dataframe["open"]

        momentum_entry = (
            momentum_regime & small_cap & active & volume_spike
            & price_breakout & vol_expanding & bullish & not_parabolic
        )

        # Mean Reversion entry
        mr_regime = dataframe["regime"] == "mean_reversion"
        oversold = dataframe["rsi_14"] < self.buy_rsi_oversold.value
        bb_below = dataframe["close"] < dataframe["bb_lower"]
        volume_fade = dataframe["volume_ratio"] < 0.8
        not_collapsing = dataframe["close"] > dataframe["ll_12"]

        mr_entry = (
            mr_regime & small_cap & active & oversold
            & bb_below & volume_fade & not_collapsing & not_parabolic
        )

        # BTC market filter: only trade when BTC is in uptrend
        if self.buy_btc_filter.value and SmallCapRegimeStrategy._btc_data is not None:
            btc_df = SmallCapRegimeStrategy._btc_data
            if not btc_df.empty:
                # Find BTC trend at current date
                current_date = dataframe["date"].iloc[-1]
                btc_row = btc_df[btc_df["date"] <= current_date]
                if not btc_row.empty:
                    btc_bullish = btc_row["close"].iloc[-1] > btc_row["btc_ema"].iloc[-1]
                    if not btc_bullish:
                        return dataframe  # Skip all entries in BTC bear market

        if self.buy_enable_mr.value:
            dataframe.loc[momentum_entry | mr_entry, "enter_long"] = 1
        else:
            dataframe.loc[momentum_entry, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

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

        # Crash exit
        candle_ret = last_candle.get("candle_return", 0)
        if pd.notna(candle_ret) and candle_ret < -0.12:
            return "crash_exit"

        if regime == "momentum":
            # Fixed 2xATR stop
            if trade_key not in self.trade_entry_atr:
                entry_candle = dataframe[dataframe["date"] <= pd.Timestamp(trade.open_date_utc)].iloc[-1]
                self.trade_entry_atr[trade_key] = float(entry_candle["atr_14"])

            entry_atr = self.trade_entry_atr[trade_key]
            atr_stop_pct = (ATR_STOP_MULTIPLIER * entry_atr) / trade.open_rate
            if current_profit < -atr_stop_pct:
                return "atr_stop_2x"

            # ADX exhaustion
            if hold_seconds > 4 * 3600:
                adx_val = last_candle["adx_14"]
                if pd.notna(adx_val) and adx_val < self.buy_adx_weak.value:
                    return "adx_exhaustion"

            # Profit pullback
            self.trade_max_profit[trade_key] = max(
                self.trade_max_profit.get(trade_key, current_profit), current_profit
            )
            max_profit = self.trade_max_profit[trade_key]
            if max_profit > self.sell_profit_pullback_threshold.value and current_profit < max_profit * self.sell_profit_pullback_ratio.value:
                return "profit_pullback_50pct"

        elif regime == "mean_reversion":
            if current_profit < MR_HARD_STOP:
                return "mr_hard_stop"
            bb_middle = last_candle["bb_middle"]
            if pd.notna(bb_middle) and current_rate >= bb_middle:
                return "mr_target_bb_middle"
            if hold_seconds > MR_TIME_STOP_HOURS * 3600:
                return "mr_time_stop"

        return None
