"""
SmallCapTurtleStrategy (v2)

基于 Turtle Trading + Dual Thrust 理论框架的小市值事件驱动策略。
与 v1 的核心差异：v1 用 momentum_fade（成交量萎缩）出场，v2 用 Turtle 经典规则
（Donchian 反向突破 + 2×ATR 固定止损 + 利润回撤）出场，理论支撑更扎实。

理论支撑
--------
1. 入场 — Turtle S1 Donchian Breakout (Dennis & Eckhardt, 1983)
   价格突破过去 N 根 K 线高点 → 趋势起点信号。
   原始 Turtle S1 使用 20-day 高点（日线），映射到 4h 框架：
   20 day × 6 bars/day = 120 bars of 4h。
   但小市值币趋势窗口更短，采用 N=24 (4h×24=96h=4天) 作为 S1 等价映射，
   保持 entry:exit = 2:1 的比例关系（Turtle 经典设计）。

2. 波动率过滤 — Dual Thrust Range (Michael Chalek)
   Range = Max(HH_12 - LC_12, HC_12 - LL_12)
   只在 Range > 20-period 平均 Range 时交易（波动率扩张 = 趋势启动）。
   避免低波动区间的假突破（Chop Filter）。

3. 成交量确认 — Event-Driven Hypothesis
   volume_ratio > 1.5：突破必须有异常成交量推动。
   文献：Clare et al. (2013); Han et al. (2016) 均指出成交量确认显著提升突破持续性。

4. 止损 — Fixed 2×ATR Stop (Turtle Trading)
   入场时记录 ATR，止损设在 entry - 2×ATR。
   Kaminski & Lo (2014): ATR-based stop 保持跨资产风险一致性。
   注意：这是"固定止损"（ anchored at entry ），不是跟踪止损。
   之前回测证明 ATR 跟踪止损在小市值币上灾难性（-38%），
   但固定 ATR 止损是 Turtle 经 40 年验证的经典做法。

5. 出场 — Reverse Donchian (Turtle S1 Exit)
   跌破过去 M 根 K 线低点 → 短期趋势结束。
   M=12 (4h×12=48h=2天)，保持 entry:exit = 2:1 比例。
   Turtle 核心思想：用更短周期出场，放弃顶部利润以换取不被洗出。

6. 利润回撤 — Profit Protection (Turtle Pyramiding 思想)
   从最高点回撤 50% 时锁定利润。
   触发门槛：曾盈利 >15%（小市值币波动大，门槛低于 Turtle 原始设计）。

7. 仓位 — ATR-Adaptive Sizing (Turtle)
   stake = (Account × 2%) / (2 × ATR_pct)
   高波动币种自动减小仓位，低波动币种自动增大仓位。
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

# Ensure data/ package is importable both locally and in Docker
import sys
from pathlib import Path
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

from features import atr

logger = logging.getLogger(__name__)

_UNIVERSE_CANDIDATES = [
    Path("/freqtrade/user_data/data/smallcap_universe.json"),
    Path(__file__).parent.parent / "data" / "smallcap_universe.json",
]
UNIVERSE_PATH = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), _UNIVERSE_CANDIDATES[0])

# Turtle 周期参数（映射到 4h 框架）
TURTLE_ENTRY_PERIOD = 24   # 24 bars of 4h = 96h ≈ 4 days  (S1 entry)
TURTLE_EXIT_PERIOD = 12    # 12 bars of 4h = 48h = 2 days   (S1 exit)
DUAL_THRUST_PERIOD = 12    # Dual Thrust Range lookback
DUAL_THRUST_AVG = 20       # Range average for chop filter
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0  # Turtle classic: 2×ATR
RISK_PER_TRADE = 0.02      # 2% account risk per trade (Turtle)
MAX_STAKE_PCT = 0.05       # Max 5% of account per position


class SmallCapTurtleStrategy(IStrategy):
    timeframe = "4h"
    stoploss = -0.50          # Ultra-wide emergency fallback only
    max_open_trades = 5
    can_short = False
    startup_candle_count = max(TURTLE_ENTRY_PERIOD, DUAL_THRUST_AVG, ATR_PERIOD) + 5

    # ROI kept as ultimate profit target; Turtle exit rules handle most exits
    minimal_roi = {"0": 0.30}

    trailing_stop = False
    use_custom_stoploss = False

    universe_coins: list[str] = []
    universe_data: dict[str, dict] = {}
    trade_max_profit: dict = {}
    trade_entry_atr: dict = {}   # Maps trade.id -> entry ATR value (for fixed ATR stop)

    def bot_start(self, **kwargs) -> None:
        logger.info("[SmallCapV2] Loading universe...")
        universe_path = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), UNIVERSE_PATH)
        if universe_path.exists():
            with open(universe_path, "r") as f:
                data = json.load(f)
            coins = data.get("coins", [])
            self.universe_coins = [c["binance_pair"] for c in coins if "binance_pair" in c]
            self.universe_data = {c["binance_pair"]: c for c in coins if "binance_pair" in c}
            logger.info(f"[SmallCapV2] Loaded {len(self.universe_coins)} coins.")
        else:
            logger.warning(f"[SmallCapV2] Universe not found at {universe_path}.")
            self.universe_coins = []
            self.universe_data = {}
        self.trade_max_profit = {}
        self.trade_entry_atr = {}

    # ------------------------------------------------------------------
    # ATR-Adaptive Position Sizing (Turtle Rule #2)
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self, pair: str, current_time, current_rate: float,
        proposed_stake: float, min_stake: Optional[float], max_stake: Optional[float],
        entry_tag: Optional[str], side: str, **kwargs,
    ) -> float:
        total = self.wallets.get_total_stake_amount()
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if not dataframe.empty:
            atr_val = dataframe["atr_14"].iloc[-1]
            if atr_val > 0 and current_rate > 0:
                atr_pct = atr_val / current_rate
                # Dollar risk = 2% of account; ATR risk = 2×ATR
                # Stake = DollarRisk / (2 × ATR_pct)
                stake = (total * RISK_PER_TRADE) / (ATR_STOP_MULTIPLIER * atr_pct)
                stake = min(stake, total * MAX_STAKE_PCT)
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
                logger.warning(f"[SmallCapV2] Failed to merge funding rate: {e}")

            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                oi_df = query("open_interest", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, oi_df, "openInterest")
            except Exception as e:
                logger.warning(f"[SmallCapV2] Failed to merge open interest: {e}")

        # Supply injection
        if pair in self.universe_data:
            supply = self.universe_data[pair].get("total_supply", 0)
            dataframe["supply"] = float(supply) if supply else 0.0
        else:
            dataframe["supply"] = 0.0

        # Volume metrics
        dataframe["volume_sma_20"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_sma_20"] + 1e-9)
        dataframe["rolling_vol_24h"] = dataframe["volume"].rolling(window=6).sum()

        # Real-time market cap & turnover
        dataframe["market_cap"] = dataframe["close"] * dataframe["supply"]
        dataframe["turnover_24h"] = (
            dataframe["rolling_vol_24h"] / (dataframe["supply"] + 1e-9) * 100.0
        )

        # ── Turtle Donchian Channels ──
        # Entry: highest high over N bars (shift 1 = strict, no intra-bar repainting)
        dataframe[f"hh_{TURTLE_ENTRY_PERIOD}"] = (
            dataframe["high"].rolling(window=TURTLE_ENTRY_PERIOD).max().shift(1)
        )
        # Exit: lowest low over M bars
        dataframe[f"ll_{TURTLE_EXIT_PERIOD}"] = (
            dataframe["low"].rolling(window=TURTLE_EXIT_PERIOD).min().shift(1)
        )

        # ── Dual Thrust Range & Chop Filter ──
        hh_dt = dataframe["high"].rolling(window=DUAL_THRUST_PERIOD).max()
        ll_dt = dataframe["low"].rolling(window=DUAL_THRUST_PERIOD).min()
        hc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).max()
        lc_dt = dataframe["close"].rolling(window=DUAL_THRUST_PERIOD).min()
        dt_range = pd.concat([hh_dt - lc_dt, hc_dt - ll_dt], axis=1).max(axis=1)
        dt_range_avg = dt_range.rolling(window=DUAL_THRUST_AVG).mean()
        dataframe["dt_range"] = dt_range
        dataframe["dt_range_avg"] = dt_range_avg
        dataframe["range_expanding"] = dt_range > dt_range_avg

        # ── Volatility ──
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], ATR_PERIOD)
        dataframe["candle_return"] = dataframe["close"] / (dataframe["open"] + 1e-9) - 1.0

        return dataframe

    # ------------------------------------------------------------------
    # Entry signal (Turtle S1 + Dual Thrust + Volume)
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        pair = metadata.get("pair", "")

        if pair not in self.universe_coins or dataframe["supply"].iloc[-1] <= 0:
            return dataframe

        # Filters
        small_cap = dataframe["market_cap"] <= 1_000_000_000
        active = dataframe["turnover_24h"] >= 1.0
        volume_spike = dataframe["volume_ratio"] > 1.5
        not_parabolic = dataframe["close"].pct_change(6) < 0.20
        bullish = dataframe["close"] > dataframe["open"]

        # Turtle S1 breakout: close > highest high of past 24 bars
        price_breakout = dataframe["close"] > dataframe[f"hh_{TURTLE_ENTRY_PERIOD}"]

        # Dual Thrust: only trade when volatility is expanding
        # (avoids false breakouts in low-volatility chop)
        vol_expanding = dataframe["range_expanding"]

        dataframe.loc[
            small_cap & active & volume_spike & price_breakout
            & bullish & not_parabolic & vol_expanding,
            "enter_long",
        ] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Exit signal (dataframe-based minimal)
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # Custom exit — ATR stop / Crash / Turtle reverse Donchian / Profit pullback
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

        # 1. Fixed 2×ATR stop (Turtle Rule #4)
        #    Anchored at entry_price - 2×ATR.  Moved here from custom_stoploss
        #    because freqtrade's custom_stoploss API is tricky with fixed stops.
        if trade_key not in self.trade_entry_atr:
            entry_candle = dataframe[dataframe["date"] <= pd.Timestamp(trade.open_date_utc)].iloc[-1]
            self.trade_entry_atr[trade_key] = float(entry_candle["atr_14"])

        entry_atr = self.trade_entry_atr[trade_key]
        atr_stop_pct = (ATR_STOP_MULTIPLIER * entry_atr) / trade.open_rate
        if current_profit < -atr_stop_pct:
            return "atr_stop_2x"

        # 2. Crash exit: single 4h candle drops > 12%
        candle_ret = last_candle.get("candle_return", 0)
        if pd.notna(candle_ret) and candle_ret < -0.12:
            return "crash_exit"

        # 3. Turtle S1 exit: close below lowest low of past 12 bars
        #    Theory: if price breaks back below the recent support band,
        #    the breakout has failed and the short-term trend is over.
        ll_col = f"ll_{TURTLE_EXIT_PERIOD}"
        if hold_seconds > 4 * 3600 and last_candle["close"] < last_candle[ll_col]:
            return "turtle_reverse_donchian"

        # 4. Profit pullback: lock in gains when retracing 50% from peak
        self.trade_max_profit[trade_key] = max(
            self.trade_max_profit.get(trade_key, current_profit), current_profit
        )
        max_profit = self.trade_max_profit[trade_key]
        if max_profit > 0.15 and current_profit < max_profit * 0.50:
            return "profit_pullback_50pct"

        return None
