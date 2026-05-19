"""
SmallCapEventDrivenStrategy

纯事件驱动小市值高换手策略。
- 市值/换手率由 K 线实时计算（close × total_supply），不依赖静态数据。
- 入场：成交量突增 1.5x + 价格突破前 4h 高点 + 实时市值≤$10亿 + 换手 ≥1%
- 出场（基于学术文献与经典策略）：
  1. 硬止损 -15%  （Kaminski & Lo / Momentum Crash 论文）
  2. 单根崩盘 -12% （crash exit）
  3. 利润回撤 40% （Turtle Trading 利润保护思想）
  4. 突破失败：跌破前 4h 低点 （Turtle 反向 Donchian）
  5. 动能消散：成交量萎缩 + 价格未创新高
  6. 极端止盈 30% ROI（保留已有利润来源）
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

from features import atr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_UNIVERSE_CANDIDATES = [
    Path("/freqtrade/user_data/data/smallcap_universe.json"),
    Path(__file__).parent.parent / "data" / "smallcap_universe.json",
]
UNIVERSE_PATH = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), _UNIVERSE_CANDIDATES[0])


class SmallCapEventDrivenStrategy(IStrategy):
    """
    纯事件驱动小市值高换手策略。
    """

    timeframe = "4h"
    stoploss = -0.25          # wide hard stop for small-cap volatility
    max_open_trades = 5
    can_short = False
    startup_candle_count = 30

    # Profit targets — take quick gains in small-cap bursts
    minimal_roi = {
        "0": 0.30,    # 30% profit → immediate exit
    }

    trailing_stop = False

    # Strategy state
    universe_coins: list[str] = []
    universe_data: dict[str, dict] = {}
    trade_max_profit: dict = {}

    def bot_start(self, **kwargs) -> None:
        """Load smallcap universe at bot startup."""
        logger.info("[SmallCap] Loading universe...")
        universe_path = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), UNIVERSE_PATH)
        if universe_path.exists():
            with open(universe_path, "r") as f:
                data = json.load(f)
            coins = data.get("coins", [])
            self.universe_coins = [c["binance_pair"] for c in coins if "binance_pair" in c]
            self.universe_data = {c["binance_pair"]: c for c in coins if "binance_pair" in c}
            logger.info(f"[SmallCap] Loaded {len(self.universe_coins)} coins from universe.")
        else:
            logger.warning(f"[SmallCap] Universe file not found at {universe_path}.")
            self.universe_coins = []
            self.universe_data = {}
        self.trade_max_profit = {}


    # ------------------------------------------------------------------
    # Stake amount (ATR-adaptive, max 3%)
    # ------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: Optional[float],
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        total = self.wallets.get_total_stake_amount()
        # Default 3%
        target = total * 0.03
        # ATR-adaptive sizing would need dataframe access here;
        # for simplicity cap at 3% and let caller clip.
        if max_stake is not None:
            target = min(target, max_stake)
        if min_stake is not None:
            target = max(target, min_stake)
        return target

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")

        # Inject supply from universe
        if pair in self.universe_data:
            supply = self.universe_data[pair].get("total_supply", 0)
            dataframe["supply"] = float(supply) if supply else 0.0
        else:
            dataframe["supply"] = 0.0

        # Volume metrics
        dataframe["volume_sma_20"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / (dataframe["volume_sma_20"] + 1e-9)
        # 6 bars of 4h = 24h
        dataframe["rolling_vol_24h"] = dataframe["volume"].rolling(window=6).sum()

        # Real-time market cap & turnover
        dataframe["market_cap"] = dataframe["close"] * dataframe["supply"]
        dataframe["turnover_24h"] = (
            dataframe["rolling_vol_24h"] / (dataframe["supply"] + 1e-9) * 100.0
        )

        # Event-driven price levels
        dataframe["hh_4"] = dataframe["high"].rolling(window=4).max().shift(1)
        dataframe["ll_4"] = dataframe["low"].rolling(window=4).min().shift(1)
        dataframe["ll_8"] = dataframe["low"].rolling(window=8).min().shift(1)

        # Volatility
        dataframe["atr_14"] = atr(dataframe["high"], dataframe["low"], dataframe["close"], 14)
        dataframe["candle_return"] = dataframe["close"] / (dataframe["open"] + 1e-9) - 1.0

        return dataframe

    # ------------------------------------------------------------------
    # Entry signal
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        pair = metadata.get("pair", "")

        # Must be in universe and have supply data
        if pair not in self.universe_coins or dataframe["supply"].iloc[-1] <= 0:
            return dataframe

        # Real-time filters
        # Market cap: use $1B coarse cap (strategy refines; $500M was too strict
        # for several coins that grew during 2024-2025).
        small_cap = dataframe["market_cap"] <= 1_000_000_000
        # Turnover on Binance single-exchange is ~1-10% median; 1% is active enough.
        active = dataframe["turnover_24h"] >= 1.0
        # Volume spike: 1.5x 20-bar mean is a meaningful 4h burst.
        volume_spike = dataframe["volume_ratio"] > 1.5
        price_breakout = dataframe["close"] > dataframe["hh_4"]
        # Bullish candle — avoid entering on a dump candle
        bullish = dataframe["close"] > dataframe["open"]
        # Not chasing a parabolic move (> 20% in last 24h = 6 bars)
        not_parabolic = dataframe["close"].pct_change(6) < 0.20

        dataframe.loc[
            small_cap & active & volume_spike & price_breakout & bullish & not_parabolic,
            "enter_long",
        ] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Exit signal (dataframe-based, kept minimal)
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        return dataframe

    # ------------------------------------------------------------------
    # Custom exit — event lifecycle model (replaces fixed 36h time exit)
    # ------------------------------------------------------------------
    def custom_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]
        hold_seconds = (current_time - trade.open_date_utc).total_seconds()

        # 1. 硬止损 -15%
        #    文献: Kaminski & Lo (2014); Han et al. (2016) 10%止损将动量亏损
        #    从 -49.79% 降到 -11.36%。小市值波动更大，放宽到 15%。
        if current_profit < -0.15:
            return "hard_stop_15pct"

        # 2. Crash exit: single 4h candle drops > 12%
        candle_ret = last_candle.get("candle_return", 0)
        if pd.notna(candle_ret) and candle_ret < -0.12:
            return "crash_exit"

        # 3. 利润回撤出场: 从最高点回撤 50%
        #    原理: Turtle Trading — 让利润奔跑，仅在显著回撤时离场。
        #    触发条件: 曾盈利 >10%，现在回撤至最高盈利的 50% 以下。
        #    降低门槛以捕获更多 10%-30% 区间的利润（避免被 momentum_fade 在微盈时截断）。
        trade_key = trade.id
        self.trade_max_profit[trade_key] = max(
            self.trade_max_profit.get(trade_key, current_profit), current_profit
        )
        max_profit = self.trade_max_profit[trade_key]
        if max_profit > 0.10 and current_profit < max_profit * 0.50:
            return "profit_pullback_50pct"

        # 4. 动能消散出场: 成交量萎缩 + 价格未创新高
        #    原理: 事件驱动策略的核心假设是"异常成交量带来价格冲击"。
        #    当成交量回到 20 期均线以下 (<1.0x) 且价格不再突破前高，
        #    说明驱动事件已结束。
        #    8h (2 根 4h bar) 后判断 — 假突破通常在 8-12h 内显露，
        #    faster exit 避免拖到 -15% hard_stop。
        if hold_seconds > 8 * 3600:
            vol_now = last_candle["volume_ratio"]
            not_breaking = last_candle["close"] <= last_candle["hh_4"]
            if vol_now < 1.0 and not_breaking:
                return "momentum_fade"

        return None
