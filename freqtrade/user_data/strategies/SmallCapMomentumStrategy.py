"""
SmallCapMomentumStrategy

小市值高换手动量策略。
- 数据源：CoinPaprika（市值/换手/涨幅）+ Binance（K线/技术形态）
- 筛选逻辑：7日涨幅 Top 20 + 市值≤$5亿 + 换手100%-500% + 上线>30天
- 技术形态：EMA30趋势向上 + 未连续调整3天 + RSI<75
- 风控：单币3%、硬止损-15%、总回撤15%暂停
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

from momentum_filters import apply_all_filters

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Try Docker path first, then fallback to local relative path
_UNIVERSE_CANDIDATES = [
    Path("/freqtrade/user_data/data/smallcap_universe.json"),
    Path(__file__).parent.parent / "data" / "smallcap_universe.json",
]
UNIVERSE_PATH = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), _UNIVERSE_CANDIDATES[0])


class SmallCapMomentumStrategy(IStrategy):
    """
    小市值高换手动量策略。
    """

    timeframe = "4h"
    stoploss = -0.15
    max_open_trades = 5
    can_short = False

    # Minimal ROI - can be overridden
    minimal_roi = {
        "0": 0.20,
        "60": 0.10,
        "120": 0.05,
    }

    # Drift / trend configs
    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    # Strategy state
    universe_coins: list[str] = []
    universe_data: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Bot lifecycle
    # ------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        """Load smallcap universe at bot startup."""
        logger.info("[SmallCap] Loading universe...")
        # Refresh path in case file was created after module import
        universe_path = next((p for p in _UNIVERSE_CANDIDATES if p.exists()), UNIVERSE_PATH)
        if universe_path.exists():
            with open(universe_path, "r") as f:
                data = json.load(f)
            coins = data.get("coins", [])
            self.universe_coins = [c["binance_pair"] for c in coins if "binance_pair" in c]
            self.universe_data = {c["binance_pair"]: c for c in coins if "binance_pair" in c}
            logger.info(f"[SmallCap] Loaded {len(self.universe_coins)} coins from universe.")
        else:
            logger.warning(f"[SmallCap] Universe file not found at {universe_path}. Strategy will not trade.")
            self.universe_coins = []
            self.universe_data = {}

    # ------------------------------------------------------------------
    # Stake amount
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
        """Limit each position to 3% of total stake."""
        total = self.wallets.get_total_stake()
        target = total * 0.03
        if max_stake is not None:
            target = min(target, max_stake)
        if min_stake is not None:
            target = max(target, min_stake)
        return target

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Build all filters and inject universe metadata."""
        dataframe = apply_all_filters(dataframe)

        # Inject universe metadata if available
        pair = metadata.get("pair", "")
        if pair in self.universe_data:
            coin = self.universe_data[pair]
            dataframe["sc_rank"] = coin.get("percent_change_7d", 0)
            dataframe["sc_market_cap"] = coin.get("market_cap", 0)
            dataframe["sc_turnover"] = coin.get("turnover_rate", 0)
        else:
            dataframe["sc_rank"] = -999
            dataframe["sc_market_cap"] = 0
            dataframe["sc_turnover"] = 0

        return dataframe

    # ------------------------------------------------------------------
    # Entry signal
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        pair = metadata.get("pair", "")

        # Must be in universe
        if pair not in self.universe_coins:
            return dataframe

        # Technical filters
        trend_up = dataframe["above_ema30"] == 1
        not_overbought = dataframe["rsi_14"] < 75
        not_weakening = dataframe["consecutive_down_days"] < 3

        dataframe.loc[trend_up & not_overbought & not_weakening, "enter_long"] = 1
        return dataframe

    # ------------------------------------------------------------------
    # Exit signal
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        pair = metadata.get("pair", "")

        # Exit if trend reverses
        trend_down = dataframe["close"] < dataframe["ema_30"]

        # Exit if dropped out of universe (no longer in top 20)
        dropped_out = pair not in self.universe_coins

        dataframe.loc[trend_down, "exit_long"] = 1
        if dropped_out:
            dataframe.loc[:, "exit_long"] = 1
        return dataframe
