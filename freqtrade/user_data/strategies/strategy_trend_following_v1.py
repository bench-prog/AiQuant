"""
趋势跟踪策略 (TrendFollowingStrategy)

纯规则驱动，零 ML 依赖：
  - 入场: EMA(20) > EMA(50) > EMA(200) 多头排列
  - 过滤: ADX > 20（只做趋势市）
  - 退出: EMA(20) < EMA(50) 或止损
  - 止损: 2x ATR 跟踪止损
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(IStrategy):
    timeframe = "4h"
    can_short = False
    process_only_new_candles = True

    # Risk
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    # ROI
    minimal_roi = {"0": 0.10, "24": 0.05, "48": 0.02}

    # Hyperopt
    adx_threshold = IntParameter(15, 30, default=20, space="buy")
    ema_short = IntParameter(5, 20, default=10, space="buy")
    ema_mid = IntParameter(20, 50, default=30, space="buy")
    ema_long = IntParameter(80, 150, default=100, space="buy")

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # EMAs
        dataframe["ema_short"] = dataframe["close"].ewm(span=self.ema_short.value).mean()
        dataframe["ema_mid"] = dataframe["close"].ewm(span=self.ema_mid.value).mean()
        dataframe["ema_long"] = dataframe["close"].ewm(span=self.ema_long.value).mean()

        # ADX
        high, low, close = dataframe["high"], dataframe["low"], dataframe["close"]
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
        dataframe["adx"] = dx.ewm(span=14).mean()

        # ATR for stop
        dataframe["atr"] = atr

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0

        # 多头排列: ema_short > ema_mid > ema_long
        trend_up = (
            (dataframe["ema_short"] > dataframe["ema_mid"])
            & (dataframe["ema_mid"] > dataframe["ema_long"])
        )
        # ADX 过滤
        trending = dataframe["adx"] > self.adx_threshold.value
        # 价格在短均线上方
        price_ok = dataframe["close"] > dataframe["ema_short"]

        dataframe.loc[trend_up & trending & price_ok, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0

        # 趋势反转: ema_short < ema_mid
        trend_down = dataframe["ema_short"] < dataframe["ema_mid"]
        dataframe.loc[trend_down, "exit_long"] = 1

        return dataframe

    def custom_stoploss(
        self, pair: str, trade, current_time, current_rate, current_profit,
        after_fill: bool, **kwargs
    ) -> Optional[float]:
        # 使用 ATR 动态止损
        if after_fill:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) == 0:
            return None

        last = dataframe.iloc[-1]
        atr = last.get("atr", 0)
        close = last.get("close", current_rate)

        if atr > 0 and close > 0:
            atr_pct = atr / close
            return -max(2.0 * atr_pct, 0.03)  # 至少 3% 止损

        return None
