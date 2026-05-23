"""
GoldPulseStrategy (v1) — 黄金脉冲传导策略

当黄金（XAUUSDT）在30分钟内出现剧烈波动（>1% 或 <-1%），
而原油（CLUSDT）未能同步跟进时，押注原油的滞后补涨/补跌。

文献支撑:
- 黄金短时间波动>1%时，原油同向联动概率 89.5%
- Zhang & Wei (2010): 黄金日内冲击对原油影响 5x 强于反向

交易标的: CL/USDT:USDT (原油永续合约)
信号来源: XAU/USDT:USDT (黄金永续合约)

Freqtrade 数据要求:
  download-data --pairs CL/USDT:USDT XAU/USDT:USDT --timeframe 5m
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)

# ==================== 策略常量 ====================
PULSE_PERIOD = 6                    # 5m x 6 = 30m
GOLD_PULSE_THRESHOLD = 0.01         # 黄金 30m 波动阈值 1%
CL_LAG_THRESHOLD = 0.003            # 原油 30m 滞后阈值 0.3%
MAX_CL_VOLATILITY = 0.02            # 原油波动率过滤阈值 2%
HARD_STOP = -0.015                  # 硬止损 1.5%
MAX_HOLD_HOURS = 4                  # 时间止损 4 小时

# 黄金数据文件搜索路径（按优先级）
_GOLD_PATH_CANDIDATES = [
    Path("/freqtrade/user_data/data/binance/XAU_USDT_USDT-5m.feather"),
    Path("/freqtrade/user_data/data/binance/XAU_USDT-5m.feather"),
    Path(__file__).parent.parent / "data" / "binance" / "XAU_USDT_USDT-5m.feather",
    Path(__file__).parent.parent / "data" / "binance" / "XAU_USDT-5m.feather",
]


class GoldPulseStrategy(IStrategy):
    """
    黄金脉冲传导策略。

    只交易 CL/USDT:USDT，通过监控 XAU/USDT:USDT 的 30 分钟涨跌幅生成信号。
    """

    # ---- Freqtrade 核心配置 ----
    timeframe = "5m"
    stoploss = HARD_STOP              # 全局硬止损 -1.5%
    can_short = True
    startup_candle_count = 100
    max_open_trades = 1

    use_exit_signal = False
    exit_profit_only = False

    # ---- 缓存的黄金数据（类级别，避免重复加载）----
    _gold_data: pd.DataFrame | None = None
    _gold_path: Path | None = None

    # ================================================================
    # 数据加载
    # ================================================================

    def _find_gold_data_path(self) -> Optional[Path]:
        """按优先级搜索黄金数据 feather 文件。"""
        for path in _GOLD_PATH_CANDIDATES:
            if path.exists():
                return path
        return None

    def _load_gold_data(self) -> None:
        """懒加载黄金历史 OHLCV 数据（仅在首次调用时执行）。"""
        if GoldPulseStrategy._gold_data is not None:
            return

        path = self._find_gold_data_path()
        if path is None:
            searched = "\n  ".join(str(p) for p in _GOLD_PATH_CANDIDATES)
            logger.warning(
                f"[GoldPulse] 未找到黄金数据文件。已搜索:\n  {searched}\n"
                f"请先运行: docker compose run --rm freqtrade "
                f"download-data --pairs XAU/USDT:USDT --timeframe 5m"
            )
            return

        try:
            import pyarrow.feather as feather

            df = feather.read_feather(path)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            GoldPulseStrategy._gold_data = df[["date", "close"]].copy()
            GoldPulseStrategy._gold_path = path
            logger.info(
                f"[GoldPulse] 加载黄金数据: {path.name} ({len(df)} rows)"
            )
        except Exception as e:
            logger.error(f"[GoldPulse] 加载黄金数据失败: {e}")

    def bot_start(self, **kwargs) -> None:
        self._load_gold_data()

    # ================================================================
    # 指标计算
    # ================================================================

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        将黄金数据按时间合并到原油 DataFrame，并计算脉冲信号。

        回测时 bot_start() 不会被调用，因此每次 populate_indicators 都尝试懒加载。
        """
        self._load_gold_data()

        pair = metadata.get("pair", "")
        if not pair.startswith("CL"):
            return dataframe

        if GoldPulseStrategy._gold_data is None or GoldPulseStrategy._gold_data.empty:
            logger.warning("[GoldPulse] 黄金数据不可用，跳过信号计算")
            return dataframe

        df = dataframe.copy()

        # 统一 datetime 精度，避免 merge_asof 类型不匹配
        df["date"] = pd.to_datetime(df["date"], utc=True).astype("datetime64[ms, UTC]")

        gold_df = GoldPulseStrategy._gold_data.copy()
        gold_df["date"] = pd.to_datetime(gold_df["date"], utc=True).astype("datetime64[ms, UTC]")

        # 按时间向后填充合并（黄金数据时间戳 <= 原油时间戳的最新值）
        df = pd.merge_asof(
            df.sort_values("date").reset_index(drop=True),
            gold_df.sort_values("date").reset_index(drop= True)[["date", "close"]],
            on="date",
            direction="backward",
            suffixes=("", "_gold"),
        )

        if "close_gold" not in df.columns:
            logger.warning("[GoldPulse] merge_asof 未生成 gold_close 列")
            return dataframe

        df = df.rename(columns={"close_gold": "gold_close"})

        # ---- 核心特征 ----
        # 30 分钟涨跌幅（6 根 5m K线）
        df["cl_return_30m"] = df["close"].pct_change(PULSE_PERIOD)
        df["gold_return_30m"] = df["gold_close"].pct_change(PULSE_PERIOD)

        # 黄金脉冲标记
        df["gold_pulse_up"] = df["gold_return_30m"] > GOLD_PULSE_THRESHOLD
        df["gold_pulse_down"] = df["gold_return_30m"] < -GOLD_PULSE_THRESHOLD

        # 原油滞后标记
        df["cl_lag_up"] = df["cl_return_30m"] < CL_LAG_THRESHOLD       # 黄金涨但原油涨得少
        df["cl_lag_down"] = df["cl_return_30m"] > -CL_LAG_THRESHOLD    # 黄金跌但原油跌得少

        # 原油波动率过滤（20 根 K线 ≈ 100 分钟）
        df["cl_volatility_20"] = df["close"].pct_change().rolling(20).std()

        # 信号就绪标记（避免 NaN 导致误触发）
        df["signal_ready"] = (
            df["gold_return_30m"].notna()
            & df["cl_return_30m"].notna()
            & df["cl_volatility_20"].notna()
        )

        return df

    # ================================================================
    # 入场信号
    # ================================================================

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0
        dataframe.loc[:, "enter_short"] = 0

        pair = metadata.get("pair", "")
        if not pair.startswith("CL"):
            return dataframe

        if "signal_ready" not in dataframe.columns:
            return dataframe

        ready = dataframe["signal_ready"]
        vol_ok = dataframe["cl_volatility_20"] < MAX_CL_VOLATILITY

        # 做多: 黄金脉冲上涨 + 原油滞后
        long_cond = (
            ready
            & vol_ok
            & dataframe["gold_pulse_up"]
            & dataframe["cl_lag_up"]
        )

        # 做空: 黄金脉冲下跌 + 原油滞后
        short_cond = (
            ready
            & vol_ok
            & dataframe["gold_pulse_down"]
            & dataframe["cl_lag_down"]
        )

        dataframe.loc[long_cond, "enter_long"] = 1
        dataframe.loc[short_cond, "enter_short"] = 1

        # 日志（仅在最后一根 K线有信号时输出）
        if not dataframe.empty:
            if long_cond.iloc[-1]:
                logger.info(
                    f"[GoldPulse] LONG signal @ {dataframe['date'].iloc[-1]}: "
                    f"gold_30m={dataframe['gold_return_30m'].iloc[-1]:.4f}, "
                    f"cl_30m={dataframe['cl_return_30m'].iloc[-1]:.4f}"
                )
            elif short_cond.iloc[-1]:
                logger.info(
                    f"[GoldPulse] SHORT signal @ {dataframe['date'].iloc[-1]}: "
                    f"gold_30m={dataframe['gold_return_30m'].iloc[-1]:.4f}, "
                    f"cl_30m={dataframe['cl_return_30m'].iloc[-1]:.4f}"
                )

        return dataframe

    # ================================================================
    # 出场信号
    # ================================================================

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0
        dataframe.loc[:, "exit_short"] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """
        时间止损: 持仓超过 MAX_HOLD_HOURS 强制平仓。
        硬止损由全局 stoploss 参数处理。
        """
        hold_seconds = (current_time - trade.open_date_utc).total_seconds()
        if hold_seconds > MAX_HOLD_HOURS * 3600:
            return f"time_stop_{MAX_HOLD_HOURS}h"
        return None
