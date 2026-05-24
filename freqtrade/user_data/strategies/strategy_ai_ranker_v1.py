"""
AiQuant 截面排序策略 (AIModelRankerStrategy)

在所有候选币种中预测未来收益率，做多预测收益最高的 Top 3。

与 AIModelStrategy 的区别：
  - 使用回归模型（预测连续收益率）
  - 通过截面排序选择币种
  - 天然对冲市场 Beta
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy, IntParameter

import sys

_data_dir = None
for candidate in [
    Path(__file__).parent.parent.parent.parent / "data",
    Path.cwd() / "data",
    Path("/freqtrade/data"),
]:
    if candidate.exists():
        _data_dir = candidate
        break

if _data_dir and str(_data_dir.parent) not in sys.path:
    sys.path.insert(0, str(_data_dir.parent))

from data.service import query, merge_into  # noqa: E402
import data.service_defaults  # noqa: E402

from features import build_all_features, get_feature_columns  # noqa: E402

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
RANKER_MODEL_PATH = MODEL_DIR / "ranker" / "ranker_model.pkl"
RANKER_CONFIG_PATH = MODEL_DIR / "ranker" / "feature_config.json"


class AIModelRankerStrategy(IStrategy):
    """截面排序策略：预测所有币种收益率，做多 Top 3。"""

    timeframe = "1w"
    can_short = False
    process_only_new_candles = True

    # Risk — 周线级别
    stoploss = -0.15
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.08
    trailing_only_offset_is_reached = True

    # ROI — 周线目标
    minimal_roi = {"0": 0.15, "4": 0.08, "8": 0.04}

    top_n = IntParameter(2, 5, default=3, space="buy")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._ranker_model = None
        self._feature_columns: list[str] = []
        self._pair_scores: dict[str, float] = {}

    def informative_pairs(self):
        return []

    def bot_start(self, **kwargs) -> None:
        if RANKER_MODEL_PATH.exists():
            try:
                self._ranker_model = joblib.load(RANKER_MODEL_PATH)
                logger.info(f"[Ranker] Loaded model from {RANKER_MODEL_PATH}")
            except Exception as e:
                logger.error(f"[Ranker] Failed to load model: {e}")

        if RANKER_CONFIG_PATH.exists():
            try:
                with open(RANKER_CONFIG_PATH) as f:
                    cfg = json.load(f)
                self._feature_columns = cfg.get("feature_columns", [])
                logger.info(f"[Ranker] {len(self._feature_columns)} feature columns")
            except Exception as e:
                logger.error(f"[Ranker] Failed to load config: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata.get("pair", "")
        exchange_name = self.config.get("exchange", {}).get("name", "binance")

        if pair and "date" in dataframe.columns:
            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                fr_df = query("funding_rate", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, fr_df, "fundingRate")
            except Exception:
                pass
            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                oi_df = query("open_interest", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, oi_df, "openInterest")
            except Exception:
                pass

        dataframe = build_all_features(dataframe)

        if self._ranker_model is not None and self._feature_columns:
            available = [c for c in self._feature_columns if c in dataframe.columns]
            if len(available) == len(self._feature_columns):
                valid_mask = dataframe[available].notnull().all(axis=1)
                X = dataframe.loc[valid_mask, available]
                if len(X) > 0:
                    preds = self._ranker_model.predict(X)
                    dataframe.loc[valid_mask, "pred_return"] = np.nan
                    dataframe.loc[valid_mask, "pred_return"] = preds
                    self._pair_scores[pair] = float(preds[-1]) if len(preds) > 0 else 0.0

        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        if "pred_return" in dataframe.columns:
            dataframe.loc[dataframe["pred_return"] > 0.005, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
