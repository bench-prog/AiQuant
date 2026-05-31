"""
策略组合 (StrategyEnsemble)

加权融合多个子策略的信号：
  - AI 模型策略 (AIModelStrategy): ai_prediction ∈ [0, 1]
  - 趋势跟踪策略 (TrendFollowingStrategy): EMA 多头排列 + ADX

融合公式: ensemble_score = w_ai × ai_signal + w_trend × trend_signal
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from freqtrade.strategy import IStrategy

# Shared feature engineering
from features import build_all_features, add_higher_timeframe_features

# Ensure data/ package is importable both locally and in Docker
import sys

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

from data.service import query, merge_into  # noqa: E402
import data.service_defaults  # noqa: E402,F401

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model directory: support local dev and Docker
_MODEL_DIR = None
for candidate in [
    Path(__file__).parent.parent / "models",
    Path("/freqtrade/user_data/models"),
]:
    if candidate.exists():
        _MODEL_DIR = candidate
        break

MODEL_DIR = _MODEL_DIR or Path("/freqtrade/user_data/models")

# Ensemble weights (must sum to 1.0)
ENSEMBLE_WEIGHTS = {
    "ai": 0.60,
    "trend": 0.40,
}

ENTRY_THRESHOLD = 0.55
EXIT_THRESHOLD = 0.45


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class StrategyEnsemble(IStrategy):
    """策略组合：加权融合 AI 模型信号和趋势跟踪信号。"""

    timeframe = "4h"
    can_short = False
    process_only_new_candles = True

    # Risk
    stoploss = -0.07
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True

    # ROI
    minimal_roi = {"0": 0.06, "30": 0.04, "60": 0.02}

    # --- AI model state ---
    sklearn_model: Optional[object] = None
    feature_config: Optional[dict] = None
    model_type: str = "fallback"

    # ------------------------------------------------------------------
    # Bot lifecycle
    # ------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        """加载 AI 模型（复用 AIModelStrategy 逻辑）。"""
        logger.info("[Ensemble] Loading AI model...")
        self._load_ai_model()
        logger.info(f"[Ensemble] AI model type: {self.model_type}")

    def _load_ai_model(self) -> None:
        """加载 sklearn 模型和配置。"""
        info = {"type": "fallback"}

        sklearn_path = MODEL_DIR / "BTC_USDT" / "sklearn_model.pkl"
        if sklearn_path.exists():
            try:
                info["sklearn_model"] = joblib.load(sklearn_path)
                info["type"] = "sklearn"
            except Exception as e:
                logger.warning(f"[Ensemble] Failed to load sklearn: {e}")

        cfg_path = MODEL_DIR / "BTC_USDT" / "feature_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path, "r") as f:
                    info["feature_config"] = json.load(f)
            except Exception as e:
                logger.warning(f"[Ensemble] Failed to load config: {e}")

        self.sklearn_model = info.get("sklearn_model")
        self.feature_config = info.get("feature_config")
        self.model_type = info.get("type", "fallback")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """构建全部特征（AI + Trend 共用）。"""
        pair = metadata.get("pair", "")
        exchange_name = self.config.get("exchange", {}).get("name", "binance")

        # Merge external data
        if pair and "date" in dataframe.columns:
            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                fr_df = query("funding_rate", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, fr_df, "fundingRate")
            except Exception as e:
                logger.warning(f"[Ensemble] Failed to merge funding rate: {e}")

        # Build all features
        dataframe = build_all_features(dataframe)

        # Multi-timeframe
        try:
            informative_1d = self.dp.get_pair_dataframe(pair, "1d")
        except Exception:
            informative_1d = None
        dataframe = add_higher_timeframe_features(dataframe, df_4h=None, df_1d=informative_1d)

        # Compute sub-strategy signals
        dataframe = self._compute_ai_signal(dataframe, metadata)
        dataframe = self._compute_trend_signal(dataframe, metadata)

        # Ensemble score
        dataframe = self._compute_ensemble_score(dataframe)

        return dataframe

    # ------------------------------------------------------------------
    # AI Signal
    # ------------------------------------------------------------------
    def _compute_ai_signal(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """AI 模型信号: sklearn 分类器 predict_proba → [0, 1]。"""
        dataframe["ai_signal"] = 0.5  # default neutral

        if self.model_type != "sklearn" or self.sklearn_model is None:
            return dataframe

        feature_cols = self.feature_config.get("feature_columns", []) if self.feature_config else []
        if not feature_cols:
            feature_cols = [c for c in dataframe.columns if c not in {
                "open", "high", "low", "close", "volume", "date",
            }]

        # Fill missing columns
        for col in feature_cols:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        valid_idx = dataframe[feature_cols].notnull().all(axis=1)
        X = dataframe.loc[valid_idx, feature_cols].values

        if len(X) == 0:
            return dataframe

        try:
            if hasattr(self.sklearn_model, "predict_proba"):
                probs = self.sklearn_model.predict_proba(X)[:, 1]
            else:
                probs = self.sklearn_model.predict(X)
            dataframe.loc[valid_idx, "ai_signal"] = probs
        except Exception as e:
            logger.warning(f"[Ensemble] AI inference failed: {e}")

        return dataframe

    # ------------------------------------------------------------------
    # Trend Signal
    # ------------------------------------------------------------------
    def _compute_trend_signal(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """趋势跟踪信号: EMA 多头排列 + ADX → 映射到 [0, 1]。"""
        # EMAs (default values matching TrendFollowingStrategy)
        ema_s = dataframe["close"].ewm(span=10).mean()
        ema_m = dataframe["close"].ewm(span=30).mean()
        ema_l = dataframe["close"].ewm(span=100).mean()

        # ADX (using pre-computed if available, else compute inline)
        if "adx_14" in dataframe.columns:
            adx = dataframe["adx_14"]
        else:
            # Simple inline ADX
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
            adx = dx.ewm(span=14).mean()

        # Signal mapping
        trend_up = (ema_s > ema_m) & (ema_m > ema_l)
        price_ok = dataframe["close"] > ema_s
        trending = adx > 20

        # Map to [0, 1]
        # 1.0: full alignment (EMA bull + ADX strong + price above short EMA)
        # 0.5: partial (EMA bull but ADX weak)
        # 0.0: no trend
        score = pd.Series(0.0, index=dataframe.index)
        score.loc[trend_up & price_ok] = 0.5
        score.loc[trend_up & trending & price_ok] = 1.0

        dataframe["trend_signal"] = score
        return dataframe

    # ------------------------------------------------------------------
    # Ensemble Score
    # ------------------------------------------------------------------
    def _compute_ensemble_score(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """加权融合: ensemble_score = w_ai × ai_signal + w_trend × trend_signal。"""
        w_ai = ENSEMBLE_WEIGHTS["ai"]
        w_trend = ENSEMBLE_WEIGHTS["trend"]

        # Sanity check: clip to [0, 1]
        ai = dataframe["ai_signal"].clip(0.0, 1.0).fillna(0.5)
        trend = dataframe["trend_signal"].clip(0.0, 1.0).fillna(0.0)

        dataframe["ensemble_score"] = w_ai * ai + w_trend * trend
        return dataframe

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["enter_long"] = 0
        score = dataframe["ensemble_score"]
        dataframe.loc[score > ENTRY_THRESHOLD, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe["exit_long"] = 0
        score = dataframe["ensemble_score"]
        dataframe.loc[score < EXIT_THRESHOLD, "exit_long"] = 1
        return dataframe
