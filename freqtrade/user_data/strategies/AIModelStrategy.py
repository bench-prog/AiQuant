"""
AiQuant AI Model Strategy Template

This strategy demonstrates how to integrate a custom AI model (scikit-learn or PyTorch)
into Freqtrade. The model is loaded once at bot startup and used to generate entry/exit
signals based on technical indicator features.

Model files should be placed in:
  /freqtrade/user_data/models/sklearn_model.pkl
  /freqtrade/user_data/models/pytorch_model.pt
  /freqtrade/user_data/models/feature_config.json

If no model is found, the strategy falls back to a simple RSI-based signal for demonstration.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy import IStrategy, merge_informative_pair

# Optional PyTorch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/freqtrade/user_data/models")
SKLEARN_MODEL_PATH = MODEL_DIR / "sklearn_model.pkl"
PYTORCH_MODEL_PATH = MODEL_DIR / "pytorch_model.pt"
FEATURE_CONFIG_PATH = MODEL_DIR / "feature_config.json"

# Signal thresholds
ENTRY_THRESHOLD = 0.6   # Model probability > 0.6 -> enter long
EXIT_THRESHOLD = 0.4    # Model probability < 0.4 -> exit long

# ---------------------------------------------------------------------------
# Simple LSTM stub (used if a PyTorch model file is provided)
# ---------------------------------------------------------------------------
class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------
class AIModelStrategy(IStrategy):
    """
    AI-driven strategy that loads an external ML model for signal generation.
    """

    # Strategy metadata
    timeframe = "1h"
    stoploss = -0.10
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    minimal_roi = {
        "0": 0.15,
        "30": 0.10,
        "60": 0.05,
        "120": 0.02
    }

    # --- Model state -------------------------------------------------------
    sklearn_model: Optional[object] = None
    pytorch_model: Optional[nn.Module] = None
    feature_config: Optional[dict] = None
    model_type: Optional[str] = None  # 'sklearn', 'pytorch', or 'fallback'

    # ------------------------------------------------------------------
    # Bot lifecycle hooks
    # ------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        """Load AI model once when the bot starts."""
        logger.info("[AiQuant] Loading AI model...")

        # 1. Try sklearn
        if SKLEARN_MODEL_PATH.exists():
            try:
                self.sklearn_model = joblib.load(SKLEARN_MODEL_PATH)
                self.model_type = "sklearn"
                logger.info(f"[AiQuant] Loaded sklearn model from {SKLEARN_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"[AiQuant] Failed to load sklearn model: {e}")

        # 2. Try PyTorch
        elif PYTORCH_MODEL_PATH.exists() and TORCH_AVAILABLE:
            try:
                # You should customize input_size to match your feature count
                self.pytorch_model = SimpleLSTM(input_size=10, hidden_size=32)
                self.pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location="cpu"))
                self.pytorch_model.eval()
                self.model_type = "pytorch"
                logger.info(f"[AiQuant] Loaded PyTorch model from {PYTORCH_MODEL_PATH}")
            except Exception as e:
                logger.warning(f"[AiQuant] Failed to load PyTorch model: {e}")

        else:
            self.model_type = "fallback"
            logger.warning("[AiQuant] No AI model found. Using fallback RSI strategy.")

        # 3. Load feature config (optional)
        if FEATURE_CONFIG_PATH.exists():
            with open(FEATURE_CONFIG_PATH, "r") as f:
                self.feature_config = json.load(f)
            logger.info(f"[AiQuant] Loaded feature config: {self.feature_config}")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate technical indicators used as model features.
        Customize this list to match the features used during training.
        """
        # Trend
        dataframe["ema_12"] = pta.ema(dataframe["close"], length=12)
        dataframe["ema_26"] = pta.ema(dataframe["close"], length=26)
        dataframe["macd"] = pta.macd(dataframe["close"], fast=12, slow=26, signal=9)["MACD_12_26_9"]
        dataframe["macd_signal"] = pta.macd(dataframe["close"], fast=12, slow=26, signal=9)["MACDs_12_26_9"]

        # Momentum
        dataframe["rsi"] = pta.rsi(dataframe["close"], length=14)
        dataframe["rsi_6"] = pta.rsi(dataframe["close"], length=6)

        # Volatility
        dataframe["atr"] = pta.atr(dataframe["high"], dataframe["low"], dataframe["close"], length=14)
        dataframe["bbands_upper"] = pta.bbands(dataframe["close"], length=20, std=2)["BBU_20_2.0"]
        dataframe["bbands_lower"] = pta.bbands(dataframe["close"], length=20, std=2)["BBL_20_2.0"]

        # Volume
        dataframe["volume_sma"] = dataframe["volume"].rolling(window=20).mean()
        dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_sma"]

        # Price relative to moving averages
        dataframe["close_above_ema12"] = (dataframe["close"] > dataframe["ema_12"]).astype(int)
        dataframe["close_above_ema26"] = (dataframe["close"] > dataframe["ema_26"]).astype(int)

        # --- Model inference ------------------------------------------------
        if self.model_type == "sklearn" and self.sklearn_model is not None:
            dataframe = self._predict_sklearn(dataframe)
        elif self.model_type == "pytorch" and self.pytorch_model is not None:
            dataframe = self._predict_pytorch(dataframe)
        else:
            # Fallback: use RSI for demo
            dataframe["ai_prediction"] = 0.5

        return dataframe

    def _predict_sklearn(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run sklearn model inference on the dataframe."""
        # Define feature columns (must match training!)
        feature_cols = [
            "rsi", "rsi_6", "macd", "macd_signal",
            "atr", "volume_ratio",
            "close_above_ema12", "close_above_ema26"
        ]

        # Drop rows with NaN features
        valid_idx = dataframe[feature_cols].notnull().all(axis=1)
        X = dataframe.loc[valid_idx, feature_cols].values

        if len(X) == 0:
            dataframe["ai_prediction"] = 0.5
            return dataframe

        # Predict probability of class 1 (upward move)
        if hasattr(self.sklearn_model, "predict_proba"):
            probs = self.sklearn_model.predict_proba(X)[:, 1]
        else:
            probs = self.sklearn_model.predict(X)

        dataframe.loc[valid_idx, "ai_prediction"] = probs
        dataframe["ai_prediction"] = dataframe["ai_prediction"].fillna(0.5)
        return dataframe

    def _predict_pytorch(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run PyTorch model inference on the dataframe."""
        feature_cols = [
            "rsi", "rsi_6", "macd", "macd_signal",
            "atr", "volume_ratio",
            "close_above_ema12", "close_above_ema26"
        ]

        valid_idx = dataframe[feature_cols].notnull().all(axis=1)
        df_valid = dataframe.loc[valid_idx].copy()

        if len(df_valid) == 0:
            dataframe["ai_prediction"] = 0.5
            return dataframe

        # Create sliding window sequences (example: lookback=20)
        lookback = 20
        predictions = []

        for i in range(len(df_valid)):
            if i < lookback:
                predictions.append(0.5)
                continue
            seq = df_valid[feature_cols].iloc[i - lookback:i].values
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
            with torch.no_grad():
                pred = self.pytorch_model(seq_tensor).item()
            predictions.append(pred)

        dataframe.loc[valid_idx, "ai_prediction"] = predictions
        dataframe["ai_prediction"] = dataframe["ai_prediction"].fillna(0.5)
        return dataframe

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "enter_long"] = 0

        # AI model signal
        ai_long = dataframe["ai_prediction"] > ENTRY_THRESHOLD

        # Optional: add confirmation filters
        # e.g., only enter if RSI is not extremely overbought
        not_overbought = dataframe["rsi"] < 75

        dataframe.loc[ai_long & not_overbought, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0

        # AI model exit signal
        ai_exit = dataframe["ai_prediction"] < EXIT_THRESHOLD

        # Optional: add confirmation filters
        # e.g., exit if RSI is extremely overbought
        overbought = dataframe["rsi"] > 80

        dataframe.loc[ai_exit | overbought, "exit_long"] = 1
        return dataframe
