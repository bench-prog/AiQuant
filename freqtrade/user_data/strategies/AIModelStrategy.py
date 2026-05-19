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
from freqtrade.strategy import IStrategy

# Shared feature engineering (same code used in training)
from feature_engineering import build_all_features

# Drift detection utilities (pure numpy/pandas, no external deps)
from drift_utils import compute_psi, append_drift_alert, send_telegram_alert

# Optional PyTorch import with graceful fallback
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

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

except ImportError:
    TORCH_AVAILABLE = False
    SimpleLSTM = None  # type: ignore

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
    drift_baseline: Optional[dict] = None
    prediction_buffer: list = []
    candle_count: int = 0

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

        # 4. Load drift baseline
        self._load_drift_baseline()
        self.prediction_buffer = []
        self.candle_count = 0

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Build all features and run model inference.
        Uses the same feature_engineering module as training to ensure consistency.

        NOTE: Freqtrade does not natively inject fundingRate/openInterest columns
        into the strategy dataframe. If these columns are present (e.g. via custom
        data-provider or dataframe prepopulation), build_all_features will compute
        the derived features. If they are absent, the feature functions gracefully
        skip them and the strategy continues to work with the existing features.
        """
        # Build all features (identical to training pipeline)
        dataframe = build_all_features(dataframe)

        # --- Model inference ------------------------------------------------
        if self.model_type == "sklearn" and self.sklearn_model is not None:
            dataframe = self._predict_sklearn(dataframe)
        elif self.model_type == "pytorch" and self.pytorch_model is not None:
            dataframe = self._predict_pytorch(dataframe)
        else:
            # Fallback: neutral prediction
            dataframe["ai_prediction"] = 0.5

        # --- Drift monitoring ---
        if self.model_type != "fallback" and self.drift_baseline is not None:
            dataframe = self._update_drift_monitor(dataframe, metadata)

        return dataframe

    def _predict_sklearn(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Run sklearn model inference on the dataframe."""
        # Use exact feature list from training config
        feature_cols = self.feature_config.get("feature_columns", []) if self.feature_config else []
        if not feature_cols:
            logger.warning("[AiQuant] No feature_columns in config. Using all non-OHLCV columns.")
            feature_cols = [c for c in dataframe.columns if c not in {"open", "high", "low", "close", "volume", "date"}]

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
        feature_cols = self.feature_config.get("feature_columns", []) if self.feature_config else []
        if not feature_cols:
            feature_cols = [c for c in dataframe.columns if c not in {"open", "high", "low", "close", "volume", "date"}]

        valid_idx = dataframe[feature_cols].notnull().all(axis=1)
        df_valid = dataframe.loc[valid_idx].copy()

        if len(df_valid) == 0:
            dataframe["ai_prediction"] = 0.5
            return dataframe

        lookback = self.feature_config.get("lookback", 20) if self.feature_config else 20
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
    # Drift monitoring
    # ------------------------------------------------------------------
    def _load_drift_baseline(self) -> None:
        baseline_path = MODEL_DIR / "drift_baseline.json"
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                self.drift_baseline = json.load(f)
            logger.info(f"[AiQuant] Loaded drift baseline from {baseline_path}")
        else:
            self.drift_baseline = None
            logger.warning("[AiQuant] No drift_baseline.json found. Drift monitoring disabled.")

    def _update_drift_monitor(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Update prediction buffer and check for drift periodically."""
        latest_pred = dataframe["ai_prediction"].iloc[-1] if dataframe["ai_prediction"].notna().any() else None
        if latest_pred is not None and not np.isnan(latest_pred):
            self.prediction_buffer.append(float(latest_pred))
            if len(self.prediction_buffer) > self.DRIFT_WINDOW_SIZE:
                self.prediction_buffer.pop(0)

        self.candle_count += 1

        # Default PSI column
        dataframe["drift_psi"] = np.nan

        if self.candle_count % self.DRIFT_CHECK_INTERVAL != 0:
            return dataframe
        if len(self.prediction_buffer) < 100:
            return dataframe

        psi = compute_psi(
            self.drift_baseline["hist_counts"],
            self.prediction_buffer,
            bins=self.drift_baseline.get("hist_bins", 20),
            range_=tuple(self.drift_baseline.get("hist_range", [0, 1])),
        )
        dataframe["drift_psi"] = psi

        if psi > self.DRIFT_PSI_THRESHOLD:
            pair = metadata.get("pair", "N/A")
            msg = (
                f"🚨 <b>AiQuant Model Drift Alert</b>\n"
                f"Pair: {pair}\n"
                f"PSI: {psi:.4f} (threshold: {self.DRIFT_PSI_THRESHOLD})\n"
                f"Buffer: {len(self.prediction_buffer)} samples\n"
                f"Model: {self.model_type}"
            )
            logger.warning(f"[DRIFT] {msg}")
            append_drift_alert(msg, {"psi": psi, "pair": pair, "model_type": self.model_type})

            # Try Freqtrade native Telegram
            try:
                if hasattr(self.dp, "send_msg"):
                    self.dp.send_msg(msg, always_send=True)
            except Exception as e:
                logger.warning(f"[DRIFT] dp.send_msg failed: {e}")

            # Fallback: direct Telegram API
            send_telegram_alert(msg)

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
        not_overbought = dataframe["rsi_14"] < 75

        dataframe.loc[ai_long & not_overbought, "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[:, "exit_long"] = 0

        # AI model exit signal
        ai_exit = dataframe["ai_prediction"] < EXIT_THRESHOLD

        # Optional: add confirmation filters
        # e.g., exit if RSI is extremely overbought
        overbought = dataframe["rsi_14"] > 80

        dataframe.loc[ai_exit | overbought, "exit_long"] = 1
        return dataframe
