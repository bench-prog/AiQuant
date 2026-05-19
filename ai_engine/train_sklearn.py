"""
Train a scikit-learn / LightGBM model for crypto direction prediction.

Output:
  ../freqtrade/user_data/models/sklearn_model.pkl
  ../freqtrade/user_data/models/feature_config.json

Usage:
  cd ai_engine
  pip install -r requirements.txt
  python train_sklearn.py
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from data_fetcher import fetch_ohlcv_ccxt
from features import build_all_features, get_feature_columns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "freqtrade" / "user_data" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Crypto settings
SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
EXCHANGE = "binance"

# Target: predict if next period close is higher than current close
HORIZON = 1


def load_data() -> pd.DataFrame:
    """Load historical OHLCV data from Binance via ccxt."""
    df = fetch_ohlcv_ccxt(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=START_DATE,
        end_date=END_DATE,
        exchange_name=EXCHANGE,
        use_cache=True,
    )
    logger.info(f"Loaded {len(df)} rows from {EXCHANGE}.")
    return df


def prepare_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Create binary target: 1 if future return > 0, else 0."""
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_return > 0).astype(int)
    return df


def train_model(df: pd.DataFrame):
    """Train LightGBM with time-series cross-validation."""
    df = build_all_features(df)
    df = prepare_target(df)

    feature_cols = get_feature_columns(df)
    # Remove non-feature columns that might have been added
    feature_cols = [c for c in feature_cols if c not in {"target"}]

    # Drop rows with NaN in features or target
    valid = df[feature_cols + ["target"]].notnull().all(axis=1)
    df = df.loc[valid].copy()

    X = df[feature_cols]
    y = df["target"]

    logger.info(f"Features used ({len(feature_cols)}): {feature_cols[:5]}...")
    logger.info(f"Training samples: {len(X)}, Positive ratio: {y.mean():.2%}")

    # Time-series split (respect temporal order!)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[],
        )

        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(auc)
        logger.info(f"Fold {fold + 1} AUC: {auc:.4f}")

    logger.info(f"Mean CV AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    # Final model: train on all data (for deployment)
    final_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X, y)

    # Save model
    model_path = MODEL_OUTPUT_DIR / "sklearn_model.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature config (so the strategy knows which columns to use)
    config = {
        "model_type": "lightgbm",
        "feature_columns": feature_cols,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "cv_auc_mean": float(np.mean(auc_scores)),
    }
    config_path = MODEL_OUTPUT_DIR / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Feature config saved to {config_path}")

    # Quick sanity check on latest data
    latest_pred = final_model.predict_proba(X.tail(100))[:, 1]
    logger.info(f"Latest 100 predictions mean: {latest_pred.mean():.4f}")

    return final_model


if __name__ == "__main__":
    df = load_data()
    model = train_model(df)
    logger.info("Training complete.")
