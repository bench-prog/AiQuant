"""
Train a scikit-learn / LightGBM model for crypto direction prediction.

Uses a strict temporal train/test split to avoid look-ahead bias:
  Training: 2022-01-01 ~ 2023-12-31
  Test (for Freqtrade backtest): 2024-01-01 ~ 2024-12-31

Output:
  ../freqtrade/user_data/models/sklearn_model.pkl
  ../freqtrade/user_data/models/feature_config.json

Usage:
  cd research
  pip install -r requirements.txt
  python train_classifier.py
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
_data_dir = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(_data_dir))
from data.market_data import fetch_funding_rate, fetch_ohlcv_ccxt, fetch_open_interest
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# Import shared feature engineering from strategies directory
_strategies_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "strategies"
sys.path.insert(0, str(_strategies_dir))
from features import build_all_features, get_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "freqtrade" / "user_data" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "BTC/USDT"
TIMEFRAME = "1h"
TRAIN_START = "2022-01-01"
TRAIN_END = "2023-12-31"      # Hard cutoff: model never sees 2024 data
FULL_END = "2024-12-31"       # Used only to cache data for convenience
EXCHANGE = "binance"

HORIZON = 1


def load_data() -> pd.DataFrame:
    """Load historical OHLCV data from Binance via ccxt."""
    df = fetch_ohlcv_ccxt(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start_date=TRAIN_START,
        end_date=FULL_END,
        exchange_name=EXCHANGE,
        use_cache=True,
    )
    logger.info(f"Loaded {len(df)} rows from {EXCHANGE}.")
    return df


def merge_external_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge funding rate and open interest data into the OHLCV dataframe.
    Uses merge_asof for time-alignment without look-ahead bias.
    """
    df = df.copy()

    # --- Funding Rate (8h intervals) ---
    try:
        fr_df = fetch_funding_rate(
            symbol=SYMBOL,
            start_date=TRAIN_START,
            end_date=FULL_END,
            exchange_name=EXCHANGE,
            use_cache=True,
        )
        if not fr_df.empty:
            df = pd.merge_asof(
                df,
                fr_df,
                on="date",
                direction="backward",
            )
            df["fundingRate"] = df["fundingRate"].ffill()
            logger.info(f"Merged funding rate data ({len(fr_df)} records).")
        else:
            logger.warning("No funding rate data returned; proceeding without it.")
    except Exception as e:
        logger.warning(f"Failed to fetch/merge funding rate: {e}")

    # --- Open Interest (1h snapshots) ---
    try:
        oi_df = fetch_open_interest(
            symbol=SYMBOL,
            start_date=TRAIN_START,
            end_date=FULL_END,
            exchange_name=EXCHANGE,
            timeframe=TIMEFRAME,
            use_cache=True,
        )
        if not oi_df.empty:
            df = pd.merge_asof(
                df,
                oi_df,
                on="date",
                direction="backward",
            )
            df["openInterest"] = df["openInterest"].ffill()
            logger.info(f"Merged open interest data ({len(oi_df)} records).")
        else:
            logger.warning("No open interest data returned; proceeding without it.")
    except Exception as e:
        logger.warning(f"Failed to fetch/merge open interest: {e}")

    # Final safety: any remaining NaNs in external columns -> 0
    for col in ["fundingRate", "openInterest"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def prepare_target(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """Create binary target: 1 if future return > 0, else 0."""
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_return > 0).astype(int)
    return df


def train_model(df: pd.DataFrame):
    """Train LightGBM with time-series cross-validation on TRAIN period only."""
    df = build_all_features(df)
    df = prepare_target(df)

    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if c != "target"]

    # STRICT TEMPORAL SPLIT: only use data up to TRAIN_END for training
    train_mask = df["date"] < pd.Timestamp(TRAIN_END, tz="UTC")
    df_train = df.loc[train_mask].copy()

    valid = df_train[feature_cols + ["target"]].notnull().all(axis=1)
    df_train = df_train.loc[valid].copy()

    if len(df_train) == 0:
        logger.error("No valid training samples after dropping NaNs. Check feature engineering.")
        raise ValueError("Dataset is empty after NaN drop. Inspect feature columns for all-NaN values.")

    X = df_train[feature_cols]
    y = df_train["target"]

    logger.info(f"Training period: {df_train['date'].min()} ~ {df_train['date'].max()}")
    logger.info(f"Features used ({len(feature_cols)}): {feature_cols[:5]}...")
    logger.info(f"Training samples: {len(X)}, Positive ratio: {y.mean():.2%}")

    # Time-series CV (respect temporal order!)
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=5,
            reg_alpha=0.0,
            reg_lambda=0.0,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(auc)
        logger.info(f"Fold {fold + 1} AUC: {auc:.4f}")

    logger.info(f"Mean CV AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    # Final model: train on entire TRAIN period
    final_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=5,
        reg_alpha=0.0,
        reg_lambda=0.0,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    final_model.fit(X, y)

    # Save drift baseline from training set predictions
    train_pred = final_model.predict_proba(X)[:, 1]
    baseline = {
        "mean": float(train_pred.mean()),
        "std": float(train_pred.std()),
        "quantiles": {
            str(q): float(v) for q, v in zip(
                [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                np.quantile(train_pred, [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
            )
        },
        "hist_counts": np.histogram(train_pred, bins=20, range=(0, 1))[0].tolist(),
        "hist_bins": 20,
        "hist_range": [0, 1],
        "n_samples": len(train_pred),
    }
    baseline_path = MODEL_OUTPUT_DIR / "drift_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Drift baseline saved to {baseline_path}")

    # Evaluate on held-out TEST period (2024) — for sanity check only
    test_mask = df["date"] >= pd.Timestamp(TRAIN_END, tz="UTC")
    df_test = df.loc[test_mask].copy()
    test_valid = df_test[feature_cols + ["target"]].notnull().all(axis=1)
    df_test = df_test.loc[test_valid]

    if len(df_test) > 0:
        X_test = df_test[feature_cols]
        y_test = df_test["target"]
        test_pred = final_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        logger.info(f"=== Held-out TEST AUC (2024): {test_auc:.4f} ===")
    else:
        logger.warning("No test data available for evaluation.")

    # Save model
    model_path = MODEL_OUTPUT_DIR / "sklearn_model.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save feature config
    config = {
        "model_type": "lightgbm",
        "feature_columns": feature_cols,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "train_range": [TRAIN_START, TRAIN_END],
        "cv_auc_mean": float(np.mean(auc_scores)),
        "test_auc_2024": float(test_auc) if len(df_test) > 0 else None,
    }
    config_path = MODEL_OUTPUT_DIR / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Feature config saved to {config_path}")

    return final_model


if __name__ == "__main__":
    df = load_data()
    df = merge_external_data(df)
    model = train_model(df)
    logger.info("Training complete.")
