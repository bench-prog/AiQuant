"""
训练 scikit-learn / LightGBM 方向预测模型。

采用严格的时间切分避免前瞻偏差:
  训练集: 2022-01-01 ~ 2023-12-31
  测试集 (仅用于 Freqtrade 回测评估): 2024-01-01 ~ 2024-12-31

输出:
  ../freqtrade/user_data/models/sklearn_model.pkl
  ../freqtrade/user_data/models/feature_config_lightgbm.json

用法:
  cd research
  pip install -r requirements.txt
  python train_classifier.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from research.training_config import (  # noqa: E402
    SYMBOL,
    TIMEFRAME,
    TRAIN_START,
    TRAIN_END,
    HORIZON,
    MODEL_OUTPUT_DIR,
)
from research.data_utils import load_training_data, merge_external_data  # noqa: E402

# Import shared feature engineering from strategies directory
_strategies_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "strategies"
sys.path.insert(0, str(_strategies_dir))
from features import build_all_features, get_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_target(
    df: pd.DataFrame, horizon: int = HORIZON, threshold: float = 0.0
) -> pd.DataFrame:
    """生成二分类目标: future return > threshold 则为 1，否则为 0。

    Args:
        threshold: 最小涨幅阈值（如 0.002 = 0.2%）。
            threshold > 0 时过滤微小波动，只预测显著上涨。
    """
    df = df.copy()
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["target"] = (future_return > threshold).astype(int)
    return df


def train_model(
    df: pd.DataFrame,
    symbol: str = SYMBOL,
    threshold: float = 0.0,
    feature_importance_threshold: float = 0.01,
):
    """在 TRAIN 时间段内训练 LightGBM，使用时序交叉验证。

    Args:
        df: 合并外部数据后的 OHLCV DataFrame
        symbol: 交易对，如 "BTC/USDT" 或 "ETH/USDT"
        threshold: 目标阈值（如 0.002 = 0.2%），过滤微小波动
        feature_importance_threshold: 特征重要性阈值，低于此值的特征被移除
    """
    df = build_all_features(df)
    df = prepare_target(df, threshold=threshold)

    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if c != "target"]

    # 严格时间切分: 训练集只用 TRAIN_END 之前的数据
    train_mask = df["date"] < pd.Timestamp(TRAIN_END, tz="UTC")
    df_train = df.loc[train_mask].copy()

    valid = df_train[feature_cols + ["target"]].notnull().all(axis=1)
    df_train = df_train.loc[valid].copy()

    if len(df_train) == 0:
        logger.error("No valid training samples after dropping NaNs. Check feature engineering.")
        raise ValueError("Dataset is empty after NaN drop. Inspect feature columns for all-NaN values.")

    X = df_train[feature_cols]
    y = df_train["target"]

    logger.info(f"Symbol: {symbol}")
    logger.info(f"Target threshold: {threshold:.4f} ({threshold*100:.2f}%)")
    logger.info(f"Training period: {df_train['date'].min()} ~ {df_train['date'].max()}")
    logger.info(f"Initial features: {len(feature_cols)}")
    logger.info(f"Training samples: {len(X)}, Positive ratio: {y.mean():.2%}")

    # 更保守的 LightGBM 超参，降低过拟合
    lgbm_params = {
        "n_estimators": 1000,
        "learning_rate": 0.01,
        "max_depth": 6,
        "num_leaves": 31,
        "min_child_samples": 50,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # 时序交叉验证（必须保持时间顺序）
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train)

        val_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_pred)
        auc_scores.append(auc)
        logger.info(f"Fold {fold + 1} AUC: {auc:.4f}")

    logger.info(f"Mean CV AUC: {np.mean(auc_scores):.4f} (+/- {np.std(auc_scores):.4f})")

    # 最终模型: 在整个 TRAIN 时间段上训练（用于特征重要性分析）
    final_model = LGBMClassifier(**lgbm_params)
    final_model.fit(X, y)

    # 特征重要性分析和筛选
    importance = pd.Series(
        final_model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    logger.info("Top 10 features by importance:")
    for feat, imp in importance.head(10).items():
        logger.info(f"  {feat}: {imp:.1f}")

    # 筛选重要性高于阈值的特征
    total_imp = importance.sum()
    importance_ratio = importance / total_imp
    selected_features = importance_ratio[
        importance_ratio >= feature_importance_threshold
    ].index.tolist()

    removed = set(feature_cols) - set(selected_features)
    logger.info(
        f"Feature selection: {len(feature_cols)} -> {len(selected_features)} "
        f"(removed {len(removed)} features with importance < {feature_importance_threshold*100:.1f}%)"
    )
    if removed:
        logger.info(f"Removed: {sorted(removed)}")

    # 用筛选后的特征重新训练最终模型
    X_selected = X[selected_features]
    final_model = LGBMClassifier(**lgbm_params)
    final_model.fit(X_selected, y)

    # 从训练集预测分布导出漂移监控基线
    train_pred = final_model.predict_proba(X_selected)[:, 1]
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

    # 按币种创建输出子目录
    pair_dir = MODEL_OUTPUT_DIR / symbol.replace("/", "_")
    pair_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = pair_dir / "drift_baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Drift baseline saved to {baseline_path}")

    # 在预留 TEST 集 (2024) 上评估 — 仅作 sanity check
    test_mask = df["date"] >= pd.Timestamp(TRAIN_END, tz="UTC")
    df_test = df.loc[test_mask].copy()
    test_valid = df_test[selected_features + ["target"]].notnull().all(axis=1)
    df_test = df_test.loc[test_valid]

    test_auc: Optional[float] = None
    if len(df_test) > 0:
        X_test = df_test[selected_features]
        y_test = df_test["target"]
        test_pred = final_model.predict_proba(X_test)[:, 1]
        test_auc = roc_auc_score(y_test, test_pred)
        logger.info(f"=== Held-out TEST AUC (2024): {test_auc:.4f} ===")
    else:
        logger.warning("No test data available for evaluation.")

    # 保存模型
    model_path = pair_dir / "sklearn_model.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    # 保存特征配置
    config = {
        "model_type": "lightgbm",
        "feature_columns": selected_features,
        "symbol": symbol,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "train_range": [TRAIN_START, TRAIN_END],
        "cv_auc_mean": float(np.mean(auc_scores)),
        "test_auc_2024": float(test_auc) if test_auc is not None else None,
        "threshold": threshold,
        "feature_importance_threshold": feature_importance_threshold,
    }
    config_path = pair_dir / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Feature config saved to {config_path}")

    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM classifier for a given trading pair.")
    parser.add_argument(
        "--symbol",
        type=str,
        default=SYMBOL,
        help=f"Trading pair to train on (default: {SYMBOL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.002,
        help="Target threshold for 'significant up move' (default: 0.002 = 0.2%%)",
    )
    parser.add_argument(
        "--feature-importance-threshold",
        type=float,
        default=0.01,
        help="Minimum feature importance ratio to keep (default: 0.01 = 1%%)",
    )
    args = parser.parse_args()

    logger.info(f"Starting training for {args.symbol}...")
    df = load_training_data(symbol=args.symbol)
    df = merge_external_data(df, symbol=args.symbol)
    model = train_model(
        df,
        symbol=args.symbol,
        threshold=args.threshold,
        feature_importance_threshold=args.feature_importance_threshold,
    )
    logger.info("Training complete.")
