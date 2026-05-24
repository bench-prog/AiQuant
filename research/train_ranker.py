"""
多品种截面排序模型训练。

训练 LightGBM 回归模型，预测每个品种的未来收益率（连续值）。
策略侧通过比较各品种预测收益的排名，做多 Top N。

用法:
  cd research
  python train_ranker.py                          # 训练所有 9 个品种
  python train_ranker.py --pairs BTC/USDT ETH/USDT  # 仅训练指定品种
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
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
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

_strategies_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "strategies"
sys.path.insert(0, str(_strategies_dir))
from features import build_all_features, get_feature_columns  # noqa: E402
from data.onchain import fetch_btc_onchain, merge_onchain_into_ohlcv  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_PAIRS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT",
]


def load_and_prepare(symbol: str) -> pd.DataFrame:
    """加载单个品种数据，计算特征和目标（连续收益率）。"""
    df = load_training_data(symbol=symbol)
    df = merge_external_data(df, symbol=symbol, timeframe=TIMEFRAME)

    # BTC 链上数据（日频，通过前向填充对齐到 1h）
    if symbol == "BTC/USDT":
        try:
            onchain = fetch_btc_onchain(use_cache=True)
            df = merge_onchain_into_ohlcv(df, onchain)
            logger.info(f"Merged BTC on-chain data: {len(onchain)} daily rows.")
        except Exception as e:
            logger.warning(f"Failed to merge BTC on-chain data: {e}")

    df = build_all_features(df)
    # 目标: 未来 HORIZON 期的收益率（连续值）
    df["target"] = df["close"].shift(-HORIZON) / df["close"] - 1
    df["symbol"] = symbol
    return df


def train_ranker(
    symbols: list[str],
    feature_importance_threshold: float = 0.005,
    params_override: Optional[dict] = None,
):
    """训练 LightGBM 回归模型，预测跨品种未来收益率排序。

    Args:
        symbols: 品种列表
        feature_importance_threshold: 特征重要性筛选阈值
        params_override: 自定义超参
    """
    # 加载并合并所有品种数据
    dfs = []
    for sym in symbols:
        logger.info(f"Loading {sym}...")
        df = load_and_prepare(sym)
        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if c not in ("target", "symbol")]
        # 只保留训练期
        df = df[df["date"] < pd.Timestamp(TRAIN_END, tz="UTC")].copy()
        dfs.append(df[feature_cols + ["target", "symbol", "date"]])

    df_all = pd.concat(dfs, ignore_index=True)
    feature_cols = [c for c in df_all.columns if c not in ("target", "symbol", "date")]

    # 仅要求 target 非空（特征中的 NaN 由 LightGBM 原生处理）
    valid = df_all["target"].notna()
    df_all = df_all.loc[valid].copy()

    X = df_all[feature_cols]
    y = df_all["target"]

    logger.info(f"Total samples: {len(X)}, Pairs: {len(symbols)}, Features: {len(feature_cols)}")
    logger.info(f"Target stats: mean={y.mean():.4%}, std={y.std():.4%}")

    # 模型参数
    reg_params = {
        "n_estimators": 500,
        "learning_rate": 0.02,
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
    if params_override:
        reg_params.update({k: v for k, v in params_override.items() if k in reg_params})

    # 时序交叉验证 + 截面排序评估
    tscv = TimeSeriesSplit(n_splits=5)
    ic_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMRegressor(**reg_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        # 截面信息系数 (IC): Spearman rank correlation
        ic, _ = spearmanr(y_val, preds)
        ic_scores.append(ic)
        logger.info(f"Fold {fold + 1}: RMSE={rmse:.4%}, IC={ic:.4f}")

    logger.info(f"Mean IC: {np.mean(ic_scores):.4f} (+/- {np.std(ic_scores):.4f})")

    # 最终模型 + 特征筛选
    final_model = LGBMRegressor(**reg_params)
    final_model.fit(X, y)

    importance = pd.Series(
        final_model.feature_importances_,
        index=feature_cols,
        name="importance",
    ).sort_values(ascending=False)

    logger.info("Top 10 features:")
    for feat, imp in importance.head(10).items():
        logger.info(f"  {feat}: {imp:.1f}")

    total_imp = importance.sum()
    importance_ratio = importance / total_imp
    selected_features = importance_ratio[
        importance_ratio >= feature_importance_threshold
    ].index.tolist()

    logger.info(f"Feature selection: {len(feature_cols)} -> {len(selected_features)}")

    X_sel = X[selected_features]
    final_model = LGBMRegressor(**reg_params)
    final_model.fit(X_sel, y)

    # 在 2024 测试集上评估截面 IC
    test_dfs = []
    for sym in symbols:
        df = load_and_prepare(sym)
        df = df[df["date"] >= pd.Timestamp(TRAIN_END, tz="UTC")].copy()
        # 确保所有选定特征列存在（非 BTC 品种缺少链上列，填 NaN）
        for col in selected_features:
            if col not in df.columns:
                df[col] = np.nan
        test_dfs.append(df[selected_features + ["target", "symbol", "date"]])

    df_test = pd.concat(test_dfs, ignore_index=True)
    valid_test = df_test[selected_features + ["target"]].notnull().all(axis=1)
    df_test = df_test.loc[valid_test]
    test_pred = final_model.predict(df_test[selected_features])
    test_ic, _ = spearmanr(df_test["target"], test_pred)
    logger.info(f"=== Test IC (2024): {test_ic:.4f} ===")

    # 保存
    pair_dir = MODEL_OUTPUT_DIR / "ranker"
    pair_dir.mkdir(parents=True, exist_ok=True)

    model_path = pair_dir / "ranker_model.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Model saved to {model_path}")

    config = {
        "model_type": "lightgbm_ranker",
        "feature_columns": selected_features,
        "pairs": symbols,
        "timeframe": TIMEFRAME,
        "horizon": HORIZON,
        "train_range": [TRAIN_START, TRAIN_END],
        "cv_ic_mean": float(np.mean(ic_scores)),
        "test_ic_2024": float(test_ic),
    }
    config_path = pair_dir / "feature_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config saved to {config_path}")

    return final_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train cross-sectional ranking model.")
    parser.add_argument("--pairs", type=str, nargs="+", default=DEFAULT_PAIRS,
                        help="Trading pairs to train on")
    parser.add_argument("--params", type=str, default=None,
                        help="Path to best_params.json")
    args = parser.parse_args()

    params_override = None
    if args.params:
        with open(args.params) as f:
            params_override = json.load(f)
        logger.info(f"Loaded params from {args.params}")

    train_ranker(args.pairs, params_override=params_override)
    logger.info("Training complete.")
