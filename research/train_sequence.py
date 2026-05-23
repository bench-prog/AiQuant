"""
训练 PyTorch LSTM 方向预测模型。

采用严格的时间切分避免前瞻偏差:
  训练集: 2022-01-01 ~ 2023-12-31
  测试集 (仅用于 Freqtrade 回测评估): 2024-01-01 ~ 2024-12-31

输出:
  ../freqtrade/user_data/models/pytorch_model.pt
  ../freqtrade/user_data/models/feature_config_lstm.json

用法:
  cd research
  pip install -r requirements.txt
  python train_sequence.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from research.training_config import (
    TRAIN_START,
    TRAIN_END,
    FULL_END,
    LOOKBACK,
    HORIZON,
    BATCH_SIZE,
    EPOCHS,
    LR,
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    MODEL_OUTPUT_DIR,
)
from research.data_utils import load_training_data, merge_external_data

# Import shared feature engineering from strategies directory
_strategies_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "strategies"
sys.path.insert(0, str(_strategies_dir))
from features import build_all_features, get_feature_columns  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoLSTM(nn.Module):
    """LSTM 序列模型。架构需与策略中 SimpleLSTM 保持一致。"""

    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


class CryptoDataset(Dataset):
    """生成 lookback 长度的序列样本。"""

    def __init__(self, X: np.ndarray, y: np.ndarray, lookback: int):
        self.X = X
        self.y = y
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.lookback]
        y_val = self.y[idx + self.lookback - 1]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)


def train():
    df = load_training_data()
    df = merge_external_data(df)
    df = build_all_features(df)
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(float)

    feature_cols = [c for c in get_feature_columns(df) if c != "target"]
    valid = df[feature_cols + ["target"]].notnull().all(axis=1)
    df = df.loc[valid].copy()

    # 严格时间切分
    train_mask = df["date"] < pd.Timestamp(TRAIN_END, tz="UTC")
    df_train = df.loc[train_mask].copy()
    df_test = df.loc[~train_mask].copy()

    if len(df_train) == 0:
        raise ValueError("No training data after temporal split.")

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values
    X_test = df_test[feature_cols].values if len(df_test) > 0 else None
    y_test = df_test["target"].values if len(df_test) > 0 else None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test = scaler.transform(X_test)

    # 训练/验证切分（时序感知，仅在 TRAIN 时间段内）
    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    train_ds = CryptoDataset(X_tr, y_tr, LOOKBACK)
    val_ds = CryptoDataset(X_val, y_val, LOOKBACK)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CryptoLSTM(input_size=len(feature_cols)).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                loss = criterion(pred, yb)
                val_losses.append(loss.item())
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(yb.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)
        val_acc = np.mean((np.array(val_preds) > 0.5) == np.array(val_targets))

        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "pytorch_model.pt")

    logger.info(f"Best val loss: {best_val_loss:.4f}")

    # 从训练集预测分布导出漂移监控基线
    train_ds_full = CryptoDataset(X_train, y_train, LOOKBACK)
    train_loader_full = DataLoader(train_ds_full, batch_size=BATCH_SIZE, shuffle=False)
    train_preds = []
    model.eval()
    with torch.no_grad():
        for xb, _ in train_loader_full:
            xb = xb.to(device)
            pred = model(xb).squeeze()
            train_preds.extend(pred.cpu().numpy())
    train_pred = np.array(train_preds)
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
    baseline_path = MODEL_OUTPUT_DIR / "drift_baseline_lstm.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"Drift baseline saved to {baseline_path}")

    # 在预留 TEST 集上评估
    if X_test is not None:
        test_ds = CryptoDataset(X_test, y_test, LOOKBACK)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_preds = []
        test_targets = []
        model.eval()
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).squeeze()
                test_preds.extend(pred.cpu().numpy())
                test_targets.extend(yb.cpu().numpy())
        test_acc = np.mean((np.array(test_preds) > 0.5) == np.array(test_targets))
        logger.info(f"=== Held-out TEST Acc (2024): {test_acc:.4f} ===")

    config = {
        "model_type": "lstm",
        "feature_columns": feature_cols,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "input_size": len(feature_cols),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "train_range": [TRAIN_START, TRAIN_END],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(MODEL_OUTPUT_DIR / "feature_config_lstm.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("PyTorch model training complete.")


if __name__ == "__main__":
    train()
