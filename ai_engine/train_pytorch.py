"""
Train a PyTorch LSTM model for crypto direction prediction.

Output:
  ../freqtrade/user_data/models/pytorch_model.pt
  ../freqtrade/user_data/models/feature_config.json

Usage:
  cd ai_engine
  pip install -r requirements.txt
  python train_pytorch.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from features import build_all_features, get_feature_columns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_OUTPUT_DIR = Path(__file__).parent.parent / "freqtrade" / "user_data" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACK = 20        # Sequence length (number of past bars)
HORIZON = 1          # Predict next bar direction
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


def load_data():
    """Load crypto OHLCV data from Binance via ccxt."""
    from data_fetcher import fetch_ohlcv_ccxt
    df = fetch_ohlcv_ccxt(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2022-01-01",
        end_date="2024-12-31",
        exchange_name="binance",
        use_cache=True,
    )
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class CryptoLSTM(nn.Module):
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
        out = out[:, -1, :]  # Take last timestep
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


class CryptoDataset(Dataset):
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
    df = load_data()
    df = build_all_features(df)

    # Target
    df["target"] = (df["close"].shift(-HORIZON) > df["close"]).astype(float)

    feature_cols = [c for c in get_feature_columns(df) if c != "target"]
    valid = df[feature_cols + ["target"]].notnull().all(axis=1)
    df = df.loc[valid].copy()

    X = df[feature_cols].values
    y = df["target"].values

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/val split (time series aware)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = CryptoDataset(X_train, y_train, LOOKBACK)
    val_ds = CryptoDataset(X_val, y_val, LOOKBACK)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
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

        # Simple accuracy
        val_acc = np.mean((np.array(val_preds) > 0.5) == np.array(val_targets))

        logger.info(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_DIR / "pytorch_model.pt")

    logger.info(f"Best val loss: {best_val_loss:.4f}")

    # Save config
    config = {
        "model_type": "lstm",
        "feature_columns": feature_cols,
        "lookback": LOOKBACK,
        "horizon": HORIZON,
        "input_size": len(feature_cols),
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    with open(MODEL_OUTPUT_DIR / "feature_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("PyTorch model training complete.")


if __name__ == "__main__":
    train()
