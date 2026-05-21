"""
AiQuant AI 模型策略 (AIModelStrategy)

集成自定义 scikit-learn 或 PyTorch 模型到 Freqtrade。
模型在 bot_start() 中一次性加载，基于技术指标特征生成入场/出场信号。

模型文件路径:
  /freqtrade/user_data/models/sklearn_model.pkl
  /freqtrade/user_data/models/pytorch_model.pt
  /freqtrade/user_data/models/feature_config.json

若未找到模型，策略回退到简单的 RSI 信号（仅供演示）。
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from freqtrade.strategy import IStrategy

# Ensure data/ package is importable both locally and in Docker
import sys
from pathlib import Path

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

from data.service import query, merge_into
import data.service_defaults  # registers built-in data sources

# Shared feature engineering (same code used in training)
from features import build_all_features

# Drift detection utilities (pure numpy/pandas, no external deps)
# Drift detection utilities (inline to keep strategy self-contained)

def compute_stability_index(baseline_hist_counts: list, current_values: list, bins: int = 20, range_: tuple = (0, 1)) -> float:
    """
    计算群体稳定性指数 (PSI) 用于检测模型漂移。

    将训练基线的直方图与当前预测分布（滑动窗口）做对比。
    PSI > 0.25 通常认为分布发生显著偏移。
    """
    baseline = np.array(baseline_hist_counts, dtype=float)
    baseline_pct = baseline / baseline.sum()
    
    current_hist, _ = np.histogram(current_values, bins=bins, range=range_)
    current = np.array(current_hist, dtype=float)
    current_sum = current.sum()
    if current_sum == 0:
        return float("inf")
    current_pct = current / current_sum
    
    # Avoid division by zero and log(0)
    baseline_pct = np.clip(baseline_pct, 1e-10, 1.0)
    current_pct = np.clip(current_pct, 1e-10, 1.0)
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def send_telegram_alert(message: str, config_path: str = "/freqtrade/config_ai_model.json") -> bool:
    """
    通过 Telegram Bot API 直接发送消息。
    从 Freqtrade 配置中读取 token 和 chat_id。
    """
    try:
        import urllib.request
        import urllib.parse
    except ImportError:
        logger.warning("urllib not available, cannot send Telegram alert.")
        return False
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        telegram_cfg = config.get("telegram", {})
        if not telegram_cfg.get("enabled"):
            return False
        token = telegram_cfg.get("token")
        chat_id = telegram_cfg.get("chat_id")
        if not token or not chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Failed to send Telegram alert: {e}")
        return False


def append_drift_alert(message: str, metrics: dict, log_dir: str = "/freqtrade/user_data/logs"):
    """将漂移告警追加写入 JSONL 文件。"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    alert_file = log_path / "drift_alerts.jsonl"
    record = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "message": message,
        **metrics,
    }
    with open(alert_file, "a") as f:
        f.write(json.dumps(record) + "\n")



# PyTorch 可选导入，未安装时优雅降级
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    class SimpleLSTM(nn.Module):
        """LSTM 序列模型，架构与 research/train_sequence.py 的 CryptoLSTM 保持一致。"""

        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(32, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            out = self.dropout(out)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
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
    """加载外部 ML 模型生成交易信号的 AI 驱动策略。"""

    # --- 策略元数据 ---
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

    # --- 模型状态 ---
    sklearn_model: Optional[object] = None
    pytorch_model: Optional[nn.Module] = None
    feature_config: Optional[dict] = None
    model_type: Optional[str] = None  # 'sklearn', 'pytorch', 或 'fallback'
    drift_baseline: Optional[dict] = None
    prediction_buffer: list = []
    candle_count: int = 0

    # 从 feature_config 加载的 StandardScaler 参数（PyTorch 序列模型用）
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None

    # --- 漂移监控配置 ---
    DRIFT_WINDOW_SIZE: int = 500
    DRIFT_CHECK_INTERVAL: int = 100
    DRIFT_PSI_THRESHOLD: float = 0.25

    # ------------------------------------------------------------------
    # Bot 生命周期钩子
    # ------------------------------------------------------------------
    def bot_start(self, **kwargs) -> None:
        """Bot 启动时一次性加载 AI 模型。"""
        logger.info("[AiQuant] Loading AI model...")

        # 1. 尝试加载 sklearn 模型
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
                # Load model architecture from feature_config (must match training)
                cfg = self.feature_config or {}
                input_size = cfg.get("input_size", cfg.get("feature_columns", []).__len__())
                hidden_size = cfg.get("hidden_size", 64)
                num_layers = cfg.get("num_layers", 2)
                dropout = cfg.get("dropout", 0.2)
                self.pytorch_model = SimpleLSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                self.pytorch_model.load_state_dict(torch.load(PYTORCH_MODEL_PATH, map_location="cpu"))
                self.pytorch_model.eval()
                self.model_type = "pytorch"
                logger.info(
                    f"[AiQuant] Loaded PyTorch model from {PYTORCH_MODEL_PATH} "
                    f"(input={input_size}, hidden={hidden_size}, layers={num_layers})"
                )
            except Exception as e:
                logger.warning(f"[AiQuant] Failed to load PyTorch model: {e}")

        else:
            self.model_type = "fallback"
            logger.warning("[AiQuant] No AI model found. Using fallback RSI strategy.")

        # 3. Load feature config (optional) — must happen before model loading for PyTorch arch
        if FEATURE_CONFIG_PATH.exists():
            with open(FEATURE_CONFIG_PATH, "r") as f:
                self.feature_config = json.load(f)
            logger.info(f"[AiQuant] Loaded feature config: {self.feature_config}")

            # Load scaler parameters if present (from train_sequence.py)
            if "scaler_mean" in self.feature_config and "scaler_scale" in self.feature_config:
                self.scaler_mean = np.array(self.feature_config["scaler_mean"], dtype=np.float32)
                self.scaler_scale = np.array(self.feature_config["scaler_scale"], dtype=np.float32)
                logger.info("[AiQuant] Loaded StandardScaler params from feature config.")

        # 4. Load drift baseline
        self._load_drift_baseline()
        self.prediction_buffer = []
        self.candle_count = 0

    # ------------------------------------------------------------------
    # 特征工程
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        构建全部特征并运行模型推理。
        使用与训练相同的 features 模块，确保特征一致性。
        """
        pair = metadata.get("pair", "")
        exchange_name = self.config.get("exchange", {}).get("name", "binance")

        # Merge external data (funding rate / open interest) via unified data service
        if pair and "date" in dataframe.columns:
            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                fr_df = query("funding_rate", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, fr_df, "fundingRate")
            except Exception as e:
                logger.warning(f"[AiQuant] Failed to merge funding rate: {e}")

            try:
                since = dataframe["date"].min().strftime("%Y-%m-%d")
                until = dataframe["date"].max().strftime("%Y-%m-%d")
                oi_df = query("open_interest", pair, since=since, until=until,
                              exchange_name=exchange_name, use_cache=True)
                dataframe = merge_into(dataframe, oi_df, "openInterest")
            except Exception as e:
                logger.warning(f"[AiQuant] Failed to merge open interest: {e}")

        # Build all features (identical to training pipeline)
        dataframe = build_all_features(dataframe)

        # --- Model inference ------------------------------------------------
        if self.model_type == "sklearn" and self.sklearn_model is not None:
            dataframe = self._predict_classifier(dataframe)
        elif self.model_type == "pytorch" and self.pytorch_model is not None:
            dataframe = self._predict_sequence_model(dataframe)
        else:
            # Fallback: neutral prediction
            dataframe["ai_prediction"] = 0.5

        # --- Drift monitoring ---
        if self.model_type != "fallback" and self.drift_baseline is not None:
            dataframe = self._update_drift_monitor(dataframe, metadata)

        return dataframe

    def _prepare_features(self, dataframe: pd.DataFrame) -> tuple[list[str], pd.Series, pd.DataFrame]:
        """解析特征列，缺失列填 0，返回有效行索引。

        Returns:
            (feature_cols, valid_idx, df_valid)
        """
        feature_cols = self.feature_config.get("feature_columns", []) if self.feature_config else []
        if not feature_cols:
            logger.warning("[AiQuant] No feature_columns in config. Using all non-OHLCV columns.")
            feature_cols = [c for c in dataframe.columns if c not in {"open", "high", "low", "close", "volume", "date"}]

        for col in feature_cols:
            if col not in dataframe.columns:
                dataframe[col] = 0.0

        valid_idx = dataframe[feature_cols].notnull().all(axis=1)
        return feature_cols, valid_idx, dataframe

    def _predict_classifier(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """运行 sklearn 模型推理。"""
        feature_cols, valid_idx, dataframe = self._prepare_features(dataframe)
        X = dataframe.loc[valid_idx, feature_cols].values

        if len(X) == 0:
            dataframe["ai_prediction"] = 0.5
            return dataframe

        if hasattr(self.sklearn_model, "predict_proba"):
            probs = self.sklearn_model.predict_proba(X)[:, 1]
        else:
            probs = self.sklearn_model.predict(X)

        dataframe.loc[valid_idx, "ai_prediction"] = probs
        dataframe["ai_prediction"] = dataframe["ai_prediction"].fillna(0.5)
        return dataframe

    def _predict_sequence_model(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """运行 PyTorch 序列模型推理。"""
        feature_cols, valid_idx, dataframe = self._prepare_features(dataframe)
        df_valid = dataframe.loc[valid_idx].copy()

        if len(df_valid) == 0:
            dataframe["ai_prediction"] = 0.5
            return dataframe

        lookback = self.feature_config.get("lookback", 20) if self.feature_config else 20

        X = df_valid[feature_cols].values.astype(np.float32)
        if self.scaler_mean is not None and self.scaler_scale is not None:
            X = (X - self.scaler_mean) / self.scaler_scale

        predictions = []
        for i in range(len(df_valid)):
            if i < lookback:
                predictions.append(0.5)
                continue
            seq = X[i - lookback:i]
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)
            with torch.no_grad():
                pred = self.pytorch_model(seq_tensor).item()
            predictions.append(pred)

        dataframe.loc[valid_idx, "ai_prediction"] = predictions
        dataframe["ai_prediction"] = dataframe["ai_prediction"].fillna(0.5)
        return dataframe

    # ------------------------------------------------------------------
    # 漂移监控
    # ------------------------------------------------------------------
    def _load_drift_baseline(self) -> None:
        """加载训练时导出的漂移基线。"""
        baseline_path = MODEL_DIR / "drift_baseline.json"
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                self.drift_baseline = json.load(f)
            logger.info(f"[AiQuant] Loaded drift baseline from {baseline_path}")
        else:
            self.drift_baseline = None
            logger.warning("[AiQuant] No drift_baseline.json found. Drift monitoring disabled.")

    def _update_drift_monitor(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """更新预测缓冲区，并周期性检测模型漂移。"""
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

        psi = compute_stability_index(
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
    # 信号生成
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
