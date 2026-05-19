"""Drift detection utilities for Freqtrade strategy. Pure numpy/pandas only."""
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_psi(baseline_hist_counts: list, current_values: list, bins: int = 20, range_: tuple = (0, 1)) -> float:
    """
    计算 Population Stability Index (PSI)。
    使用训练时保存的基准直方图 bins，与当前滑动窗口的分布比较。
    """
    baseline = np.array(baseline_hist_counts, dtype=float)
    baseline_pct = baseline / baseline.sum()
    
    current_hist, _ = np.histogram(current_values, bins=bins, range=range_)
    current = np.array(current_hist, dtype=float)
    current_sum = current.sum()
    if current_sum == 0:
        return float("inf")
    current_pct = current / current_sum
    
    # 避免除零和对数为零
    baseline_pct = np.clip(baseline_pct, 1e-10, 1.0)
    current_pct = np.clip(current_pct, 1e-10, 1.0)
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def send_telegram_alert(message: str, config_path: str = "/freqtrade/config.json") -> bool:
    """
    直接通过 Telegram Bot API 发送消息。
    从 Freqtrade 的 config.json 中读取 token 和 chat_id。
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
    """Append drift alert to JSONL file."""
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
