"""
AiQuant 训练公共配置。

所有训练脚本共享的常量和路径配置。
用法:
    from research.training_config import SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# 数据参数
# ---------------------------------------------------------------------------
SYMBOL: str = "BTC/USDT"
TIMEFRAME: str = "1h"
TRAIN_START: str = "2022-01-01"
TRAIN_END: str = "2023-12-31"
FULL_END: str = "2024-12-31"
EXCHANGE: str = "binance"

# ---------------------------------------------------------------------------
# 模型参数
# ---------------------------------------------------------------------------
HORIZON: int = 1

# LSTM 特有
LOOKBACK: int = 20
BATCH_SIZE: int = 64
EPOCHS: int = 30
LR: float = 1e-3
HIDDEN_SIZE: int = 64
NUM_LAYERS: int = 2
DROPOUT: float = 0.2

# ---------------------------------------------------------------------------
# 输出路径
# ---------------------------------------------------------------------------
MODEL_OUTPUT_DIR: Path = Path(__file__).parent.parent / "freqtrade" / "user_data" / "models"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
