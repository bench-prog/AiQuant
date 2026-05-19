"""
Feature Engineering Module for AiQuant

Re-exports from the shared feature_engineering module located in
freqtrade/user_data/strategies/ so that training and backtesting use
identical feature logic.
"""

import sys
from pathlib import Path

# Add strategies dir to path so we can import the shared module
_strategies_dir = Path(__file__).parent.parent / "freqtrade" / "user_data" / "strategies"
sys.path.insert(0, str(_strategies_dir))

from feature_engineering import (  # noqa: E402
    add_trend_features,
    add_momentum_features,
    add_volatility_features,
    add_volume_features,
    add_price_structure,
    add_lag_features,
    add_time_features,
    add_crypto_features,
    add_funding_rate_features,
    add_open_interest_features,
    build_all_features,
    get_feature_columns,
)
