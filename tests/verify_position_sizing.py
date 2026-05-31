"""
动态仓位管理简化验证脚本。

基于本地 feather 数据，模拟仓位分配效果。
不依赖交易所 API 或 Freqtrade 运行时。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 添加策略路径
sys.path.insert(0, str(Path(__file__).parent.parent / "freqtrade/user_data/strategies"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))

from features import build_all_features

# ---------------------------------------------------------------------------
# 模拟模型预测（使用未来收益率作为代理）
# ---------------------------------------------------------------------------

def simulate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """模拟 ai_prediction：基于未来 6h 收益率的简化模型。"""
    df = df.copy()
    future_return = df["close"].shift(-6) / df["close"] - 1
    # 将收益率映射到 [0, 1] 概率空间
    df["ai_prediction"] = np.clip(
        0.5 + future_return * 20,  # 放大信号
        0.3, 1.0
    )
    return df


# ---------------------------------------------------------------------------
# 仓位计算（与策略实现一致）
# ---------------------------------------------------------------------------

POSITION_SIZING_CONFIG = {
    "base_wallet_pct": 0.10,
    "min_position_pct": 0.20,
    "max_position_pct": 2.00,
    "confidence": {
        "threshold_low": 0.55,
        "threshold_high": 0.90,
        "mapping": "linear",
    },
    "volatility": {
        "target_atr_pct": 0.015,
        "max_atr_pct": 0.05,
    },
}


def compute_confidence_factor(pred: float, cfg: dict) -> float:
    conf_cfg = cfg["confidence"]
    low, high = conf_cfg["threshold_low"], conf_cfg["threshold_high"]
    factor = (pred - low) / (high - low)
    return float(np.clip(factor, 0.0, 1.0))


def compute_volatility_factor(atr_pct: float, cfg: dict) -> float:
    target = cfg["volatility"]["target_atr_pct"]
    max_atr = cfg["volatility"]["max_atr_pct"]
    if atr_pct >= max_atr:
        factor = cfg["min_position_pct"]
    else:
        factor = target / atr_pct
    return float(np.clip(factor, cfg["min_position_pct"], 1.0))


def compute_dynamic_stake(
    wallet_balance: float,
    max_trades: int,
    prediction: float,
    atr_pct: float,
    cfg: dict,
) -> float:
    base = wallet_balance * cfg["base_wallet_pct"] / max_trades
    conf = compute_confidence_factor(prediction, cfg)
    vol = compute_volatility_factor(atr_pct, cfg)
    final = base * conf * vol
    final = max(final, base * cfg["min_position_pct"])
    final = min(final, base * cfg["max_position_pct"])
    return float(final)


# ---------------------------------------------------------------------------
# 回测模拟
# ---------------------------------------------------------------------------

def run_simulation(df: pd.DataFrame, wallet: float = 10000, max_trades: int = 9) -> dict:
    """模拟固定仓位 vs 动态仓位的资金分配效果。"""
    df = simulate_predictions(df)
    df = build_all_features(df)

    if "atr_14" not in df.columns:
        raise ValueError("atr_14 not found after feature engineering")

    df["atr_pct"] = df["atr_14"] / df["close"]

    fixed_stake = 200.0
    cfg = POSITION_SIZING_CONFIG

    fixed_positions = []
    dynamic_positions = []

    for _, row in df.dropna(subset=["ai_prediction", "atr_pct"]).iterrows():
        pred = row["ai_prediction"]
        atr_pct = row["atr_pct"]

        # 固定仓位
        fixed_positions.append(fixed_stake)

        # 动态仓位
        if pred >= cfg["confidence"]["threshold_low"]:
            dynamic = compute_dynamic_stake(wallet, max_trades, pred, atr_pct, cfg)
        else:
            dynamic = 0.0  # 不入场
        dynamic_positions.append(dynamic)

    fixed_positions = np.array(fixed_positions)
    dynamic_positions = np.array(dynamic_positions)

    return {
        "total_signals": len(fixed_positions),
        "fixed": {
            "mean_stake": fixed_positions.mean(),
            "total_stake": fixed_positions.sum(),
            "std_stake": fixed_positions.std(),
        },
        "dynamic": {
            "mean_stake": dynamic_positions[dynamic_positions > 0].mean() if dynamic_positions.any() else 0,
            "total_stake": dynamic_positions.sum(),
            "std_stake": dynamic_positions[dynamic_positions > 0].std() if dynamic_positions.any() else 0,
            "min_stake": dynamic_positions[dynamic_positions > 0].min() if dynamic_positions.any() else 0,
            "max_stake": dynamic_positions.max(),
            "signal_count": (dynamic_positions > 0).sum(),
        },
        "position_series": {
            "fixed": fixed_positions,
            "dynamic": dynamic_positions,
        },
    }


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / "freqtrade/user_data/data/binance/BTC_USDT-4h.feather"
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    df = pd.read_feather(data_path)
    print(f"Loaded {len(df)} candles from {df['date'].min()} to {df['date'].max()}")
    print()

    results = run_simulation(df)

    print("=" * 60)
    print("固定仓位 (stake_amount = 200)")
    print("=" * 60)
    print(f"  信号次数:     {results['total_signals']}")
    print(f"  平均仓位:     {results['fixed']['mean_stake']:.2f} USDT")
    print(f"  总投入:       {results['fixed']['total_stake']:.2f} USDT")
    print(f"  仓位标准差:   {results['fixed']['std_stake']:.2f}")
    print()

    print("=" * 60)
    print("动态仓位 (confidence × volatility)")
    print("=" * 60)
    print(f"  入场信号次数: {results['dynamic']['signal_count']}")
    print(f"  平均仓位:     {results['dynamic']['mean_stake']:.2f} USDT")
    print(f"  总投入:       {results['dynamic']['total_stake']:.2f} USDT")
    print(f"  最小仓位:     {results['dynamic']['min_stake']:.2f} USDT")
    print(f"  最大仓位:     {results['dynamic']['max_stake']:.2f} USDT")
    print(f"  仓位标准差:   {results['dynamic']['std_stake']:.2f}")
    print()

    print("=" * 60)
    print("对比分析")
    print("=" * 60)
    fixed_total = results['fixed']['total_stake']
    dynamic_total = results['dynamic']['total_stake']
    if fixed_total > 0:
        ratio = dynamic_total / fixed_total
        print(f"  动态/固定总投入比: {ratio:.2f}x")
        if ratio < 0.8:
            print("  → 动态仓位更保守（过滤了弱信号/高波动）")
        elif ratio > 1.2:
            print("  → 动态仓位更激进（强信号/低波动时加仓）")
        else:
            print("  → 总投入接近，但分配更智能")
    print()
