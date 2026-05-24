"""
资金费率套利 (Cash-and-Carry Arbitrage) — 完整回测 + 实盘脚本

策略: 做多现货 + 做空等量永续合约，收取资金费率。
      Delta 中性，不依赖价格方向预测。

用法:
  python funding_arbitrage.py                     # 回测 9 个品种
  python funding_arbitrage.py --live BTC/USDT      # 实盘监控（单品种）
  python funding_arbitrage.py --stake 5000         # 自定义本金
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent
sys.path.insert(0, str(_project_root))

from data.service import query
import data.service_defaults  # noqa
from research.training_config import TRAIN_START, FULL_END, EXCHANGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
    "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT", "AVAX/USDT",
]

# 成本参数
SPOT_FEE = 0.001      # 0.1% 现货手续费
FUTURES_FEE = 0.0004  # 0.04% 合约手续费
MARGIN_RATIO = 0.1    # 10% 保证金（10x 杠杆用 1/10）
MIN_FUNDING_ANNUAL = 5.0  # 最低年化费率阈值 (%)


def load_funding_data(symbol: str) -> pd.DataFrame:
    """加载历史资金费率数据。"""
    fr = query("funding_rate", symbol, since=TRAIN_START, until=FULL_END,
               exchange_name=EXCHANGE, use_cache=True)
    if fr.empty:
        return fr
    fr["date"] = pd.to_datetime(fr["date"])
    fr = fr.sort_values("date")
    return fr


def backtest_single(
    symbol: str,
    stake: float = 10000,
    min_annual_pct: float = MIN_FUNDING_ANNUAL,
    compound: bool = True,
) -> dict:
    """单品种资金费率套利回测。

    Args:
        symbol: 交易对
        stake: 本金 (USDT)
        min_annual_pct: 最低年化费率阈值
        compound: 是否复利（收益再投资）

    Returns:
        dict with backtest metrics
    """
    fr = load_funding_data(symbol)
    if fr.empty or "fundingRate" not in fr.columns:
        return {}

    rate = fr["fundingRate"].dropna().values
    dates = fr["date"].values

    if len(rate) < 100:
        return {}

    # 年化费率（百分比）= rate × 3 × 365 × 100
    annual_pct = rate * 3 * 365 * 100

    # 仅在年化费率 > 阈值时开仓
    active = annual_pct > min_annual_pct

    # 简化: 假设始终持仓，但只在费率 > 阈值时计入收益
    # 实际更精确的做法是只在活跃时开仓

    equity = stake
    equity_curve = [equity]
    total_fees_collected = 0
    active_periods = 0

    for i, (r, is_active) in enumerate(zip(rate, active)):
        if is_active:
            # 每期利润 = 资金费率 × 本金
            profit = r * equity
            total_fees_collected += profit
            equity += profit
            active_periods += 1
        equity_curve.append(equity)

    equity_curve = np.array(equity_curve)

    # 扣除开仓手续费（一次性）
    entry_fee = stake * (SPOT_FEE + FUTURES_FEE)
    equity -= entry_fee

    total_return_pct = (equity / stake - 1) * 100
    years = len(rate) / (3 * 365)  # 8h 周期 → 年
    cagr = ((equity / stake) ** (1 / max(years, 0.1)) - 1) * 100

    # 最大回撤
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / peak * 100
    max_dd = float(dd.max())

    # 夏普（基于每期收益）
    period_returns = []
    for i, (r, is_active) in enumerate(zip(rate, active)):
        period_returns.append(r * 100 if is_active else 0.0)
    period_returns = np.array(period_returns)
    sharpe = float(np.mean(period_returns) / (np.std(period_returns) + 1e-10) * np.sqrt(3 * 365))

    return {
        "symbol": symbol,
        "stake": stake,
        "final_equity": round(equity, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe": round(sharpe, 2),
        "total_periods": len(rate),
        "active_periods": active_periods,
        "active_pct": round(active_periods / len(rate) * 100, 1),
        "total_fees_usdt": round(total_fees_collected, 2),
        "date_start": str(dates[0]),
        "date_end": str(dates[-1]),
    }


def backtest_portfolio(
    symbols: list[str],
    total_stake: float = 50000,
    min_annual_pct: float = MIN_FUNDING_ANNUAL,
) -> pd.DataFrame:
    """多品种组合回测 — 等权分配本金。"""
    per_symbol = total_stake / len(symbols)
    results = []

    for sym in symbols:
        r = backtest_single(sym, stake=per_symbol, min_annual_pct=min_annual_pct)
        if r:
            results.append(r)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    df = df.sort_values("cagr_pct", ascending=False)
    return df


def print_backtest_report(df: pd.DataFrame, total_stake: float):
    """打印回测报告。"""
    print(f"\n{'='*90}")
    print(f"资金费率套利回测 | 本金: ${total_stake:,.0f} | 阈值: >{MIN_FUNDING_ANNUAL}% 年化")
    print(f"费率: 现货 {SPOT_FEE*100}% + 合约 {FUTURES_FEE*100}% | 保证金: {MARGIN_RATIO*100}%")
    print(f"{'='*90}")

    print(f"\n{'Symbol':<14} {'年化%':>7} {'总收益%':>8} {'终值$':>10} {'最大回撤%':>9} {'夏普':>7} {'活跃%':>7}")
    print("-" * 75)
    for _, r in df.iterrows():
        print(
            f"{r['symbol']:<14} {r['cagr_pct']:>6.1f}% {r['total_return_pct']:>7.1f}% "
            f"${r['final_equity']:>9,.0f} {r['max_drawdown_pct']:>8.1f}% "
            f"{r['sharpe']:>6.1f} {r['active_pct']:>6.1f}%"
        )

    total_final = df["final_equity"].sum()
    total_return = (total_final / total_stake - 1) * 100
    print("-" * 75)
    print(f"{'PORTFOLIO':<14} {'':>7} {total_return:>7.1f}% ${total_final:>9,.0f}")

    # 风险提示
    print(f"\n⚠️  风险提示:")
    print(f"  1. 假设始终能以资金费率 > 阈值时开仓、< 阈值时平仓")
    print(f"  2. 未计入现货-合约基差波动（极端行情可能扩大）")
    print(f"  3. 需要交易所同时支持现货和合约账户")
    print(f"  4. 极端行情下合约空头可能被强平（需充足保证金）")


def live_monitor(symbol: str):
    """实盘资金费率监控。"""
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: pip install ccxt")
        return

    exchange = ccxt.binance({"options": {"defaultType": "future"}})
    markets = exchange.load_markets()

    pair = symbol.replace("/", ":")
    if pair not in markets:
        pair = f"{symbol.split('/')[0]}/{symbol.split('/')[1]}:{symbol.split('/')[1]}"
    if pair not in markets:
        logger.error(f"Pair {pair} not found on Binance futures")
        return

    ticker = exchange.fetch_ticker(pair)
    funding = exchange.fetch_funding_rate(pair)

    rate_8h = funding.get("fundingRate", 0)
    rate_annual = rate_8h * 3 * 365 * 100

    print(f"\n{'='*60}")
    print(f"实盘资金费率监控 — {symbol}")
    print(f"时间: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"价格: ${ticker['last']:,.2f}")
    print(f"8h 费率: {rate_8h*100:.4f}%")
    print(f"年化费率: {rate_annual:.2f}%")
    print(f"状态: {'✅ 可开仓' if rate_annual > MIN_FUNDING_ANNUAL else '❌ 费率不足'}")
    print(f"{'='*60}")

    if rate_annual > MIN_FUNDING_ANNUAL:
        print(f"\n建议操作:")
        print(f"  1. 现货买入 ${10000:,.0f} {symbol}")
        print(f"  2. 合约做空 ${10000:,.0f} {symbol} 永续")
        print(f"  3. 每 8h 收取 ~${10000 * rate_8h:.2f}")
        print(f"  4. 预计月收益: ~${10000 * rate_8h * 3 * 30:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Funding Rate Arbitrage")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS,
                        help="Trading pairs")
    parser.add_argument("--stake", type=float, default=50000,
                        help="Total capital (USDT)")
    parser.add_argument("--live", type=str, default=None,
                        help="Live monitor mode (symbol, e.g. BTC/USDT)")
    parser.add_argument("--min-funding", type=float, default=MIN_FUNDING_ANNUAL,
                        help="Minimum annualized funding rate %")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    if args.live:
        live_monitor(args.live)
    else:
        df = backtest_portfolio(args.symbols, total_stake=args.stake,
                                min_annual_pct=args.min_funding)
        if args.json:
            print(df.to_json(orient="records", indent=2))
        else:
            print_backtest_report(df, args.stake)
