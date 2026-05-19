# AiQuant - AI Crypto Quant Trading Project

## Project Overview

AiQuant is a **personal AI-powered cryptocurrency quantitative trading system** built on top of [Freqtrade](https://www.freqtrade.io/) (the most popular open-source crypto trading bot). The goal is to minimize infrastructure building effort while allowing flexible integration of custom AI models (scikit-learn, PyTorch, etc.) for signal generation.

**Target Market:** Cryptocurrency (Binance, OKX, Bybit via ccxt)
**Trading Modes:** Spot and Perpetual Futures
**Primary Framework:** Freqtrade (Docker-based)
**AI Integration:** Custom models saved as `.pkl` (sklearn) or `.pt` (PyTorch) and loaded inside Freqtrade strategies

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Trading Engine | Freqtrade (Docker) |
| Exchange API | ccxt (via Freqtrade) |
| AI/ML | scikit-learn, PyTorch, pandas, numpy |
| Data Processing | pandas, polars, pandas-ta (technical indicators) |
| Feature Storage | Parquet/CSV in `ai_engine/cache/` |
| Monitoring | Freqtrade Web UI (port 8080), Telegram Bot |
| Environment | Docker + Docker Compose |

---

## Directory Structure

```
AiQuant/
├── CLAUDE.md                           # This file
├── README.md                           # Human-readable quick start
├── docker/
│   └── docker-compose.yml              # Freqtrade Docker service definition
├── scripts/
│   └── setup.sh                        # One-click setup script
├── freqtrade/
│   ├── config.json                     # Main Freqtrade configuration (API keys, pairs, risk)
│   └── user_data/
│       ├── strategies/
│       │   ├── __init__.py
│       │   └── AIModelStrategy.py      # Example strategy with AI model loading
│       ├── models/                     # Trained AI models (.pkl, .pt)
│       ├── data/                       # Historical price data downloaded by Freqtrade
│       ├── notebooks/                  # Jupyter notebooks for research
│       └── logs/                       # Strategy logs
├── ai_engine/                          # Local AI model training scripts (outside Docker)
│   ├── requirements.txt
│   ├── features.py                     # Feature engineering utilities
│   ├── train_sklearn.py                # Train a LightGBM/RF model
│   └── train_pytorch.py              # Train an LSTM/Transformer model
└── .gitignore
```

---

## Quick Start

### 1. One-Click Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Check Docker/Docker Compose installation
- Create the directory structure
- Pull the `freqtradeorg/freqtrade:stable` image

### 2. Train AI Model (No API Key Needed)

The training scripts use **ccxt** to download historical crypto data directly from Binance (or OKX/Bybit). No exchange API key is required for public OHLCV data.

```bash
cd ai_engine
pip install -r requirements.txt
python train_sklearn.py        # Downloads BTC/USDT 1h data, trains LightGBM
python train_pytorch.py        # Or train an LSTM
```

Data is automatically cached to `ai_engine/cache/` as Parquet files to avoid re-downloading.

### 3. Configure Exchange API Keys (for Trading Only)

Edit `freqtrade/config.json`:

```json
"exchange": {
    "name": "binance",
    "key": "YOUR_API_KEY",
    "secret": "YOUR_SECRET",
    "ccxt_config": {
        "options": {
            "defaultType": "future"
        }
    }
}
```

> **Security:** By default, `dry_run: true` is set. Do not change to `false` until you have completed backtesting and dry-run paper trading.

### 4. Run Backtest

```bash
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231
```

### 5. Start Paper Trading (Dry Run)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Access the Web UI at `http://localhost:8080`.

### 6. Stop

```bash
docker compose -f docker/docker-compose.yml down
```

---

## How AI Model Integration Works

The strategy `AIModelStrategy.py` demonstrates the integration pattern:

1. **Model Loading:** In `bot_start()` or `__init__`, load the model from `/freqtrade/user_data/models/`
2. **Feature Building:** In `populate_indicators()`, calculate technical indicators as features
3. **Inference:** In `populate_entry_trend()` / `populate_exit_trend()`, call `model.predict()`
4. **Signal Mapping:** Convert model output (e.g., probability > 0.6) to Freqtrade's `enter_long`, `exit_long` columns

### Model File Naming Convention

- Sklearn models: `freqtrade/user_data/models/sklearn_model.pkl`
- PyTorch models: `freqtrade/user_data/models/pytorch_model.pt`
- Feature config: `freqtrade/user_data/models/feature_config.json` (feature list, scaler params, etc.)

---

## Key Files Explained

| File | Purpose | When to Modify |
|------|---------|----------------|
| `freqtrade/config.json` | Exchange keys, trading pairs, timeframe, risk limits | Always edit before first run |
| `freqtrade/user_data/strategies/AIModelStrategy.py` | Core strategy logic + AI inference | Modify model path and signal thresholds |
| `ai_engine/train_sklearn.py` | Train tree-based models | When you want to try new features or algorithms |
| `ai_engine/train_pytorch.py` | Train neural networks | For LSTM/Transformer experiments |
| `ai_engine/features.py` | Shared feature engineering | When adding new crypto-specific features (funding rate, OI, etc.) |
| `docker/docker-compose.yml` | Container orchestration | Rarely needed |

---

## Common Commands

```bash
# Download historical data
 docker compose -f docker/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20230101-20241231

# Hyperparameter optimization
 docker compose -f docker/docker-compose.yml run --rm freqtrade \
    hyperopt --strategy AIModelStrategy --spaces buy roi stoploss

# List available strategies
 docker compose -f docker/docker-compose.yml run --rm freqtrade list-strategies

# View logs
 docker compose -f docker/docker-compose.yml logs -f freqtrade

# Enter container shell
 docker compose -f docker/docker-compose.yml exec freqtrade /bin/bash
```

---

## Important Constraints & Notes

1. **Never commit API keys.** `config.json` contains secrets. It is listed in `.gitignore`.
2. **Dry Run First.** Freqtrade defaults to `dry_run: true`. Verify profitability in backtest + paper trade before live trading.
3. **Crypto-Specific Risks:**
   - Exchanges may delist pairs suddenly
   - Funding rate changes affect perpetual positions
   - High volatility causes stop-loss slippage
4. **AI Model Lifecycle:** Models go stale quickly in crypto. Plan for weekly/bi-weekly retraining.
5. **Freqtrade Limitations:**
   - Freqtrade's backtesting assumes immediate fill (no order book simulation)
   - For HFT (<1m timeframe), Freqtrade is not suitable; consider Nautilus Trader instead
   - Native funding rate data requires custom data provider extensions

---

## Roadmap Ideas

- [ ] Integrate on-chain data (exchange inflows/outflows) via Glassnode/Dune API
- [ ] Add funding rate and open interest features
- [ ] Build RL-based position sizing module
- [ ] Telegram bot alerts for model drift detection
- [ ] Multi-exchange arbitrage strategy (may require Hummingbot instead)

---

## References

- [Freqtrade Documentation](https://www.freqtrade.io/en/stable/)
- [Freqtrade Strategy Customization](https://www.freqtrade.io/en/stable/strategy-customization/)
- [Freqtrade with Machine Learning](https://www.freqtrade.io/en/stable/freqai/)
- [ccxt Documentation](https://docs.ccxt.com/)
