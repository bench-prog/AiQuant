# AiQuant - AI Crypto Quant Trading Project

## Project Overview

AiQuant is a **personal AI-powered cryptocurrency quantitative trading system** built on top of [Freqtrade](https://www.freqtrade.io/). The goal is to minimize infrastructure building effort while allowing flexible integration of custom AI models (scikit-learn, PyTorch, etc.) for signal generation.

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
| AI/ML | scikit-learn, LightGBM, PyTorch, pandas, numpy |
| Feature Engineering | **Pure pandas/numpy** (no external TA libraries) |
| Data Fetching | ccxt (Binance public OHLCV) |
| Feature Storage | Parquet in `data/cache/` |
| Monitoring | Freqtrade Web UI (port 8080), Telegram Bot |
| Environment | Docker + Docker Compose |

---

## Directory Structure

```
AiQuant/
├── CLAUDE.md                           # This file
├── README.md                           # Human-readable quick start
├── setup.sh                            # Environment initialization script
├── data/                               # Data ingestion layer
│   ├── market_data.py                  # CCXT-based OHLCV / funding rate / OI downloader with caching
│   └── cache/                          # Parquet cache for downloaded market data
├── deploy/                             # Deployment configuration
│   ├── Dockerfile                      # Custom Freqtrade image (+LightGBM, +sklearn)
│   └── docker-compose.yml              # Freqtrade Docker service definition
├── freqtrade/                          # Freqtrade runtime
│   ├── config_ai_model.json            # AI model strategy configuration (API keys, pairs, risk)
│   ├── config_smallcap.json            # Small-cap momentum strategy configuration
│   └── user_data/
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── features.py             # **Shared indicators + feature pipelines**
│       │   ├── strategy_ai_model.py    # AI model strategy (self-contained, with drift monitor)
│       │   └── strategy_smallcap.py    # Small-cap momentum strategy (self-contained, with filters)
│       ├── models/                     # Trained AI models (.pkl, .pt) + feature_config.json + drift_baseline.json
│       ├── data/                       # Historical price data downloaded by Freqtrade
│       ├── notebooks/                  # Jupyter notebooks for research
│       └── logs/                       # Strategy logs + drift_alerts.jsonl
├── research/                           # AI model training scripts
│   ├── requirements.txt
│   ├── alert_cli.py                    # Standalone CLI for drift alerts
│   ├── train_classifier.py             # LightGBM with strict temporal split + drift baseline export
│   └── train_sequence.py               # LSTM with temporal split
├── tools/                              # Operational tools
│   └── update_smallcap_whitelist.py    # Refresh small-cap trading pair whitelist
└── .gitignore
```

---

## Quick Start

### 1. One-Click Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Docker/Docker Compose installation
- Create the directory structure
- Pull the `freqtradeorg/freqtrade:stable` base image

### 2. Build Custom Docker Image

```bash
docker compose -f deploy/docker-compose.yml build
```

The custom `deploy/Dockerfile` installs `lightgbm`, `scikit-learn`, `joblib`, and `numpy` on top of the official Freqtrade image. **No pandas-ta is required** — all indicators are implemented with pure pandas/numpy in `features.py`.

### 3. Train AI Model (No API Key Needed)

Training scripts download historical crypto data via **ccxt** from Binance. No exchange API key is required for public OHLCV data.

**Strict temporal split** is enforced to prevent look-ahead bias:
- **Training period:** 2022-01-01 ~ 2023-12-31
- **Test period:** 2024-01-01 ~ 2024-12-31 (held out for evaluation only)

```bash
cd research
pip install -r requirements.txt
python train_classifier.py        # Downloads BTC/USDT 1h, trains LightGBM
python train_sequence.py       # Or train an LSTM
```

Data is automatically cached to `data/cache/` as Parquet files to avoid re-downloading.

### 4. Configure Exchange API Keys (for Trading Only)

Edit `freqtrade/config_ai_model.json`:

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

### 5. Download Freqtrade Backtest Data

Freqtrade uses its own data format separate from the training cache.

```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20240101-20241231
```

### 6. Run Backtest

**AI Model Strategy** (default config):
```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231
```

**Small-Cap Momentum Strategy** (specify config):
```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    backtesting --config /freqtrade/config_smallcap.json --strategy SmallCapMomentumStrategy --timerange 20240101-20241231
```

### 7. Start Paper Trading (Dry Run)

**AI Model Strategy** (default):
```bash
docker compose -f deploy/docker-compose.yml up -d
```

**Small-Cap Momentum Strategy** (override command):
```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    trade --config /freqtrade/config_smallcap.json --strategy SmallCapMomentumStrategy
```

Access the Web UI at `http://localhost:8080`.

### 8. Stop

```bash
docker compose -f deploy/docker-compose.yml down
```

---

## How AI Model Integration Works

The strategy `strategy_ai_model.py` demonstrates the integration pattern:

1. **Model Loading:** In `bot_start()`, load the model and `feature_config.json` from `/freqtrade/user_data/models/`
2. **Feature Building:** In `populate_indicators()`, call `build_all_features()` from the shared `features.py` module — **identical to the training pipeline**
3. **Inference:** In `_predict_classifier()` / `_predict_sequence_model()`, use the exact feature column list stored in `feature_config.json` to ensure training/backtest consistency
4. **Signal Mapping:** In `populate_entry_trend()` / `populate_exit_trend()`, convert model probability to `enter_long` / `exit_long`

### Model File Naming Convention

- Sklearn models: `freqtrade/user_data/models/sklearn_model.pkl`
- PyTorch models: `freqtrade/user_data/models/pytorch_model.pt`
- Feature config: `freqtrade/user_data/models/feature_config.json` (feature list, scaler params, train/test date ranges, etc.)

---

## Key Files Explained

| File | Purpose | When to Modify |
|------|---------|----------------|
| `freqtrade/config_ai_model.json` | Exchange keys, trading pairs, timeframe, risk limits for AI model strategy | Always edit before first run |
| `freqtrade/config_smallcap.json` | Exchange keys, pair whitelist, protections for small-cap strategy | When running small-cap strategy |
| `freqtrade/user_data/strategies/features.py` | **Shared pure-pandas indicators and feature pipelines** | When adding new features — affects both training and backtest |
| `freqtrade/user_data/strategies/strategy_ai_model.py` | Core strategy + AI inference + inline drift monitor + Telegram alerts | Modify signal thresholds or add filters |
| `freqtrade/user_data/strategies/strategy_smallcap.py` | Small-cap momentum strategy with inline EMA/RSI filters | When tuning universe criteria or technical filters |
| `data/market_data.py` | CCXT OHLCV / funding rate / OI downloader with caching | When switching exchanges or timeframes |
| `research/train_classifier.py` | Train LightGBM + export drift baseline | When tuning hyperparameters or target horizon |
| `research/train_sequence.py` | Train LSTM with temporal split | For neural network experiments |
| `research/alert_cli.py` | Standalone CLI to send/test drift Telegram alerts | Rarely needed |
| `tools/update_smallcap_whitelist.py` | Refresh small-cap pair whitelist from CoinPaprika | When updating universe criteria or exchange mappings |
| `deploy/Dockerfile` | Custom image with ML dependencies | When adding new Python packages |
| `deploy/docker-compose.yml` | Container orchestration | Rarely needed |

---

## Common Commands

```bash
# Download historical data
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20230101-20241231

# Hyperparameter optimization
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    hyperopt --strategy AIModelStrategy --spaces buy roi stoploss

# List available strategies
docker compose -f deploy/docker-compose.yml run --rm freqtrade list-strategies

# View logs
docker compose -f deploy/docker-compose.yml logs -f freqtrade

# Enter container shell
docker compose -f deploy/docker-compose.yml exec freqtrade /bin/bash
```

---

## Important Constraints & Notes

1. **Never commit API keys.** `config_ai_model.json` and `config_smallcap.json` contain secrets. Both are listed in `.gitignore`.
2. **Dry Run First.** Freqtrade defaults to `dry_run: true`. Verify profitability in backtest + paper trade before live trading.
3. **Temporal Split Discipline:** Training scripts hard-code `TRAIN_END = "2023-12-31"`. Do not expand the training window to include the backtest period, or you will invalidate the results.
4. **AI Model Lifecycle:** Models go stale quickly in crypto. Plan for weekly/bi-weekly retraining.
5. **Crypto-Specific Risks:**
   - Exchanges may delist pairs suddenly
   - Funding rate changes affect perpetual positions
   - High volatility causes stop-loss slippage
6. **Freqtrade Limitations:**
   - Freqtrade's backtesting assumes immediate fill (no order book simulation)
   - For HFT (<1m timeframe), Freqtrade is not suitable; consider Nautilus Trader instead
   - Native funding rate data requires custom data provider extensions

---

## Completed

- [x] **Add funding rate and open interest features** — `market_data.py` + `features.py`, 9 new features, graceful fallback when data unavailable
- [x] **Telegram bot alerts for model drift detection** — Online drift monitor (stability index) in `strategy_ai_model.py`, `alert_cli.py` CLI, drift baseline exported during training

## Roadmap Ideas

- [ ] Integrate on-chain data (exchange inflows/outflows) via Glassnode/Dune API
- [ ] Build RL-based position sizing module
- [ ] Multi-exchange arbitrage strategy (may require Hummingbot instead)

---

## References

- [Freqtrade Documentation](https://www.freqtrade.io/en/stable/)
- [Freqtrade Strategy Customization](https://www.freqtrade.io/en/stable/strategy-customization/)
- [Freqtrade with Machine Learning](https://www.freqtrade.io/en/stable/freqai/)
- [ccxt Documentation](https://docs.ccxt.com/)
