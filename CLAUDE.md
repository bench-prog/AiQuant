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
| Feature Storage | Parquet in `ai_engine/cache/` |
| Monitoring | Freqtrade Web UI (port 8080), Telegram Bot |
| Environment | Docker + Docker Compose |

---

## Directory Structure

```
AiQuant/
├── CLAUDE.md                           # This file
├── README.md                           # Human-readable quick start
├── docker/
│   ├── Dockerfile                      # Custom Freqtrade image (+LightGBM, +sklearn)
│   └── docker-compose.yml              # Freqtrade Docker service definition
├── scripts/
│   └── setup.sh                        # One-click setup script
├── freqtrade/
│   ├── config.json                     # Main Freqtrade configuration (API keys, pairs, risk)
│   └── user_data/
│       ├── strategies/
│       │   ├── __init__.py
│       │   ├── feature_engineering.py  # **Shared indicators + feature pipelines**
│       │   ├── drift_utils.py          # **Drift detection (PSI) + Telegram sender**
│       │   └── AIModelStrategy.py      # Strategy with AI model loading + online drift monitor
│       ├── models/                     # Trained AI models (.pkl, .pt) + feature_config.json + drift_baseline.json
│       ├── data/                       # Historical price data downloaded by Freqtrade
│       ├── notebooks/                  # Jupyter notebooks for research
│       └── logs/                       # Strategy logs + drift_alerts.jsonl
├── ai_engine/                          # Local AI model training scripts
│   ├── requirements.txt
│   ├── data_fetcher.py                 # CCXT-based OHLCV / funding rate / open interest downloader with caching
│   ├── features.py                     # Re-exports from feature_engineering.py
│   ├── drift_telegram.py               # Standalone CLI for drift Telegram alerts
│   ├── train_sklearn.py                # LightGBM with strict temporal split + drift baseline export
│   └── train_pytorch.py              # LSTM with strict temporal split
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
- Pull the `freqtradeorg/freqtrade:stable` base image

### 2. Build Custom Docker Image

```bash
docker compose -f docker/docker-compose.yml build
```

The custom `docker/Dockerfile` installs `lightgbm`, `scikit-learn`, `joblib`, and `numpy` on top of the official Freqtrade image. **No pandas-ta is required** — all indicators are implemented with pure pandas/numpy in `feature_engineering.py`.

### 3. Train AI Model (No API Key Needed)

Training scripts download historical crypto data via **ccxt** from Binance. No exchange API key is required for public OHLCV data.

**Strict temporal split** is enforced to prevent look-ahead bias:
- **Training period:** 2022-01-01 ~ 2023-12-31
- **Test period:** 2024-01-01 ~ 2024-12-31 (held out for evaluation only)

```bash
cd ai_engine
pip install -r requirements.txt
python train_sklearn.py        # Downloads BTC/USDT 1h, trains LightGBM
python train_pytorch.py        # Or train an LSTM
```

Data is automatically cached to `ai_engine/cache/` as Parquet files to avoid re-downloading.

### 4. Configure Exchange API Keys (for Trading Only)

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

### 5. Download Freqtrade Backtest Data

Freqtrade uses its own data format separate from the training cache.

```bash
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20240101-20241231
```

### 6. Run Backtest

```bash
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231
```

### 7. Start Paper Trading (Dry Run)

```bash
docker compose -f docker/docker-compose.yml up -d
```

Access the Web UI at `http://localhost:8080`.

### 8. Stop

```bash
docker compose -f docker/docker-compose.yml down
```

---

## How AI Model Integration Works

The strategy `AIModelStrategy.py` demonstrates the integration pattern:

1. **Model Loading:** In `bot_start()`, load the model and `feature_config.json` from `/freqtrade/user_data/models/`
2. **Feature Building:** In `populate_indicators()`, call `build_all_features()` from the shared `feature_engineering.py` module — **identical to the training pipeline**
3. **Inference:** In `_predict_sklearn()` / `_predict_pytorch()`, use the exact feature column list stored in `feature_config.json` to ensure training/backtest consistency
4. **Signal Mapping:** In `populate_entry_trend()` / `populate_exit_trend()`, convert model probability to `enter_long` / `exit_long`

### Model File Naming Convention

- Sklearn models: `freqtrade/user_data/models/sklearn_model.pkl`
- PyTorch models: `freqtrade/user_data/models/pytorch_model.pt`
- Feature config: `freqtrade/user_data/models/feature_config.json` (feature list, scaler params, train/test date ranges, etc.)

---

## Key Files Explained

| File | Purpose | When to Modify |
|------|---------|----------------|
| `freqtrade/config.json` | Exchange keys, trading pairs, timeframe, risk limits | Always edit before first run |
| `freqtrade/user_data/strategies/feature_engineering.py` | **Shared pure-pandas indicators and feature pipelines** | When adding new features — affects both training and backtest |
| `freqtrade/user_data/strategies/AIModelStrategy.py` | Core strategy + AI inference + online drift monitor | Modify signal thresholds or add filters |
| `freqtrade/user_data/strategies/drift_utils.py` | PSI computation, Telegram alert sender, JSONL logger | When tuning drift thresholds or alert format |
| `ai_engine/train_sklearn.py` | Train LightGBM + export drift baseline | When tuning hyperparameters or target horizon |
| `ai_engine/train_pytorch.py` | Train LSTM with temporal split | For neural network experiments |
| `ai_engine/data_fetcher.py` | CCXT OHLCV / funding rate / OI downloader with caching | When switching exchanges or timeframes |
| `ai_engine/drift_telegram.py` | Standalone CLI to send/test drift Telegram alerts | Rarely needed |
| `docker/Dockerfile` | Custom image with ML dependencies | When adding new Python packages |
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

- [x] **Add funding rate and open interest features** — `data_fetcher.py` + `feature_engineering.py`, 9 new features, graceful fallback when data unavailable
- [x] **Telegram bot alerts for model drift detection** — Online PSI monitor in `AIModelStrategy.py`, `drift_utils.py`, `drift_telegram.py` CLI, drift baseline exported during training

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
