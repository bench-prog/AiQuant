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
| External Data Service | Registry-based `data/service.py` |
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
│   ├── service.py                      # **Unified external data service** (registry-based)
│   ├── service_defaults.py             # Pre-registered default data sources
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
│       │   ├── strategy_ai_model_v1.py # AI model strategy (drift monitor, scaler, Telegram alerts)
│       │   ├── strategy_smallcap_v3_regime.py  # Small-cap regime-switching strategy (hyperopt)
│       │   ├── strategy_smallcap_v2_turtle.py  # Small-cap turtle strategy
│       │   └── strategy_smallcap_v1_event_driven.py  # Small-cap event-driven strategy
│       │   └── strategy_gold_pulse_v1.py     # Gold pulse传导策略
│       ├── models/                     # Trained AI models (.pkl, .pt) + feature_config + drift_baseline
│       ├── data/                       # Historical price data downloaded by Freqtrade
│       ├── notebooks/                  # Jupyter notebooks for research
│       └── logs/                       # Strategy logs + drift_alerts.jsonl
├── research/                           # AI model training scripts
│   ├── requirements.txt
│   ├── alert_cli.py                    # Standalone CLI for drift alerts
│   ├── train_classifier.py             # LightGBM with strict temporal split + drift baseline export
│   ├── train_sequence.py               # LSTM with temporal split + scaler export
│   ├── training_config.py              # **Shared training config** (symbol, timeframe, date ranges)
│   └── data_utils.py                   # **Shared data utilities** (load + merge external data)
├── tests/                              # Unit and integration tests
│   ├── conftest.py                     # Shared fixtures (synthetic OHLCV data)
│   └── test_features.py                # features.py full coverage (39 test cases)
├── tools/                              # Operational tools
│   └── update_smallcap_whitelist.py    # Refresh small-cap trading pair whitelist
├── cclt/                                # AI coding collaboration framework (skill-driven)
│   ├── README.md
│   ├── rules/                          # Project rules (coding style, security, domain)
│   ├── knowledge/                      # Domain knowledge index
│   └── changes/                        # Change management (templates + archives)
├── deploy/                             # Docker deployment
│   ├── Dockerfile
│   └── docker-compose.yml
├── Makefile                            # Common command shortcuts
├── pytest.ini                          # Test configuration
├── setup.sh
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

**Small-Cap Regime Strategy** (specify config):
```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    backtesting --config /freqtrade/config_smallcap.json --strategy SmallCapRegimeStrategy --timerange 20240101-20241231
```

### 7. Start Paper Trading (Dry Run)

**AI Model Strategy** (default):
```bash
docker compose -f deploy/docker-compose.yml up -d
```

**Small-Cap Regime Strategy** (override command):
```bash
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    trade --config /freqtrade/config_smallcap.json --strategy SmallCapRegimeStrategy
```

Access the Web UI at `http://localhost:8080`.

### 8. Stop

```bash
docker compose -f deploy/docker-compose.yml down
```

---

## How AI Model Integration Works

The strategy `strategy_ai_model_v1.py` demonstrates the integration pattern:

1. **Model Loading:** In `bot_start()`, load the model and `feature_config.json` from `/freqtrade/user_data/models/`
2. **Feature Building:** In `populate_indicators()`, call `build_all_features()` from the shared `features.py` module — **identical to the training pipeline**
3. **Inference:** In `_predict_classifier()` / `_predict_sequence_model()`, use the exact feature column list stored in `feature_config.json` to ensure training/backtest consistency. PyTorch models also load and apply `StandardScaler` parameters from the config.
4. **Signal Mapping:** In `populate_entry_trend()` / `populate_exit_trend()`, convert model probability to `enter_long` / `exit_long`

### Model File Naming Convention

- Sklearn models: `freqtrade/user_data/models/sklearn_model.pkl`
- PyTorch models: `freqtrade/user_data/models/pytorch_model.pt`
- Feature config: `freqtrade/user_data/models/feature_config_{model_type}.json` — `feature_config_lightgbm.json` or `feature_config_lstm.json`
- Drift baseline: `freqtrade/user_data/models/drift_baseline_{model_type}.json` — `drift_baseline.json` (legacy) or `drift_baseline_lstm.json`

---

## Key Files Explained

| File | Purpose | When to Modify |
|------|---------|----------------|
| `freqtrade/config_ai_model.json` | Exchange keys, trading pairs, timeframe, risk limits for AI model strategy | Always edit before first run |
| `freqtrade/config_smallcap.json` | Exchange keys, pair whitelist, protections for small-cap strategy | When running small-cap strategy |
| `freqtrade/user_data/strategies/features.py` | **Shared pure-pandas indicators and feature pipelines** | When adding new features — affects both training and backtest |
| `freqtrade/user_data/strategies/strategy_ai_model_v1.py` | Core strategy + AI inference + drift monitor + scaler + Telegram alerts | Modify signal thresholds or add filters |
| `freqtrade/user_data/strategies/strategy_smallcap_v3_regime.py` | Small-cap regime-switching strategy with hyperopt support | When tuning regime thresholds or filters |
| `data/market_data.py` | CCXT OHLCV / funding rate / OI downloader with caching | When switching exchanges or timeframes |
| `data/service.py` | **Unified data service layer** — registry-based external data access | When adding new external data sources |
| `data/service_defaults.py` | Pre-registers default data sources (funding_rate, open_interest) | When adding new default sources |
| `research/train_classifier.py` | Train LightGBM + export drift baseline | When tuning hyperparameters or target horizon |
| `research/train_sequence.py` | Train LSTM with temporal split + scaler export | For neural network experiments |
| `research/training_config.py` | Shared training configuration (symbol, timeframe, date ranges) | When changing training parameters |
| `research/data_utils.py` | Shared data utilities (load + merge external data) | When changing data fetching logic |
| `research/alert_cli.py` | Standalone CLI to send/test drift Telegram alerts | Rarely needed |
| `tests/test_features.py` | Unit and integration tests for features.py | When adding/modifying features |
| `tools/update_smallcap_whitelist.py` | Refresh small-cap pair whitelist from CoinPaprika | When updating universe criteria or exchange mappings |
| `cclt/` | AI coding collaboration framework (Spec-driven development, skill-driven) | When updating coding standards or workflows |
| `Makefile` | Common command shortcuts (test, train, backtest, lint) | When adding new common commands |
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
    hyperopt --strategy SmallCapRegimeStrategy --spaces buy roi stoploss

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
- [x] **Telegram bot alerts for model drift detection** — Online drift monitor (stability index) in `strategy_ai_model_v1.py`, `alert_cli.py` CLI, drift baseline exported during training
- [x] **Unified external data service layer** — `data/service.py` with registry-based `query()` interface, used by both training scripts and strategies
- [x] **LSTM model architecture consistency** — `SimpleLSTM` in strategy matches `CryptoLSTM` in training; dynamic arch loading from `feature_config.json`
- [x] **StandardScaler support for PyTorch inference** — Training exports scaler params, strategy loads and applies them during sequence model inference
- [x] **Backtest-compatible BTC market filter** — Lazy-loaded in `populate_indicators()` so regime strategy works in both backtest and live modes
- [x] **Code deduplication** — `_prepare_features()` shared between classifier and sequence prediction; `_fetch_paginated()` / `_init_exchange()` in `market_data.py`
- [x] **Training script refactoring** — Extracted `training_config.py` + `data_utils.py`, eliminated duplication, fixed LSTM `shuffle=True` data leakage
- [x] **Model config naming convention** — `feature_config_{model_type}.json` + `drift_baseline_{model_type}.json` to prevent overwrite conflicts
- [x] **pytest test framework** — 39 tests covering all features.py functions (unit + integration + degradation)
- [x] **Type annotation completeness** — `features.py` (3 functions) + `market_data.py` (5 functions)
- [x] **Makefile** — Common commands: `test`, `train-classifier`, `train-lstm`, `backtest-ai`, `backtest-smallcap`, `lint`
- [x] **cclt framework** — AI coding collaboration framework with Spec-driven development workflow (v2: skill-driven)

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

---

## Workflow — cclt 协作模式

> 本工程使用 **cclt** skill 驱动协作流程。SessionStart hook 自动加载 skill，
> 自然语言即可触发子命令，无需手动记命令。详见 `cclt/` 目录。

### 核心文档

| 文档 | 路径 | 用途 |
|------|------|------|
| 策略演进跟踪 | `cclt/knowledge/strategy-evolution.md` | 策略总览、演进时间线、路线图、变更记录 |
| 开发流程规范 | `cclt/process.md` | 何时创建变更、命名规范、审查检查清单 |
| 知识索引 | `cclt/knowledge/index.md` | 技术约定、特征清单、踩坑记录 |

### 每次策略变更必须记录

- **新增/修改策略** → 更新 `strategy-evolution.md` 策略总览 + 演进时间线
- **特征/模型变更** → 更新 `knowledge/index.md` 特征清单
- **cclt 变更归档** → 追加变更记录到 `strategy-evolution.md`

### 量化交易硬红线（始终生效）

- **涉及资金/交易逻辑变更** → ⚠️ 高亮提醒人工审查
- **涉及特征工程/模型推理变更** → ⚠️ 提醒检查训练-推理一致性
- **策略代码和训练脚本** 必须使用相同的特征计算逻辑
- **新增特征** 必须同步更新 `feature_config.json`
- **时序数据处理** 严格按时间顺序 split，`shuffle=False`
- **`dry_run: true`** 是默认状态，切换实盘必须经过代码审查

### Git 规范

- Commit message 格式：`[<变更名>] <中文简述>`
- 禁止 main 分支直接变更
- 禁止自动 push
- 代码检查前置：commit 前执行 `ruff check` / `mypy`
