# AiQuant - AI Crypto Trading Bot

基于 [Freqtrade](https://www.freqtrade.io/) 的 AI 量化交易系统，专为加密货币设计。

## 快速启动

```bash
# 1. 一键配置环境
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. 构建自定义 Docker 镜像（含 LightGBM + scikit-learn）
docker compose -f docker/docker-compose.yml build

# 3. 训练 AI 模型（自动从币安下载历史数据，无需 API Key）
# 模型使用 2022-2023 年数据训练，2024 年数据仅用于验证
cd ai_engine
pip install -r requirements.txt
python train_sklearn.py        # LightGBM 分类器
python train_pytorch.py        # 或 LSTM

# 4. 下载 Freqtrade 回测数据
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20240101-20241231

# 5. 回测
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231

# 6. 启动模拟交易（Web UI: http://localhost:8080）
docker compose -f docker/docker-compose.yml up -d
```

## 项目结构

- `freqtrade/` - Freqtrade 配置与策略
  - `user_data/strategies/feature_engineering.py` - **共享特征工程**（训练与回测共用）
  - `user_data/strategies/drift_utils.py` - **模型漂移检测工具**（PSI 计算、Telegram 告警）
- `ai_engine/` - AI 模型训练脚本
  - `data_fetcher.py` - 数据下载（含资金费率、持仓量）
  - `drift_telegram.py` - Telegram 漂移告警独立脚本
- `docker/` - Docker 编排与自定义镜像
- `CLAUDE.md` - AI 助手可读的项目文档

## 核心设计

- **纯 pandas/numpy 技术指标**：不依赖 pandas-ta / numba，兼容 Python 3.14
- **严格时间切分**：训练集仅到 2023-12-31，避免数据泄露
- **共享特征模块**：`feature_engineering.py` 同时用于训练脚本和 Freqtrade 策略，确保特征一致
- **合约特征增强**：已集成资金费率（funding rate）和持仓量（open interest）特征
- **模型漂移监控**：在线 PSI 检测 + Telegram 告警，实时发现模型失效

## 安全提示

- `dry_run` 默认为 `true`，实盘前务必完成回测与模拟交易
- `config.json` 已加入 `.gitignore`，切勿泄露 API Key
