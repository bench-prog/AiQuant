# AiQuant - AI Crypto Trading Bot

基于 [Freqtrade](https://www.freqtrade.io/) 的 AI 量化交易系统，专为加密货币设计。

## 快速启动

```bash
# 1. 一键配置环境
chmod +x setup.sh
./setup.sh

# 2. 构建自定义 Docker 镜像（含 LightGBM + scikit-learn）
docker compose -f deploy/docker-compose.yml build

# 3. 训练 AI 模型（自动从币安下载历史数据，无需 API Key）
# 模型使用 2022-2023 年数据训练，2024 年数据仅用于验证
cd research
pip install -r requirements.txt
python train_classifier.py     # LightGBM 分类器
python train_sequence.py       # 或 LSTM

# 4. 下载 Freqtrade 回测数据
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    download-data --pairs BTC/USDT ETH/USDT --timeframe 1h --timerange 20240101-20241231

# 5. 回测
docker compose -f deploy/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231

# 6. 启动模拟交易（Web UI: http://localhost:8080）
docker compose -f deploy/docker-compose.yml up -d
```

## 项目结构

- `freqtrade/` - Freqtrade 配置与策略
  - `user_data/strategies/features.py` - **共享特征工程**（训练与回测共用）
  - `user_data/strategies/strategy_ai_model.py` - AI 模型策略（含漂移监控）
  - `user_data/strategies/strategy_smallcap.py` - 小市值动量策略（含技术过滤器）
- `data/` - 数据采集与缓存（OHLCV、资金费率、持仓量）
- `research/` - AI 模型训练脚本
  - `train_classifier.py` - LightGBM 分类器训练
  - `train_sequence.py` - LSTM 序列模型训练
  - `alert_cli.py` - 漂移告警 CLI
- `tools/` - 业务运维工具
  - `update_smallcap_whitelist.py` - 刷新小市值交易对白名单
- `deploy/` - Docker 部署配置
- `setup.sh` - 环境初始化脚本

## 核心设计

- **纯 pandas/numpy 技术指标**：不依赖 pandas-ta / numba，兼容 Python 3.14
- **严格时间切分**：训练集仅到 2023-12-31，避免数据泄露
- **共享特征模块**：`features.py` 同时用于训练脚本和 Freqtrade 策略，确保特征一致
- **合约特征增强**：已集成资金费率（funding rate）和持仓量（open interest）特征
- **模型漂移监控**：在线稳定性指数检测 + Telegram 告警，实时发现模型失效

## 安全提示

- `dry_run` 默认为 `true`，实盘前务必完成回测与模拟交易
- `config_ai_model.json` 已加入 `.gitignore`，切勿泄露 API Key
