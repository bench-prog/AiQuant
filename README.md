# AiQuant - AI Crypto Trading Bot

基于 [Freqtrade](https://www.freqtrade.io/) 的 AI 量化交易系统，专为加密货币设计。

## 快速启动

```bash
# 1. 一键配置环境
chmod +x scripts/setup.sh
./scripts/setup.sh

# 2. 训练 AI 模型（自动从币安下载历史数据，无需 API Key）
cd ai_engine
pip install -r requirements.txt
python train_sklearn.py        # LightGBM 分类器
python train_pytorch.py        # 或 LSTM

# 3. 回测
docker compose -f docker/docker-compose.yml run --rm freqtrade \
    backtesting --strategy AIModelStrategy --timerange 20240101-20241231

# 4. 启动模拟交易（Web UI: http://localhost:8080）
docker compose -f docker/docker-compose.yml up -d
```

## 项目结构

- `freqtrade/` - Freqtrade 配置与策略
- `ai_engine/` - AI 模型训练脚本
- `docker/` - Docker 编排
- `CLAUDE.md` - AI 助手可读的项目文档

## 安全提示

- `dry_run` 默认为 `true`，实盘前务必完成回测与模拟交易
- `config.json` 已加入 `.gitignore`，切勿泄露 API Key
