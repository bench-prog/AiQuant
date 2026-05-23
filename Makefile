# AiQuant Makefile
# 常用开发命令快捷方式

.PHONY: help test lint train-classifier train-lstm backtest-ai backtest-smallcap

help:
	@echo "AiQuant 常用命令:"
	@echo "  make test              - 运行 pytest 测试"
	@echo "  make lint              - 运行 ruff 代码检查"
	@echo "  make train-classifier  - 训练 LightGBM 分类器"
	@echo "  make train-lstm        - 训练 PyTorch LSTM 模型"
	@echo "  make backtest-ai       - 回测 AI 模型策略"
	@echo "  make backtest-smallcap - 回测小市值策略"

test:
	pytest tests/ -v

lint:
	ruff check freqtrade/user_data/strategies/ data/ research/ tests/

train-classifier:
	cd research && python train_classifier.py

train-lstm:
	cd research && python train_sequence.py

backtest-ai:
	docker compose -f deploy/docker-compose.yml run --rm freqtrade \
		backtesting --strategy AIModelStrategy --timerange 20240101-20241231

backtest-smallcap:
	docker compose -f deploy/docker-compose.yml run --rm freqtrade \
		backtesting --config /freqtrade/config_smallcap.json \
		--strategy SmallCapRegimeStrategy --timerange 20240101-20241231
