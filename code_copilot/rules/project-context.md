---
alwaysApply: true
---

# 工程上下文

## 1. 应用概况

- 应用名: AiQuant
- 简介: 基于 Freqtrade 的 AI 驱动加密货币量化交易系统
- 技术栈: Python 3.11 / Freqtrade (Docker) / scikit-learn / LightGBM / PyTorch / pandas / numpy / ccxt
- 构建工具: Docker + Docker Compose
- 数据获取: ccxt (Binance 为主)
- 特征工程: 纯 pandas/numpy，不依赖外部 TA 库（如 pandas-ta）

## 2. 目录结构与模块职责

```
AiQuant/
├── data/                   # 数据层
│   ├── market_data.py      # CCXT OHLCV/资金费率/OI 下载器（带缓存）
│   ├── service.py          # 统一外部数据服务（注册表模式）
│   └── cache/              # Parquet 缓存
├── deploy/                 # 部署层
│   ├── Dockerfile          # 自定义 Freqtrade 镜像
│   └── docker-compose.yml  # Docker Compose 配置
├── freqtrade/              # Freqtrade 运行时
│   ├── config_*.json       # 策略配置文件
│   └── user_data/
│       ├── strategies/     # 策略代码
│       │   ├── features.py         # 共享特征工程（纯 pandas/numpy）
│       │   ├── strategy_ai_model_v1.py
│       │   ├── strategy_smallcap_v3_regime.py
│       │   └── ...
│       ├── models/         # 训练好的模型 (.pkl, .pt)
│       └── logs/           # 策略日志
├── research/               # 模型训练脚本
│   ├── train_classifier.py # LightGBM 训练 + 漂移基线导出
│   ├── train_sequence.py   # LSTM 训练
│   └── alert_cli.py        # 漂移告警 CLI
└── tools/                  # 运维工具
    └── update_smallcap_whitelist.py
```

## 3. 分层架构

```
策略层 (strategies/*.py)    ← Freqtrade 策略，调用模型推理
    ↓
特征层 (features.py)        ← 纯 pandas/numpy 指标计算
    ↓
数据层 (data/*.py)          ← CCXT 数据获取 + 缓存
    ↓
模型层 (research/*.py)      ← 训练脚本 + 模型导出
```

## 4. 关键依赖

| 依赖 | 用途 | 备注 |
|------|------|------|
| Freqtrade | 交易引擎 | Docker 运行 |
| ccxt | 交易所 API | 获取 OHLCV/资金费率 |
| pandas/numpy | 数据处理 | 特征工程纯手写，无外部 TA 库 |
| scikit-learn | 机器学习 | 分类模型 + StandardScaler |
| lightgbm | 梯度提升 | 主力分类器 |
| PyTorch | 深度学习 | LSTM 序列模型 |
| joblib | 模型序列化 | .pkl 模型保存 |

## 5. 现有策略清单

| 策略文件 | 类型 | 说明 |
|----------|------|------|
| `strategy_ai_model_v1.py` | AI 模型策略 | 加载 sklearn/LSTM 模型，特征工程驱动 |
| `strategy_smallcap_v3_regime.py` | 小市值策略 | Regime Switching + Hyperopt 优化 |
| `strategy_smallcap_v2_turtle.py` | 小市值策略 | Turtle 趋势跟踪 |
| `strategy_smallcap_v1_event_driven.py` | 小市值策略 | 事件驱动 |
| `strategy_gold_pulse_v1.py` | 黄金脉冲策略 | （待补充） |

## 6. 特征工程 (features.py)

- **指标**: EMA, RSI, MACD, ATR, ADX, Bollinger Bands, Stochastic, CCI, VWAP, OBV
- **价格行为**: 实体比例、影线比例、收盘价与均线关系
- **滞后特征**: 收益率/成交量滞后 1/2/3/5/10 周期
- **时间特征**: 小时正弦/余弦编码
- **资金费率**: 资金费率原始值、EMA、符号、变化率
- **总特征数**: 46 列（见 `feature_config.json`）

## 7. 模型配置

- **模型类型**: LSTM（当前活跃）/ sklearn（备选）
- **序列长度**: 20 (`lookback`)
- **预测 horizon**: 1 (`horizon`)
- **隐藏层**: 64, 2 层
- **训练期**: 2022-01-01 ~ 2023-12-31
- **Scaler**: StandardScaler（均值/方差保存在 `feature_config.json`）

## 8. 数据缓存

- **OHLCV**: `data/cache/binance_<symbol>_<timeframe>_<start>_<end>.parquet`
- **资金费率**: `data/cache/binance_<symbol>_funding_rate_<start>_<end>.parquet`
- **Freqtrade 数据**: `freqtrade/user_data/data/binance/<symbol>-<timeframe>.feather`
- **小市值清单**: `freqtrade/user_data/data/smallcap_universe.json`
