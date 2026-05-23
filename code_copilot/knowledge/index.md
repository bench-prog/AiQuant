# 知识索引

> 领域知识的轻量索引。每条用一句话说清核心逻辑。
> 格式：`- **触发关键词**: 一句话核心逻辑 → `文件路径::类名/函数名`（可选）`

## 技术约定

- **特征工程入口**: 所有特征通过 `build_all_features(df)` 统一构建 → `freqtrade/user_data/strategies/features.py::build_all_features`
- **特征列提取**: 训练/推理使用 `get_feature_columns(df)` 自动剔除基础列 → `freqtrade/user_data/strategies/features.py::get_feature_columns`
- **数据服务注册**: 外部数据通过 `data.service.register()` 注册后用 `query()` 查询 → `data/service.py`
- **默认数据源**: `import data.service_defaults` 自动注册 funding_rate 和 open_interest → `data/service_defaults.py`
- **策略数据路径查找**: 策略通过候选路径列表同时支持本地开发和 Docker 环境 → `strategy_ai_model_v1.py` / `strategy_smallcap_v3_regime.py`
- **模型配置一致性**: `feature_config.json` 存储特征列顺序 + scaler 参数，训练/推理必须严格一致 → `freqtrade/user_data/models/feature_config.json`
- **资金费率特征降级**: `add_funding_rate_features` 在列不存在时原样返回，策略不崩溃 → `features.py::add_funding_rate_features`
- **持仓量特征降级**: `add_open_interest_features` 在列不存在时原样返回，策略不崩溃 → `features.py::add_open_interest_features`
- **训练脚本公共模块**: `research/data_utils.py` 封装 `load_training_data()` 和 `merge_external_data()`，训练脚本复用 → `research/data_utils.py`
- **模型配置命名约定**: 配置文件按 `feature_config_{model_type}.json`（lightgbm/lstm），漂移基线按 `drift_baseline_{model_type}.json` → `freqtrade/user_data/models/`
- **时序数据 DataLoader**: 时序模型训练必须设置 `shuffle=False`，否则破坏时间顺序导致数据泄漏

## 特征清单

| 类别 | 特征名 | 说明 |
|------|--------|------|
| 趋势 | ema_12/26/50, macd/macd_signal/macd_hist | EMA + MACD |
| 动量 | rsi_14/6, stoch_k/d, cci_20 | RSI + 随机指标 + CCI |
| 波动 | atr_14, bb_lower/middle/upper/width | ATR + 布林带 |
| 成交量 | volume_sma_20, volume_ratio, obv, vwap | 成交量分析 |
| K线 | close_above_ema12/26, body_pct, upper/lower_wick_pct | 价格行为 |
| 滞后 | return_lag_1/2/3/5/10, volume_lag_1/2/3/5/10 | 历史收益率/成交量 |
| 时间 | hour, hour_sin/cos | 小时编码 |
| 收益 | return_6h/24h, volatility_12h | 收益率/波动率 |
| 资金费率 | funding_rate, funding_rate_ema_8, sign, change | 资金费率特征 |
| 持仓量 | open_interest, oi_ema_12/24, oi_change_1h/6h/24h, oi_velocity | OI 特征 |

## 踩坑记录

（随实践积累补充）
