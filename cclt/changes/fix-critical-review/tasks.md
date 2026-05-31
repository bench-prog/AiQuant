# 任务拆分 — 修复 Critical 问题

## Task 1: features.py 除零/inf 修复

- **目标**: 修复 7 个指标函数的除零/inf 风险
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — `rsi()`, `adx()`, `stoch()`, `williams_r()`, `cci()`, `add_volume_features()`
- **修复模式**: 分母 `.replace(0, np.nan)`
- **验证**: `pytest tests/test_features.py -v`

## Task 2: features.py 参数化修复

- **目标**: 修复 3 个参数化相关 Critical 问题
- **涉及文件**:
  - `features.py` — `FeatureRegistry.compute()`, `add_candle_features()`, `add_lag_features()`
- **验证**: `pytest tests/test_features.py -v`

## Task 3: strategy + config 修复

- **目标**: 修复 5 个策略/配置 Critical 问题
- **涉及文件**:
  - `strategy_ai_model_v1.py` — LSTM lookback 日志、RSI 防御、higher_tf 断言、外部数据窗口
  - `config_ai_model.json` — trading_mode 统一
- **验证**: 配置文件语法检查 + 源码审查

## Task 4: 回归验证

- **目标**: 完整测试套件 + ruff
- **验证**:
  - `pytest tests/test_features.py -v`
  - `ruff check freqtrade/user_data/strategies/features.py`
