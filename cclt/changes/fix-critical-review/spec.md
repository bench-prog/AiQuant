# fix-critical-review — 修复 Stage 2 Critical 问题

> status: done
> created: 2026-05-31
> complexity: 🔴复杂
> source: Stage 2 Code Quality Review (features-enhancement + features-parametrization + multi-pair-strategy)

## 1. 背景与目标

Stage 2 代码质量审查发现 15 个 Critical 问题，分布在 `features.py`（除零/inf + 参数化）和 `strategy_ai_model_v1.py` + `config_ai_model.json`（资金/安全/数据泄漏）。

**目标:** 修复所有 Critical 问题，确保代码在生产环境中安全运行。

## 2. 问题清单

### features.py — 除零/inf（7 个）
- `rsi()`: `avg_loss=0` 时 `rs=inf`
- `adx()`: `plus_di + minus_di = 0` 时除零
- `stoch()`: `highest_high == lowest_low` 时除零
- `williams_r()`: 同上
- `cci()`: `mean_dev=0` 时除零
- `add_volume_features()`: `volume_sma=0` 时 `volume_ratio=inf`
- `vwap()`: 全历史 cumsum 设计需评估

### features.py — 参数化（3 个）
- `FeatureRegistry.compute()`: `config` 优先级逻辑不明确
- `add_candle_features()`: 硬编码 `ema_12/26` 与 `FEATURE_PARAMS` 不同步
- `add_lag_features()`: 空字典 `{}` 导致 `TypeError`

### strategy_ai_model_v1.py + config（5 个）
- `config_ai_model.json`: `trading_mode: "spot"` vs `defaultType: "future"` 不匹配
- `add_higher_timeframe_features()`: 4h 主框架下注释/偏移量误导
- `_predict_sequence_model()`: 前 lookback 条硬编码 0.5 无日志
- `populate_exit_trend()`: `rsi_14` 缺失时崩溃
- `populate_indicators()`: 外部数据 since/until 范围过大

## 3. 修复策略

统一模式：**安全除法**（`.replace(0, np.nan)`）+ **防御性检查** + **参数同步**

## 4. 验证标准

- `pytest tests/test_features.py -v` 全部通过
- `ruff check` 无错误
- 修复后无新增测试失败

## 5. 执行日志

| Task | 状态 | 修复内容 | 验证 |
|------|------|---------|------|
| Task 1: features.py 除零/inf | ✅ | rsi/adx/stoch/williams_r/cci/volume_ratio/vwap 安全除法 | 60/60 测试通过 |
| Task 2: features.py 参数化 | ✅ | FeatureRegistry.compute 优先级、add_candle_features params、add_lag_features 空字典 | 60/60 测试通过 |
| Task 3: strategy + config | ✅ | trading_mode futures、LSTM 跳过日志、RSI 防御、外部数据 90 天窗口、注释更新 | ruff 通过 |
| Task 4: 回归验证 | ✅ | pytest + ruff | 全部通过 |

## 6. 审查结论

✅ 全部 15 个 Critical 问题已修复。
✅ 60/60 测试通过。
✅ ruff check 通过。
