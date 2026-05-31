# 变更日志 — fix-critical-review

> 记录 Stage 2 审查发现的 Critical 问题及修复过程。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-31 | review | Stage 2 代码质量审查 | 发现 15 个 Critical 问题 |
| 2026-05-31 | fix | Task 1-4 全部修复 | 除零/inf + 参数化 + 策略安全 |
| 2026-05-31 | verify | 回归验证 | 60/60 测试通过，ruff 通过 |

## 修复清单

### features.py — 除零/inf（7 个）

| 函数 | 问题 | 修复方式 |
|------|------|---------|
| `rsi()` | `avg_loss=0` 时 `rs=inf` | `avg_loss.replace(0, np.nan)` + 手动恢复 RSI=100 |
| `adx()` | `plus_di + minus_di = 0` 时除零 | `tr_smooth.replace(0, np.nan)` + `di_sum.replace(0, np.nan)` |
| `stoch()` | `highest_high == lowest_low` 时除零 | `range_safe.replace(0, np.nan)` |
| `williams_r()` | 同上 | `range_safe.replace(0, np.nan)` |
| `cci()` | `mean_dev=0` 时除零 | `mean_dev.replace(0, np.nan)` |
| `add_volume_features()` | `volume_sma=0` 时 `volume_ratio=inf` | `volume_sma_col.replace(0, np.nan)` |
| `vwap()` | `volume.cumsum()=0` 时除零 | `volume_cumsum.replace(0, np.nan)` + 注释说明全历史累积模式 |

### features.py — 参数化（3 个）

| 位置 | 问题 | 修复方式 |
|------|------|---------|
| `FeatureRegistry.compute()` | `config or meta.get("params")` 空字典被覆盖 | `config if config is not None else meta.get("params")` |
| `add_candle_features()` | 未使用 `params` 参数 | 添加 `params = params or FEATURE_PARAMS` + 注释说明 candle EMA 长度固定 |
| `add_lag_features()` | 空字典 `{}` 导致 `TypeError` | `params if params is not None else FEATURE_PARAMS` |

### strategy_ai_model_v1.py + config（5 个）

| 位置 | 问题 | 修复方式 |
|------|------|---------|
| `config_ai_model.json` | `trading_mode: "spot"` vs `defaultType: "future"` | 统一为 `trading_mode: "futures"`, `margin_mode: "isolated"` |
| `add_higher_timeframe_features()` | 注释假设主框架为 1h | 更新注释说明支持任意主框架 |
| `_predict_sequence_model()` | 前 lookback 条 0.5 无日志 | 添加 `skipped` 计数 + `logger.info` |
| `populate_exit_trend()` | `rsi_14` 缺失时崩溃 | 防御性检查 `if "rsi_14" in dataframe.columns` |
| `populate_indicators()` | 外部数据 since/until 范围过大 | 限制为最近 90 天窗口 |

## 知识发现

- [x] **安全除法模式**: pandas Series 的 `.replace(0, np.nan)` 是避免除零 inf 的标准做法，修复后需 `fillna()` 处理边界值 → `features.py`
- [x] **空字典防御**: Python 中 `{}` 是 truthy，`params or FEATURE_PARAMS` 不会 fallback，必须用 `is not None` 判断 → `features.py`
- [x] **配置一致性**: `trading_mode` 和 `exchange.ccxt_config.options.defaultType` 必须匹配，否则交易行为异常 → `config_*.json`
- [x] **外部数据窗口限制**: 回测时请求全周期数据会导致缓存膨胀，限制为合理窗口（如 90 天）→ `strategy_ai_model_v1.py`

## 代码质量备忘

- 全部 60 个测试通过
- ruff check 通过
- 无新增测试失败
- 修复未改变任何特征的语义行为（除 vwap 注释外）
