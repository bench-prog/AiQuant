# features.py 增强

> status: propose
> created: 2026-05-23
> complexity: 🟡中等

## 1. 背景与目标

`features.py` 是 AiQuant 的核心共享特征模块，训练脚本和 Freqtrade 策略共用。当前实现已覆盖趋势、动量、波动率、成交量、K线、滞后、时间、收益、资金费率等特征（46 列），但存在以下问题：

1. **Bug: ADX 函数定义未使用** — `adx()` 已定义但未加入 `build_all_features()`，实际未生成特征
2. **性能: EMA 重复计算** — `add_candle_features()` 中重复计算 `ema(close, 12/26)`，而 `add_trend_features()` 已生成相同列
3. **缺失指标** — 缺少 Williams %R、Momentum、OBV 变化率、VWAP 偏离度等常用动量/趋势指标

**目标:** 修复已知 bug、减少冗余计算、补充高价值技术指标，保持纯 pandas/numpy 实现，确保训练-推理一致性。

## 2. 代码现状

### 2.1 相关入口与链路

- **特征入口**: `freqtrade/user_data/strategies/features.py::build_all_features()` → 按顺序调用各 add_* 函数
- **特征提取**: `features.py::get_feature_columns()` → 剔除 OHLCV + date 基础列
- **训练使用**: `research/train_classifier.py` / `train_sequence.py` → 调用 `build_all_features()`
- **策略使用**: `strategy_ai_model_v1.py::populate_indicators()` → 调用 `build_all_features()`
- **配置**: `freqtrade/user_data/models/feature_config.json` → 存储当前 46 列特征名 + scaler 参数
- **测试**: `tests/test_features.py` → 39 个用例覆盖所有现有函数

### 2.2 现有实现

| 类别 | 已有特征 | 列数 |
|------|---------|------|
| 趋势 | ema_12/26/50, macd/signal/hist | 6 |
| 动量 | rsi_14/6, stoch_k/d, cci_20 | 5 |
| 波动 | atr_14, bb_lower/middle/upper/width | 5 |
| 成交量 | volume_sma_20, volume_ratio, obv, vwap | 4 |
| K线 | close_above_ema12/26, body_pct, upper/lower_wick_pct | 5 |
| 滞后 | return_lag_1/2/3/5/10, volume_lag_1/2/3/5/10 | 10 |
| 时间 | hour, hour_sin/cos | 3 |
| 收益 | return_6h/24h, volatility_12h | 3 |
| 资金费率 | funding_rate, ema_8, sign, change | 4 |
| 持仓量 | open_interest, oi_ema_12/24, change_1h/6h/24h, velocity | 6 |
| **合计** | | **46** |

### 2.3 发现与风险

- **ADX 已定义未使用** (`features.py:L43-L78`): `adx()` 返回 `(adx, plus_di, minus_di)`，但 `add_trend_features()` 和 `build_all_features()` 均未调用
- **EMA 重复计算** (`features.py:L158-L159`): `add_candle_features()` 内联调用 `ema(df["close"], 12)` 和 `ema(df["close"], 26)`，与 `add_trend_features()` 生成的 `ema_12`/`ema_26` 冗余
- **测试覆盖 ADX 函数但非特征列**: `TestADX` 测试了函数返回值，但 `TestAddTrendFeatures` 未断言 ADX 列存在
- **feature_config.json 影响**: 新增特征后，现有模型配置（46 列 + scaler）与新增列数不一致，需要重新训练或保留旧配置兼容

## 3. 功能点

- [ ] **修复 ADX 未使用**: 在 `add_trend_features()` 中调用 `adx()`，生成 `adx_14`, `plus_di_14`, `minus_di_14` 列
- [ ] **修复 EMA 重复计算**: `add_candle_features()` 改为复用 `df["ema_12"]` / `df["ema_26"]` 而非重新计算
- [ ] **新增 Williams %R**: 14 周期威廉指标，加入 `add_momentum_features()`
- [ ] **新增 Momentum (MOM)**: 10 周期价格动量，加入 `add_momentum_features()`
- [ ] **新增 OBV 变化率**: `obv_change_1h` = OBV.diff(1)，加入 `add_volume_features()`
- [ ] **新增 VWAP 偏离度**: `vwap_distance` = (close - vwap) / vwap，加入 `add_volume_features()`
- [ ] **新增价格-布林带偏离**: `bb_position` = (close - bb_lower) / (bb_upper - bb_lower)，加入 `add_volatility_features()`
- [ ] **补充完整类型注解**: `adx()`, `bbands()`, `stoch()`, `cci()`, `obv()`, `vwap()` 返回值类型
- [ ] **同步测试**: 为新增特征补充测试用例，修复 ADX 列缺失断言
- [ ] **同步 feature_config**: 更新训练脚本中的特征列列表（或标记需要重新训练）

## 4. 业务规则

- 所有新增特征必须是纯 pandas/numpy 实现，禁止引入外部 TA 库
- 特征列名保持 snake_case，与现有命名风格一致
- 涉及未来信息的计算（如 `shift(-1)`）严格禁止
- 新特征在 `build_all_features()` 中的调用顺序：趋势 → 动量 → 波动率 → 成交量 → K线 → 滞后 → 时间 → 收益 → 资金费率 → 持仓量（已有顺序不变，新增特征插入对应类别）

## 5. 数据变更

| 操作 | 文件/配置 | 字段/参数 | 说明 |
|------|----------|----------|------|
| 新增列 | `features.py` | `adx_14`, `plus_di_14`, `minus_di_14` | ADX 趋势强度 |
| 新增列 | `features.py` | `williams_r_14` | Williams %R |
| 新增列 | `features.py` | `mom_10` | 价格动量 |
| 新增列 | `features.py` | `obv_change_1h` | OBV 变化 |
| 新增列 | `features.py` | `vwap_distance` | VWAP 偏离度 |
| 新增列 | `features.py` | `bb_position` | 布林带位置 |
| 修改 | `feature_config.json` | 列数 46 → 52 | 新增 6 列，需重新训练 |

## 6. 接口变更

| 操作 | 接口/函数 | 变更内容 |
|------|----------|---------|
| 修改 | `add_trend_features()` | 新增 ADX 调用 |
| 修改 | `add_momentum_features()` | 新增 Williams %R + MOM |
| 修改 | `add_volatility_features()` | 新增 bb_position |
| 修改 | `add_volume_features()` | 新增 obv_change_1h + vwap_distance |
| 修改 | `add_candle_features()` | 复用已有 ema_12/26 列 |
| 修改 | 多个指标函数 | 补充返回值类型注解 |

## 7. 影响范围

- **训练脚本**: `train_classifier.py`, `train_sequence.py` — 特征列数变化，需要重新训练（或标记旧模型不兼容）
- **策略代码**: `strategy_ai_model_v1.py` — 通过 `feature_config.json` 加载特征列，新增列需同步
- **测试**: `tests/test_features.py` — 需要补充新特征测试 + 修复 ADX 列断言
- **模型配置**: `feature_config.json`, `drift_baseline.json` — 需要重新生成

## 8. 风险与关注点

> ⚠️ 涉及特征工程变更 → **必须检查训练-推理一致性**
> ⚠️ 现有模型基于 46 列特征，新增 6 列后旧模型无法直接加载 — 需要重新训练或版本管理

- **模型兼容性**: 旧 `.pkl` / `.pt` 模型 + `feature_config.json` 与新增列不兼容
- **数据泄漏风险**: 所有新增特征仅使用当前及历史数据，不涉及未来信息
- **NaN/inf 风险**: 新增特征需通过 `test_no_inf` 验证

## 8.5 测试策略

- **测试范围**: features.py 全部函数（新增 + 现有）
- **覆盖率目标**: 新增特征每个都有列存在断言 + 值域断言
- **独立 Test Spec**: 否（直接在现有 test_features.py 中补充）

## 9. 待澄清

- [ ] **Q1: 旧模型兼容性** — 是否需要保留旧 feature_config.json 作为备份，还是直接覆盖？
- [ ] **Q2: 新增特征数量** — 6 列是否合适，还是更多/更少？
- [ ] **Q3: 是否需要立即重新训练** — 还是仅更新特征模块，训练留到后续？

## 10. 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| ADX 窗口 | 14（与 ATR/RSI 一致） | 行业标准 |
| Williams %R 窗口 | 14 | 与 RSI 同周期，便于对比 |
| MOM 窗口 | 10 | 覆盖约半交易日（1h 周期） |
| VWAP 偏离 | 百分比标准化 | 消除价格量级影响 |
| BB position | [0, 1] 归一化 | 与 bb_width 互补 |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|

## 12. 审查结论

## 13. 确认记录（HARD-GATE）

- **确认时间**:
- **确认人**:
