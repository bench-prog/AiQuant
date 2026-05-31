# features.py 参数化重构 — 渐进路径

> status: done
> created: 2026-05-23
> complexity: 🟡中等

## 1. 背景与目标

当前 `features.py` 中所有指标窗口、周期均为硬编码（magic number）。这导致：
- 超参优化时无法调整特征窗口
- 不同交易对/周期需要改代码
- 消融实验无法自动化

**目标**: 将硬编码参数抽离为可配置字典，保持 100% 向后兼容，为后续特征注册表和 Feature Store 打下基础。

## 2. 渐进路径设计

### 阶段 1: 参数化配置（本次）

将 magic number 抽成 `FEATURE_PARAMS` 字典，`add_*_features()` 和 `build_all_features()` 支持传入配置。默认值 = 当前硬编码值。

### 阶段 2: 特征注册表（后续）

`FeatureRegistry` 类管理特征元数据，支持按需计算、特征版本管理。

### 阶段 3: 配置驱动流水线（远期）

YAML/JSON 配置定义特征集，`build_all_features()` 从配置构建，支持多特征集 A/B 实验。

### 阶段 4: Feature Store（远期）

在线/离线一致性校验、特征缓存、血缘追踪。

## 3. 代码现状

当前 `features.py` 中硬编码参数：

| 函数 | 硬编码值 |
|------|---------|
| `add_trend_features` | ema(12,26,50), macd(12,26,9), adx(14) |
| `add_momentum_features` | rsi(14,6), stoch(14,3), cci(20), williams_r(14), mom(10) |
| `add_volatility_features` | atr(14), bbands(20,2.0) |
| `add_volume_features` | volume_sma_20, obv, vwap |
| `add_lag_features` | lags=[1,2,3,5,10] |
| `add_return_features` | pct_change(6,24), rolling(12) |
| `add_funding_rate_features` | ema(8) |
| `add_open_interest_features` | ema(12,24), diff(1,6,24) |

## 4. 阶段 1 功能点

- [ ] 创建 `FEATURE_PARAMS` 字典，集中管理所有参数
- [ ] `add_trend_features(df, params=None)` 支持自定义 EMA/MACD/ADX 参数
- [ ] `add_momentum_features(df, params=None)` 支持自定义 RSI/Stoch/CCI/WR/MOM 参数
- [ ] `add_volatility_features(df, params=None)` 支持自定义 ATR/BBands 参数
- [ ] `add_volume_features(df, params=None)` 支持自定义 Volume/OBV/VWAP 参数
- [ ] `add_lag_features(df, params=None)` 支持自定义滞后周期
- [ ] `add_return_features(df, params=None)` 支持自定义收益/波动率窗口
- [ ] `add_funding_rate_features(df, params=None)` 支持自定义资金费率参数
- [ ] `add_open_interest_features(df, params=None)` 支持自定义 OI 参数
- [ ] `build_all_features(df, config=None)` 统一接收配置
- [ ] 测试验证默认配置输出与之前完全一致（向后兼容）

## 5. 数据变更

| 操作 | 文件/配置 | 说明 |
|------|----------|------|
| 新增 | `features.py` | `FEATURE_PARAMS` 字典 + 函数签名变更 |
| 修改 | `tests/test_features.py` | 新增参数化测试 |

## 6. 接口变更

| 操作 | 接口/函数 | 变更内容 |
|------|----------|---------|
| 修改 | `add_*_features(df)` → `add_*_features(df, params=None)` | 新增可选参数 |
| 修改 | `build_all_features(df)` → `build_all_features(df, config=None)` | 新增可选参数 |

## 7. 影响范围

- **策略代码**: `strategy_ai_model_v1.py` — 调用 `build_all_features()` 无变更（向后兼容）
- **训练脚本**: `train_classifier.py` / `train_sequence.py` — 调用 `build_all_features()` 无变更
- **测试**: `tests/test_features.py` — 需补充参数化测试

## 8. 风险与关注点

> ⚠️ 涉及特征工程变更 → **必须保证默认配置输出与之前完全一致**

- **向后兼容性**: 默认参数必须 = 当前硬编码值，否则训练-推理一致性被破坏
- **列名稳定性**: 默认配置下生成的列名必须与之前完全相同

## 8.5 测试策略

- **回归测试**: 默认配置下 `build_all_features()` 输出与之前完全一致
- **参数化测试**: 自定义参数下正确生成对应列
- **覆盖率目标**: 100%

## 9. 待澄清

- [ ] **Q1**: 阶段 1 只做参数化，不做特征注册表？
- [ ] **Q2**: `FEATURE_PARAMS` 放在 `features.py` 内还是单独 `features_config.py`？

## 10. 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 参数类型 | `dict` 而非 dataclass | 简单，无需额外依赖 |
| 默认值 | 全部保留当前硬编码值 | 向后兼容 |
| 列名生成 | `f"{func}_{param}"` | 与现有命名一致 |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| Task 1: FEATURE_PARAMS + 趋势/动量参数化 | ✅ | `features.py` | 新增 `FEATURE_PARAMS` 字典；`add_trend_features()` / `add_momentum_features()` 支持 `params` 参数 |
| Task 2: 波动率/成交量/滞后/收益参数化 | ✅ | `features.py` | 剩余 4 个 `add_*_features()` 均支持 `params` 参数 |
| Task 3: 资金费率/OI参数化 + build_all_features | ✅ | `features.py` | `build_all_features(df, config=None)` 统一入口；`load_feature_config()` 支持 YAML/字典/None |
| Task 4: 回归测试 + 向后兼容 | ✅ | `tests/test_features.py` | 60/60 测试通过；默认配置列名与之前一致；ruff 通过 |

## 12. 审查结论

✅ 代码实现与 Spec 一致（Reverse Sync 确认：代码已领先于 Spec，无偏差）。
✅ 60/60 测试通过。
✅ ruff check 通过。
✅ 默认参数 = 原硬编码值，100% 向后兼容。
✅ FeatureRegistry（阶段 2）已预构建，支持按需计算。

## 13. 确认记录（HARD-GATE）

- **确认时间**: 2026-05-31
- **确认人**: cclt
