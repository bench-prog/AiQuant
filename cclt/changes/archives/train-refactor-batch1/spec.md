# 训练脚本重构 — Batch 1（核心 P0 修复）

> status: propose
> created: 2026-05-23
> complexity: 🟡中等

## 1. 背景与目标

当前 `research/` 下的两个训练脚本（`train_classifier.py` / `train_sequence.py`）存在严重的代码重复和配置分散问题，且 `feature_config.json` 存在命名冲突。本批次修复这些 P0 级别问题，为后续批次（测试补全、工程化）打好基础。

目标：
- 消除训练脚本间的代码重复（DRY）
- 统一训练配置到单一来源
- 修复 LSTM 训练的数据泄漏风险
- 解决模型配置文件相互覆盖问题
- 将 `gold_pulse` 相关未跟踪文件纳入版本控制

## 2. 代码现状（Research Findings）

### 2.1 相关入口与链路

- `research/train_classifier.py` → `data/market_data.py::fetch_ohlcv_ccxt`
- `research/train_sequence.py` → `data/market_data.py::fetch_ohlcv_ccxt`
- 两者都 → `data/service.py::query/merge_into`
- 两者都 → `features.py::build_all_features/get_feature_columns`

### 2.2 现有实现

**重复代码（两处完全相同的函数）：**
- `train_classifier.py:73-96` — `merge_external_data()` 函数
- `train_sequence.py:76-99` — `merge_external_data()` 函数，逐行一致

**分散配置：**
- `train_classifier.py:49-54` — SYMBOL="BTC/USDT", TIMEFRAME="1h", TRAIN_START/END, EXCHANGE
- `train_sequence.py:49-60` — TRAIN_START/END, LOOKBACK, HORIZON, BATCH_SIZE 等，"BTC/USDT" 直接硬编码在 `load_data()` 中

**命名冲突：**
- `train_classifier.py:233` 输出 `feature_config.json`
- `train_sequence.py:254` 输出同名 `feature_config.json`
- 策略 `strategy_ai_model_v1.py` 加载时无法区分是哪个模型的配置

**数据泄漏：**
- `train_sequence.py:176` — `DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)`
- 时序数据的训练集不应 shuffle，这会破坏时间依赖关系

**未跟踪文件：**
- `freqtrade/config_gold_pulse.json` — 新增配置
- `freqtrade/user_data/strategies/strategy_gold_pulse_v1.py` — 新增策略

### 2.3 发现与风险

| 风险 | 影响 | 当前状态 |
|------|------|----------|
| 代码重复 | 修改外部数据合并逻辑需改两处，容易遗漏 | 🔴 高 |
| 配置分散 | 改训练时间范围需改多个文件 | 🟡 中 |
| feature_config 覆盖 | 先跑分类器后跑 LSTM，策略加载的是 LSTM 配置 | 🔴 高 |
| LSTM shuffle | 训练时 shuffle 破坏时序，降低模型泛化能力 | 🔴 高 |
| 未跟踪文件 | 新策略代码未入版本控制，存在丢失风险 | 🟡 中 |

## 3. 功能点

- [ ] 功能 1：创建 `research/training_config.py` 统一存放公共训练参数
- [ ] 功能 2：创建 `research/data_utils.py` 抽取公共 `merge_external_data()`
- [ ] 功能 3：重构 `train_classifier.py` 使用新公共模块
- [ ] 功能 4：重构 `train_sequence.py` 使用新公共模块 + 修复 shuffle
- [ ] 功能 5：修改模型配置输出文件名，避免覆盖（`feature_config.json` → `feature_config_lightgbm.json` / `feature_config_lstm.json`）
- [ ] 功能 6：策略 `strategy_ai_model_v1.py` 适配新的配置文件名加载逻辑
- [ ] 功能 7：将 `gold_pulse` 相关未跟踪文件纳入版本控制（确认内容后 `git add`）

## 4. 业务规则

- 训练脚本重构后，运行命令不变（`cd research && python train_classifier.py`）
- 公共模块的导出接口必须兼容两个训练脚本的所有现有用法
- 配置文件名变更后，策略加载逻辑优先尝试新文件名，兼容旧文件名（fallback）

## 5. 数据变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `research/training_config.py` | 公共训练配置 |
| 新增 | `research/data_utils.py` | 公共数据合并工具 |
| 修改 | `research/train_classifier.py` | 使用公共模块，修改配置输出名 |
| 修改 | `research/train_sequence.py` | 使用公共模块，修复 shuffle，修改配置输出名 |
| 修改 | `freqtrade/user_data/strategies/strategy_ai_model_v1.py` | 适配新配置文件名 |
| 新增（版本控制） | `freqtrade/config_gold_pulse.json` | git add |
| 新增（版本控制） | `freqtrade/user_data/strategies/strategy_gold_pulse_v1.py` | git add |

## 6. 接口变更

| 操作 | 接口 | 变更内容 |
|------|------|---------|
| 新增 | `research.data_utils.merge_external_data()` | 从两个脚本中抽取的公共函数 |
| 新增 | `research.training_config` | 常量模块：SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END 等 |

## 7. 影响范围

- `research/` 下所有训练脚本
- `freqtrade/user_data/strategies/strategy_ai_model_v1.py`
- `freqtrade/config_gold_pulse.json`（仅版本控制）
- `freqtrade/user_data/strategies/strategy_gold_pulse_v1.py`（仅版本控制）

## 8. 风险与关注点

> ⚠️ `strategy_ai_model_v1.py` 修改配置加载逻辑时，需确保旧版 `feature_config.json` 仍能兼容（fallback 机制），避免破坏现有部署

## 8.5 测试策略

- **测试范围**：训练脚本运行流程、策略配置加载
- **覆盖率目标**：手工验证训练脚本可正常执行至完成
- **独立 Test Spec**：否（本批次以重构为主，测试在 Batch 2 补全）

## 9. 待澄清

- [x] Q1: 分批次处理 — 已确认 Batch 1 处理 P0 核心问题
- [x] Q2: gold_pulse 纳入 — 已确认
- [x] Q3: VWAP 保持现状 — 已确认

## 10. 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 配置集中方式 | `training_config.py`（Python 模块） | 简单，无需引入 YAML/JSON 解析依赖 |
| feature_config 命名 | `feature_config_{model_type}.json` | 清晰区分模型类型，避免覆盖 |
| 旧配置兼容 | 策略加载优先新名，fallback 旧名 | 避免破坏现有模型文件布局 |
| gold_pulse 处理 | 先确认内容后 git add | 确保不包含敏感信息（API Key 等） |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| 1 | ✅ | `research/training_config.py` | 新增，包含所有公共常量 |
| 2 | ✅ | `research/data_utils.py` | 新增，load_training_data + merge_external_data |
| 3 | ✅ | `research/train_classifier.py` | 使用公共模块，-63 +18 行 |
| 4 | ✅ | `research/train_sequence.py` | 使用公共模块，修复 shuffle，+53 -67 行 |
| 5 | ✅ | `strategy_ai_model_v1.py` | 适配多配置名加载 + 漂移基线 fallback |
| 6 | ✅ | `config_gold_pulse.json` + `strategy_gold_pulse_v1.py` | 入版本控制 |

## 12. 审查结论

（/review 后填写）

## 13. 确认记录（HARD-GATE）

- **确认时间**：2026-05-23
- **确认人**：用户确认
