# multi-pair-strategy — AI 模型多币种策略

> status: done
> created: 2026-05-31
> complexity: 🟡中等

## 1. 背景与目标

`AIModelStrategy` 原本是单币种（BTC/USDT）AI 模型驱动策略。随着模型在不同币种上的泛化验证完成，需要支持同时交易多个币种，每个币种使用独立的模型和配置。

**目标:** 扩展 `AIModelStrategy` 支持多币种同时交易，每个币种独立加载模型、独立推理、独立漂移监控，复用现有训练脚本和特征工程体系。

## 2. 代码现状

### 2.1 相关入口与链路

- **策略入口**: `freqtrade/user_data/strategies/strategy_ai_model_v1.py::AIModelStrategy`
- **模型加载**: `bot_start()` → `_load_all_pair_models()` → `_load_pair_model()`
- **Pair 状态切换**: `populate_indicators()` → `_activate_pair()` / `_save_pair()`
- **多时间框架**: `informative_pairs()` → `add_higher_timeframe_features()`
- **推理入口**: `_predict_classifier()` / `_predict_sequence_model()`
- **漂移监控**: `_update_drift_monitor()`
- **信号生成**: `populate_entry_trend()` / `populate_exit_trend()`
- **模型目录**: `freqtrade/user_data/models/<PAIR_NAME>/`

### 2.2 现有实现

| 功能 | 实现 |
|------|------|
| 多模型加载 | `bot_start()` 扫描 `models/` 下所有子目录，按 pair 加载模型和配置 |
| Pair 状态管理 | `_pair_models` 字典存储每个 pair 的模型、配置、漂移基线、Scaler 参数 |
| 状态切换 | `populate_indicators()` 前 `_activate_pair()`，后 `_save_pair()` |
| 独立推理 | sklearn 分类器 / PyTorch LSTM 均按当前激活 pair 的模型执行 |
| 独立漂移 | 每个 pair 独立的 `prediction_buffer` 和 `candle_count` |
| 多时间框架 | `informative_pairs()` 为所有白名单币种返回 `(pair, "1d")` |
| 向后兼容 | `_load_legacy_model()` 支持旧版单币种模型布局 |

### 2.3 发现与风险

- **模型目录结构**: `models/BTC_USDT/sklearn_model.pkl` + `feature_config.json` + `drift_baseline.json`
- **配置已就绪**: `config_ai_model.json` 设置 `max_open_trades: 9`, `number_assets: 9`
- **10 个币种模型已训练**: BTC, ETH, SOL, BNB, ADA, AVAX, DOGE, LINK, PAXG, XRP
- **风险**: 全局 `ENTRY_THRESHOLD = 0.6` 所有币种共用，未来可按币种定制

## 3. 功能点

- [x] **多模型目录扫描**: `bot_start()` 自动扫描 `models/` 下所有 pair 子目录
- [x] **按 Pair 加载模型**: 每个子目录独立加载 sklearn / PyTorch 模型 + feature_config + drift_baseline + scaler
- [x] **Pair 状态隔离**: `_pair_models` 字典确保各 pair 模型状态互不干扰
- [x] **动态状态切换**: `populate_indicators()` 中按当前处理 pair 激活对应模型
- [x] **独立推理**: `_predict_classifier()` / `_predict_sequence_model()` 使用当前激活 pair 的模型
- [x] **独立漂移监控**: 每个 pair 独立的 PSI 计算和 Telegram 告警
- [x] **多时间框架支持**: `informative_pairs()` 为所有白名单 pair 提供 1d 数据
- [x] **向后兼容**: 无子目录时回退到 `_load_legacy_model()` 单币种模式

## 4. 业务规则

- 每个币种必须有自己的模型目录，包含 `feature_config.json`
- 模型类型（sklearn/pytorch）按 pair 独立，可混用
- 漂移监控阈值全局统一（后续可按币种定制）
- `populate_indicators()` 必须保证 `_activate_pair()` / `_save_pair()` 配对使用

## 5. 数据变更

| 操作 | 文件/配置 | 说明 |
|------|----------|------|
| 新增目录结构 | `models/<PAIR_NAME>/` | 每个币种独立模型目录 |
| 配置 | `config_ai_model.json` | `max_open_trades: 9`, `pair_whitelist` 可扩展多币种 |

## 6. 接口变更

| 操作 | 接口/函数 | 变更内容 |
|------|----------|---------|
| 新增属性 | `AIModelStrategy._pair_models` | `dict[str, dict]` 存储所有 pair 的模型状态 |
| 新增方法 | `_load_all_pair_models()` | 扫描 models/ 下所有 pair 子目录 |
| 新增方法 | `_load_pair_model(pair, pair_dir)` | 加载单个 pair 的模型+配置+基线 |
| 新增方法 | `_activate_pair(pair)` | 将指定 pair 的状态加载到 self 属性 |
| 新增方法 | `_save_pair(pair)` | 将当前 self 属性保存回 _pair_models |
| 修改 | `bot_start()` | 调用 `_load_all_pair_models()` 而非单模型加载 |
| 修改 | `populate_indicators()` | 前后插入 `_save_pair()` / `_activate_pair()` |
| 修改 | `informative_pairs()` | 返回所有白名单 pair 的 1d 数据 |

## 7. 影响范围

- **策略代码**: `strategy_ai_model_v1.py` — 核心多币种支持
- **模型部署**: `models/` 目录需要按 pair 子目录组织
- **配置文件**: `config_ai_model.json` — `pair_whitelist` 扩展多币种
- **训练脚本**: 无需修改，继续按单币种训练，输出到对应子目录即可

## 8. 风险与关注点

> ⚠️ 涉及模型推理变更 → 训练-推理一致性已由 `features.py` 保证
> ⚠️ 涉及多币种资金分配 → 由 Freqtrade `max_open_trades` + `stake_amount` 控制

- **内存占用**: N 个币种 × 模型大小，需监控大模型（LSTM）加载数量
- **推理延迟**: `populate_indicators()` 每 pair 切换状态可能引入微量开销
- **模型版本不一致**: 不同币种的模型可能基于不同特征版本，需确保 feature_config 一致

## 8.5 测试策略

- **测试范围**: 模型加载、pair 状态切换、推理隔离
- **覆盖率目标**: 核心状态管理逻辑覆盖
- **独立 Test Spec**: 否（依赖 Freqtrade 运行时框架，集成测试为主）

## 9. 待澄清

- [x] 全部已解决

## 10. 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 状态管理方式 | `_pair_models` 字典 + 激活/保存模式 | 兼容 Freqtrade 策略单实例设计，避免全局变量 |
| 模型目录结构 | `models/<PAIR>/` | 清晰、可扩展、便于版本管理 |
| 向后兼容 | `_load_legacy_model()` | 保护已有单币种部署 |
| 推理方式 | 每 pair 独立推理 | 不同币种特征分布不同，独立模型更合理 |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| 多模型目录扫描 + 加载 | ✅ | `strategy_ai_model_v1.py` | `_load_all_pair_models()` 扫描所有子目录 |
| Pair 状态隔离 + 切换 | ✅ | `strategy_ai_model_v1.py` | `_pair_models` + `_activate_pair()` / `_save_pair()` |
| 独立推理 + 漂移监控 | ✅ | `strategy_ai_model_v1.py` | 推理和漂移按当前激活 pair 执行 |
| 多时间框架 + 向后兼容 | ✅ | `strategy_ai_model_v1.py` | `informative_pairs()` + `_load_legacy_model()` |
| 模型部署 + 配置 | ✅ | `models/*_USDT/`, `config_ai_model.json` | 10 个币种模型已部署 |

## 12. 审查结论

✅ 代码实现完整，支持 10 个币种独立模型加载和推理。
✅ 向后兼容单币种模式。
✅ 状态切换逻辑正确，无交叉污染风险。
⚠️ 全局阈值 `ENTRY_THRESHOLD = 0.6` 未来可按币种定制优化。

## 13. 确认记录（HARD-GATE）

- **确认时间**: 2026-05-31
- **确认人**: cclt
