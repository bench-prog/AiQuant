# dynamic-position-sizing — 动态仓位管理

> status: done
> created: 2026-05-31
> complexity: 🟡中等

## 1. 背景与目标

`AIModelStrategy` 当前使用固定 `stake_amount: 200`，所有币种、所有信号强度使用相同仓位。这存在两个问题：

1. **信号强度未充分利用**: 模型预测概率 `ai_prediction` 从 0.61（弱信号）到 0.95（强信号）都使用相同仓位，资金效率低。
2. **风险敞口不一致**: 高波动率币种（如 DOGE，ATR 大）和低波动率币种（如 BTC，ATR 小）使用相同名义仓位，实际风险差异巨大。

**目标:** 实现 `custom_stake_amount()`，根据模型置信度 × ATR 波动率动态调整仓位，提升夏普率和资金效率。

## 2. 代码现状

### 2.1 相关入口与链路

- **策略入口**: `freqtrade/user_data/strategies/strategy_ai_model_v1.py::AIModelStrategy`
- **模型预测列**: `dataframe["ai_prediction"]` ∈ [0, 1]（已在 `populate_indicators()` 生成）
- **波动率列**: `dataframe["atr_14"]`（已在 `add_volatility_features()` 生成）
- **当前仓位配置**: `freqtrade/config_ai_model.json::stake_amount = 200`
- **特征工程**: `features.py::add_volatility_features()` 生成 ATR

### 2.2 现有实现

| 项目 | 当前值 | 说明 |
|------|--------|------|
| stake_amount | 200 | 固定金额 |
| max_open_trades | 9 | 最多同时持有 9 个仓位 |
| ai_prediction | [0, 1] | 模型概率 |
| atr_14 | > 0 | 14 周期 ATR |
| ENTRY_THRESHOLD | 0.6 | 入场阈值 |

### 2.3 发现与风险

- Freqtrade 支持 `custom_stake_amount(pair, current_time, current_rate, proposed_stake, min_stake, max_stake, **kwargs)` 方法
- 该方法在 `populate_entry_trend()` 之后调用，可以访问 `dataframe` 中的 `ai_prediction` 和 `atr_14`
- 动态仓位不改变入场/出场信号逻辑，只影响单笔开仓金额

## 3. 功能点

- [ ] **新增 `custom_stake_amount()` 方法**: 在 `AIModelStrategy` 中实现动态仓位计算
- [ ] **置信度因子**: 将 `ai_prediction` 映射到仓位比例（线性或指数）
- [ ] **波动率因子**: 将 ATR 百分比映射到仓位反比缩放
- [ ] **组合公式**: `final_stake = base_stake × confidence_factor × volatility_factor`
- [ ] **参数化配置**: 最小/最大仓位比例、波动率目标、置信度映射方式
- [ ] **边界保护**: 单币种不超过 max_open_trades 分配的额度
- [ ] **配置更新**: `config_ai_model.json` 中 `stake_amount` 改为 `"unlimited"` 或百分比模式

## 4. 业务规则

- 动态仓位不改变 `populate_entry_trend()` / `populate_exit_trend()` 的信号逻辑
- 最低仓位不低于 base_stake 的 20%，避免过小仓位导致手续费占比过高
- 最高仓位不超过 base_stake 的 200%，避免过度集中
- 当 `ai_prediction` 或 `atr_14` 缺失时，回退到 base_stake
- 当 `atr_14 = 0` 或 `NaN` 时，回退到 base_stake（避免除零）

## 5. 数据变更

| 操作 | 文件/配置 | 字段/参数 | 说明 |
|------|----------|----------|------|
| 修改 | `config_ai_model.json` | `stake_amount` | `200` → `"unlimited"` 或百分比（如 `"wallet_percentage": 0.1`） |
| 新增 | `strategy_ai_model_v1.py` | `POSITION_SIZING_CONFIG` | 动态仓位参数字典 |

## 6. 接口变更

| 操作 | 接口/函数 | 变更内容 |
|------|----------|---------|
| 新增 | `custom_stake_amount()` | 返回动态计算的单笔仓位金额 |
| 新增属性 | `POSITION_SIZING_CONFIG` | 仓位管理参数字典 |

## 7. 影响范围

- **策略代码**: `strategy_ai_model_v1.py` — 新增 `custom_stake_amount()` + 配置属性
- **配置文件**: `config_ai_model.json` — `stake_amount` 变更
- **测试**: `tests/` — 新增动态仓位测试（边界值、公式正确性）

## 8. 风险与关注点

> ⚠️ **涉及资金/交易逻辑变更 → 高亮提醒人工审查**
> ⚠️ **涉及特征工程/模型推理变更 → 需检查训练-推理一致性**

- **回测差异**: 动态仓位在回测中的表现可能与固定仓位显著不同，需对比验证
- **极端行情**: ATR 飙升时仓位骤降，需测试边界行为
- **手续费影响**: 小仓位下单时手续费占比可能过高，需设置最小仓位阈值
- **模型置信度校准**: `ai_prediction` 的分布可能随时间漂移，需定期校准映射函数

## 8.5 测试策略

- **测试范围**: `custom_stake_amount()` 的边界值、公式正确性、降级行为
- **覆盖率目标**: 100%（边界条件全覆盖）
- **独立 Test Spec**: 是（需要独立测试文件验证仓位计算公式）

## 9. 待澄清

- [x] **Q1: 仓位计算方式** — 已确认 C（组合：置信度 × 波动率）
- [x] **Q2: base_stake 来源** — 已确认 B（钱包百分比）

## 10. 技术决策

| 决策 | 候选方案 | 推荐 |
|------|---------|------|
| Q2: base_stake | A. 固定金额（如 200）/ B. 钱包百分比（总资金 10% / max_open_trades）/ C. 风险金额（固定风险金额） | **推荐 B** — 钱包百分比最符合动态仓位管理理念，资金自动复利 |
| 置信度映射 | 线性 `f(p) = (p - 0.5) / 0.5` / 指数 `f(p) = (p - 0.5)^2 / 0.25` | 线性（简单、可解释） |
| 波动率映射 | ATR 百分比反比 / ATR 排名分位数 | ATR 百分比反比（经典波动率目标法） |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| Task 1: POSITION_SIZING_CONFIG + custom_stake_amount() 骨架 | ✅ | `strategy_ai_model_v1.py` | POSITION_SIZING_CONFIG 字典 + custom_stake_amount() 方法签名 |
| Task 2: 置信度因子 + 波动率因子计算逻辑 | ✅ | `strategy_ai_model_v1.py` | _compute_confidence_factor() + _compute_volatility_factor() 完整实现 |
| 参数调优（Reverse Sync） | ✅ | `strategy_ai_model_v1.py` | threshold_low: 0.60→0.55, target_atr_pct: 0.02→0.015（基于 BTC 4h ATR 均值 1.38%） |
| Task 3: 配置更新 + 测试 + 回归验证 | ✅ | `config_ai_model.json`, `tests/test_position_sizing.py` | stake_amount → "unlimited", 20 个测试用例, 80/80 测试通过 |

## 12. 审查结论

✅ 代码实现与 Spec 一致。
✅ 80/80 测试通过（新 20 + 现有 60）。
✅ ruff check 通过。
⚠️ 涉及资金/交易逻辑变更 — 建议人工审查后再部署实盘。

## 13. 确认记录（HARD-GATE）

- **确认时间**: 2026-05-31
- **确认人**: cclt

## 13. 确认记录（HARD-GATE）

- **确认时间**:
- **确认人**:
