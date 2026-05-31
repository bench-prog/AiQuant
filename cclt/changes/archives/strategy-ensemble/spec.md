# strategy-ensemble — 策略加权融合

> status: done
> created: 2026-05-31
> complexity: 🔴复杂

## 1. 背景与目标

现有策略各自独立运行，各有优劣：
- **AIModelStrategy**: AI 模型驱动，高置信度时表现好，但模型漂移时失效
- **TrendFollowingStrategy**: 纯规则趋势跟踪，市场趋势明确时表现好，震荡市失效
- **SmallCapRegimeStrategy**: Regime 切换，适应性强，但小市值币种流动性差

**目标:** 创建 `StrategyEnsemble`，通过加权融合多个策略的信号，平滑收益曲线，降低单一策略失效风险。

## 2. 代码现状

### 2.1 相关策略信号

| 策略 | 核心信号 | 信号范围 | 时间框架 |
|------|---------|---------|---------|
| AIModelStrategy | `ai_prediction` | [0, 1] | 4h |
| TrendFollowingStrategy | EMA 多头排列 + ADX | {0, 1} | 4h |
| SmallCapRegimeStrategy | `regime` + 多重条件 | {0, 1} | 4h |

### 2.2 现有实现

- `AIModelStrategy.populate_entry_trend()`: `ai_prediction > ENTRY_THRESHOLD` + ADX > 20 + RSI < 75
- `TrendFollowingStrategy.populate_entry_trend()`: EMA 多头排列 + ADX > threshold + close > EMA_short
- 三个策略均使用 `features.py` 共享特征工程

### 2.3 发现与风险

- Freqtrade 不支持原生策略组合（一个 bot 只能运行一个策略）
- 需要在一个策略类中集成多个子策略的逻辑
- 子策略信号可能冲突（如 AI 看多但 Trend 看空）

## 3. 功能点

- [ ] **创建 `StrategyEnsemble` 类**: 继承 IStrategy，集成多个子策略逻辑
- [ ] **集成 AI 信号**: 复用 AIModelStrategy 的模型推理逻辑
- [ ] **集成 Trend 信号**: 复用 TrendFollowingStrategy 的 EMA/ADX 逻辑
- [ ] **信号归一化**: 将各策略输出统一映射到 [0, 1]
- [ ] **加权融合**: `ensemble_score = Σ(weight_i × signal_i)`
- [ ] **阈值入场**: `ensemble_score > threshold` → enter_long
- [ ] **权重参数化**: 支持 Hyperopt 优化权重
- [ ] **模型缺失降级**: 子策略模型缺失时，权重自动重分配
- [ ] **新增配置**: `config_ensemble.json`

## 4. 业务规则

- 各子策略信号独立计算，互不干扰
- 权重和为 1.0（归一化）
- 子策略模型缺失时，该策略权重按比例分配给其他策略
- `ensemble_score` 必须经过 sanity check（[0, 1] 范围）
- 出场信号可由任一子策略触发（OR 逻辑），或统一 ensemble_score < 阈值

## 5. 数据变更

| 操作 | 文件/配置 | 说明 |
|------|----------|------|
| 新增 | `strategy_ensemble_v1.py` | 组合策略类 |
| 新增 | `config_ensemble.json` | 权重配置 + pair_whitelist |

## 6. 接口变更

| 操作 | 接口/函数 | 变更内容 |
|------|----------|---------|
| 新增 | `StrategyEnsemble` 类 | 新策略类 |
| 新增 | `_compute_ai_signal()` | 复用 AI 模型推理 |
| 新增 | `_compute_trend_signal()` | 复用 Trend EMA/ADX 逻辑 |
| 新增 | `_compute_ensemble_score()` | 加权融合函数 |
| 新增 | `populate_entry_trend()` | 基于 ensemble_score 入场 |

## 7. 影响范围

- **新增文件**: `strategy_ensemble_v1.py`, `config_ensemble.json`
- **无修改现有策略**: 纯新增，不影响现有策略运行
- **训练脚本**: 无需修改
- **测试**: 新增 `test_ensemble.py`

## 8. 风险与关注点

> ⚠️ **涉及资金/交易逻辑变更 → 高亮提醒人工审查**

- **信号冲突**: 子策略信号可能方向相反，加权后可能产生弱信号或振荡
- **权重过拟合**: Hyperopt 优化的权重可能在样本外表现差
- **计算开销**: 多个子策略同时计算特征和推理，可能增加回测时间
- **模型一致性**: 子策略模型版本需保持一致

## 8.5 测试策略

- **测试范围**: 各子策略信号独立性、加权融合正确性、权重归一化、降级行为
- **覆盖率目标**: 80%+
- **独立 Test Spec**: 是

## 9. 待澄清

- [x] **Q1: 组合方式** — 已确认 A（信号加权融合）
- [ ] **Q2: 组合哪些策略** — AI + Trend + ?（推荐 AI + Trend，互补性最强）
- [ ] **Q3: 权重优化** — 固定权重 vs Hyperopt 动态优化

## 10. 技术决策

| 决策 | 候选方案 | 推荐 |
|------|---------|------|
| Q2: 子策略 | AI + Trend / AI + Trend + Regime / 全部 | **AI + Trend** — 互补性强，实现复杂度适中 |
| Q3: 权重 | 固定权重 / Hyperopt | **固定权重** 起步，后续可扩展 Hyperopt ⭐ |
| 信号归一化 | ai_prediction 原生 [0,1] + trend 映射 [0,1] | 已确定 |
| 出场逻辑 | ensemble_score < threshold / 任一子策略触发 | **ensemble_score < threshold**（与入场对称） |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| Task 1: StrategyEnsemble 骨架 + 信号归一化 | ✅ | `strategy_ensemble_v1.py` | StrategyEnsemble 类，集成 AI 和 Trend 信号计算逻辑 |
| Task 2: AI 信号集成 + Trend 信号集成 | ✅ | `strategy_ensemble_v1.py` | _compute_ai_signal() + _compute_trend_signal() 完整实现 |
| Task 3: 加权融合 + 配置 + 测试 | ✅ | `config_ensemble.json`, `tests/test_ensemble.py` | ensemble_score 计算、入场/出场逻辑、21 个测试用例 |

## 12. 审查结论

✅ 代码实现与 Spec 一致。
✅ 21/21 测试通过。
✅ ruff check 通过。
⚠️ 涉及资金/交易逻辑变更 — 建议人工审查后再部署实盘。

## 13. 确认记录（HARD-GATE）

- **确认时间**: 2026-05-31
- **确认人**: cclt
