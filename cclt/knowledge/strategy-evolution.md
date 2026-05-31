# 策略演进跟踪

> 记录 AiQuant 所有策略的创建、迭代和废弃过程。每次策略相关的 cclt 变更归档时同步更新本文件。
> 格式：`YYYY-MM-DD: 变更摘要 → [[change-name]]`

---

## 策略总览

| # | 策略类名 | 文件 | 版本 | 类型 | 启动日期 | 状态 |
|---|---------|------|------|------|---------|------|
| 1 | `AIModelStrategy` | `strategy_ai_model_v1.py` | v1 | AI 模型驱动 | 2025-Q1 | 🟢 活跃 |
| 2 | `SmallCapEventDrivenStrategy` | `strategy_smallcap_v1_event_driven.py` | v1 | 事件驱动 | 2025-Q2 | 🟡 维护 |
| 3 | `SmallCapTurtleStrategy` | `strategy_smallcap_v2_turtle.py` | v2 | 趋势跟踪 | 2025-Q2 | 🟡 维护 |
| 4 | `SmallCapRegimeStrategy` | `strategy_smallcap_v3_regime.py` | v3 | Regime Switching | 2025-Q3 | 🟢 活跃 |
| 5 | `GoldPulseStrategy` | `strategy_gold_pulse_v1.py` | v1 | 跨品种传导 | 2025-Q4 | 🟡 实验 |
| 6 | `TrendFollowingStrategy` | `strategy_trend_following_v1.py` | v1 | 纯规则趋势 | 2026-Q1 | 🟢 活跃 |
| 7 | `AIModelRankerStrategy` | `strategy_ai_ranker_v1.py` | v1 | 截面排序 | 2026-Q1 | 🟢 活跃 |

### 策略分类

```
AI 模型驱动:
  ├── AIModelStrategy (v1)         ← 加载 sklearn/LSTM 模型，单币种分类
  └── AIModelRankerStrategy (v1)   ← 截面排序回归，多币种选 Top 3

规则驱动:
  ├── TrendFollowingStrategy (v1)  ← EMA 多头排列 + ADX 过滤 + ATR 止损
  ├── SmallCapTurtleStrategy (v2)  ← Turtle 法则适配小市值币种
  ├── SmallCapRegimeStrategy (v3)  ← 市场状态切换 + Hyperopt 优化
  └── GoldPulseStrategy (v1)       ← 黄金→原油脉冲传导

事件驱动:
  └── SmallCapEventDrivenStrategy (v1) ← 小市值币种事件驱动
```

### 配置文件映射

| 配置 | 对应策略 |
|------|---------|
| `config_ai_model.json` | `AIModelStrategy`, `AIModelRankerStrategy` |
| `config_smallcap.json` | `SmallCapRegimeStrategy`, `SmallCapTurtleStrategy`, `SmallCapEventDrivenStrategy` |
| `config_gold_pulse.json` | `GoldPulseStrategy` |

---

## 演进时间线

### 第一阶段：基础设施 (2025-Q1 ~ 2025-Q2)

```
2025-01  ▶ 项目初始化
         ├── Freqtrade Docker 环境搭建
         ├── features.py 共享特征模块（纯 pandas/numpy）
         └── data/market_data.py CCXT 数据下载器

2025-02  ▶ AIModelStrategy 初版
         ├── 加载 LightGBM 模型，单币种分类信号
         ├── 训练脚本 train_classifier.py
         └── 回测框架就绪

2025-03  ▶ LSTM 模型支持
         ├── train_sequence.py → CryptoLSTM
         ├── StandardScaler 导出/推理
         └── 模型配置命名约定 (feature_config_{model_type}.json)
```

### 第二阶段：策略扩展 (2025-Q2 ~ 2025-Q4)

```
2025-04  ▶ 小市值策略三件套
         ├── SmallCapEventDrivenStrategy (v1) — 事件驱动
         ├── SmallCapTurtleStrategy (v2) — Turtle 趋势
         └── SmallCapRegimeStrategy (v3) — Regime Switching + Hyperopt

2025-06  ▶ 外部数据层统一
         ├── data/service.py 注册表模式
         ├── funding_rate / open_interest 特征
         └── 特征总数: 36 → 46

2025-09  ▶ 黄金脉冲策略
         └── GoldPulseStrategy (v1) — XAU→CL 跨品种传导
```

### 第三阶段：质量工程 (2025-Q4 ~ 2026-Q1)

```
2025-10  ▶ 训练脚本重构
         ├── [[train-refactor-batch1]]: 提取 training_config.py + data_utils.py
         ├── 修复 LSTM shuffle=True 数据泄漏
         └── 知识沉淀: 时序模型 DataLoader 规则

2025-11  ▶ 测试 + 类型
         ├── [[test-typing-batch2]]: pytest 39 用例 + 类型注解补全
         └── Makefile 命令快捷方式

2025-12  ▶ cclt 框架 v1
         ├── [[code-copilot]]: 渐进式 Spec 协作流程
         └── 规则文件 (rules/) + 知识索引 (knowledge/)
```

### 第四阶段：多币种 + 优化 (2026-Q1)

```
2026-01  ▶ 多币种 AI 策略
         ├── [[multi-pair-strategy]]: 10 个币种独立模型 (BTC/ETH/SOL/BNB/ADA/AVAX/DOGE/LINK/PAXG/XRP)
         ├── 策略币种路由改造 (_pair_models + _activate_pair/_save_pair)
         └── 回测优化: 目标阈值 + 正则化 + 特征筛选

2026-02  ▶ 策略参数优化
         ├── [[optimize]]: 止损收紧 + ROI 调整 + ADX 趋势过滤
         └── Optuna 超参搜索

2026-03  ▶ 模型 v2 + 截面排序
         ├── [[model-v2]]: 修复未来信息泄漏
         ├── AIModelRankerStrategy 截面排序模型
         └── Optuna 超参搜索
```

### 第五阶段：新策略 + cclt v2 (2026-Q1 ~ 现在)

```
2026-04  ▶ 新策略
         ├── [[funding-arbitrage]]: 资金费率套利 + 趋势跟踪
         └── TrendFollowingStrategy (v1): 纯规则趋势跟踪

2026-05  ▶ cclt Skill 驱动 (v2)
         ├── [[cclt-skill]]: Skill 驱动 + SessionStart/Stop hooks
         ├── [[features-enhancement]]: 特征增强 ✅ 已归档 (ADX修复+6新指标+类型注解)
         └── [[features-parametrization]]: 参数化重构 ✅ 已归档 (FEATURE_PARAMS+FeatureRegistry+YAML配置)

2026-05  ▶ 动态仓位管理
         ├── [[fix-critical-review]]: Stage 2 审查，15 个 Critical 问题修复
         └── [[dynamic-position-sizing]]: custom_stake_amount() 置信度×波动率动态仓位
```

---

## 特征演进

| 阶段 | 特征数 | 新增 | 触发变更 |
|------|--------|------|---------|
| 初始 | ~36 | 趋势/动量/波动/成交量/K线/滞后/时间/收益 | 项目初始化 |
| +资金费率 | ~40 | funding_rate, ema_8, sign, change | 外部数据层统一 |
| +持仓量 | 46 | oi, oi_ema_12/24, oi_change_1h/6h/24h, oi_velocity | 外部数据层统一 |
| +ADX/动量/波动 | 52 (待定) | adx_14, plus_di_14, minus_di_14, williams_r_14, mom_10, obv_change_1h, vwap_distance, bb_position | [[features-enhancement]] |

---

## 模型演进

| 模型 | 类型 | 输入特征数 | 预测目标 | 状态 |
|------|------|-----------|---------|------|
| LightGBM 分类器 | 分类 | 46 | 未来 1h 涨/跌 | 备选 |
| LSTM (CryptoLSTM) | 序列分类 | 46 × 20 lookback | 未来 1h 涨/跌 | 🟢 主模型 |
| Ranker (回归) | 回归 | 46 | 未来收益率 | 🟢 活跃 |

---

## 路线图

### 短期（当前活跃）

- [ ] **特征增强** → `[[features-enhancement]]`: 修复 ADX + 新增 6 个特征
- [ ] **参数化重构** → `[[features-parametrization]]`: FEATURE_PARAMS 抽离

### 中期（规划中）

- [ ] 资金费率套利策略完善 ([[funding-arbitrage]])
- [ ] 多时间框架融合（1h + 4h 信号共振）
- [ ] 动态仓位管理（Kelly Criterion / 风险平价）
- [ ] 模型定期自动重训练流水线

### 长期（探索）

- [ ] 链上数据集成（Glassnode/Dune API）
- [ ] RL 仓位管理模块
- [ ] 多交易所套利策略（可能需要 Hummingbot）
- [ ] Feature Store + 在线/离线一致性校验

---

## 废弃策略

| 策略 | 废弃日期 | 原因 | 替代方案 |
|------|---------|------|---------|
| (暂无) | — | — | — |

---

## 变更记录

> 每次策略相关的 cclt 变更归档时在此追加一行

| 日期 | 变更 | 影响策略 | cclt 变更 |
|------|------|---------|----------|
| 2026-05-23 | 特征增强 Spec 创建 | 全部 AI 策略 | `features-enhancement` |
| 2026-05-23 | 参数化重构 Spec 创建 | 全部策略 | `features-parametrization` |
| 2026-04 | 资金费率套利 + 趋势跟踪 | TrendFollowingStrategy (新增) | `funding-arbitrage` |
| 2026-03 | 截面排序模型 + 信息泄漏修复 | AIModelRankerStrategy (新增) | `model-v2` |
| 2026-02 | 策略参数优化 | SmallCapRegimeStrategy | `optimize` |
| 2026-01 | 多币种模型 + 路由 | AIModelStrategy | `multi-pair-strategy` |
| 2025-11 | 测试框架 + 类型注解 | 全部策略 | `test-typing-batch2` |
| 2025-10 | 训练脚本重构 | AI 策略 | `train-refactor-batch1` |
