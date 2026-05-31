# 变更日志 — features.py 增强

> 记录决策、踩坑和知识发现。知识飞轮的输入。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-23 | propose | 创建 Spec + Tasks | 发现 ADX 未使用、EMA 重复计算、6 个缺失指标 |
| 2026-05-23 | apply | 代码实现（提前完成）| 在实际编码中已完成所有功能 |
| 2026-05-31 | archive | Reverse Sync 归档 | 代码领先于 Spec，验证后归档 |

## 技术决策

| 决策 | 选择 | 放弃的方案 | 原因 |
|------|------|-----------|------|
| ADX 窗口 | 14 | 其他 | 与 ATR/RSI 一致，行业标准 |
| Williams %R 窗口 | 14 | 其他 | 与 RSI 同周期便于对比 |
| MOM 窗口 | 10 | 其他 | 覆盖约半交易日（1h 周期）|
| VWAP 偏离 | 百分比标准化 | 绝对差值 | 消除价格量级影响 |
| BB position | [0, 1] 归一化 | 原始价格差 | 与 bb_width 互补 |

## 踩坑记录

| 问题 | 原因 | 解决方案 | 沉淀？ |
|------|------|---------|--------|
| ADX 函数已定义但未生成特征列 | 函数写了但没在 `add_trend_features()` 中调用 | 补充调用并生成 `adx_14`, `plus_di_14`, `minus_di_14` | ✅ |
| EMA 在 `add_candle_features()` 中重复计算 | 内联调用 `ema(close, 12)` 而非复用已有列 | 改为 `df["ema_12"] if "ema_12" in df.columns else ema(...)` | ✅ |
| 新增特征后旧模型不兼容 | 特征列数从 46 → 52 | 标记旧模型需重新训练，保留旧配置备份 | ✅ |

## 知识发现

> 每个 task 后实时记录，/archive 时逐条确认沉淀到 knowledge/

- [x] **ADX 已定义未使用**: `adx()` 函数存在但 `add_trend_features()` 未调用 → 指标函数定义 ≠ 特征列生成，必须 double check 调用链
- [x] **EMA 重复计算**: `add_candle_features()` 内联计算 EMA 与 `add_trend_features()` 冗余 → 特征函数间存在依赖时应复用已有列
- [x] **模型兼容性**: 新增特征列会改变 `feature_config.json` 列数 → 旧 `.pkl` / `.pt` 模型无法加载，需版本管理或重新训练
- [x] **向后兼容复用模式**: `df["ema_12"] if "ema_12" in df.columns else ema(...)` 是安全的跨函数列复用模式

## Spec-Code 偏差记录

| 偏差点 | Spec 预期 | 实际情况 | 处理方式 |
|--------|----------|---------|---------|
| 代码实现时机 | Spec → Code 顺序执行 | 代码已在之前提交中实现 | Reverse Sync：验证代码符合 Spec 后更新 Spec 状态 |
| 特征注册表 | 本次不做 | 代码中已包含 `FeatureRegistry` | 属于 `features-parametrization` 变更范围，不影响本次 |

## 代码质量备忘

- 全部 60 个测试通过（含新增特征测试）
- ruff check 通过
- 所有指标函数有完整类型注解
- 新增特征均通过 `test_no_inf` 验证
