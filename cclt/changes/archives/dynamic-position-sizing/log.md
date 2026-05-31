# 变更日志 — dynamic-position-sizing

> 记录动态仓位管理策略的开发决策和知识发现。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-31 | propose | 创建 Spec + Tasks | 用户选择 C（组合：置信度 × 波动率）+ B（钱包百分比） |
| 2026-05-31 | apply | Task 1-3 全部执行 | 代码实现 + 20 个测试用例 + 回归验证 |
| 2026-05-31 | archive | 归档 | 验证后归档 |

## 技术决策

| 决策 | 选择 | 放弃的方案 | 原因 |
|------|------|-----------|------|
| 仓位计算方式 | 置信度 × 波动率 | 纯置信度 / 纯波动率 / 凯利公式 | 组合方式最全面，兼顾信号质量和风险控制 |
| base_stake 来源 | 钱包百分比 | 固定金额 / 风险金额 | 资金自动复利，最符合量化最佳实践 |
| 置信度映射 | 线性 | 指数 | 简单、可解释 |
| 波动率映射 | ATR 百分比反比 | ATR 排名分位数 | 经典波动率目标法 |

## 踩坑记录

| 问题 | 原因 | 解决方案 | 沉淀？ |
|------|------|---------|--------|
| ruff 误解析 JSON | ruff check 对 .json 文件报 undefined name | 只 ruff check .py 文件 | ✅ |
| 测试导入 Freqtrade | IStrategy 依赖 Freqtrade 运行时 | 纯函数提取，不导入策略类 | ✅ |

## 知识发现

- [x] **动态仓位公式**: `final = base × confidence × volatility`，base 用钱包百分比，confidence 用线性映射，volatility 用 ATR 反比 → `strategy_ai_model_v1.py::custom_stake_amount`
- [x] **边界保护设计**: 最小仓位（20%）防止手续费占比过高，最大仓位（200%）防止过度集中 → `strategy_ai_model_v1.py::POSITION_SIZING_CONFIG`
- [x] **降级策略**: 特征缺失时回退到 base_stake，避免策略崩溃 → `strategy_ai_model_v1.py::_compute_confidence_factor`

## Spec-Code 偏差记录

| 偏差点 | Spec 预期 | 实际情况 | 处理方式 |
|--------|----------|---------|---------|
| Task 1+2 合并 | 分两个 Task | 骨架和逻辑在一次编辑中完成 | 无偏差，提前完成 |

## 代码质量备忘

- 80/80 测试通过（新 20 + 现有 60）
- ruff check 通过
- ⚠️ 涉及资金逻辑变更，部署前需人工审查
