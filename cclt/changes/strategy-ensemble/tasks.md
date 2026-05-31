# 任务拆分 — 策略加权融合

## 前置条件

- [x] Spec 已确认（Q1=A, Q2=A, Q3=A）
- [x] 三段全部确认

## Task 1: StrategyEnsemble 骨架 + 信号归一化

- **目标**: 创建 StrategyEnsemble 类，集成 AI 和 Trend 的信号计算逻辑
- **涉及文件**:
  - `freqtrade/user_data/strategies/strategy_ensemble_v1.py` — 新策略类
- **关键签名**:
  ```python
  class StrategyEnsemble(IStrategy):
      ENSEMBLE_WEIGHTS = {"ai": 0.60, "trend": 0.40}
      ENTRY_THRESHOLD = 0.55
      EXIT_THRESHOLD = 0.45
  ```
- **依赖**: 无
- **验收标准**:
  - 类结构正确，继承 IStrategy
  - 包含 `_compute_ai_signal()` 和 `_compute_trend_signal()` 方法骨架
  - ruff 无错误
- **验证命令**: `ruff check freqtrade/user_data/strategies/strategy_ensemble_v1.py`

## Task 2: AI 信号集成 + Trend 信号集成

- **目标**: 复用 AIModelStrategy 和 TrendFollowingStrategy 的核心逻辑
- **涉及文件**:
  - `strategy_ensemble_v1.py` — 实现 `_compute_ai_signal()` 和 `_compute_trend_signal()`
- **关键逻辑**:
  - `_compute_ai_signal()`: 加载 sklearn 模型 → 推理 → 返回 [0,1] 概率
  - `_compute_trend_signal()`: EMA 多头排列 + ADX → 映射到 [0,1]
    - EMA 多头 + ADX > 20 → 1.0
    - EMA 多头但 ADX < 20 → 0.5
    - 其他 → 0.0
- **依赖**: Task 1
- **验收标准**:
  - AI 信号与 AIModelStrategy 一致
  - Trend 信号与 TrendFollowingStrategy 一致
  - 两者均归一化到 [0, 1]

## Task 3: 加权融合 + 配置 + 测试

- **目标**: 实现 ensemble_score 计算、入场/出场逻辑、配置、测试
- **涉及文件**:
  - `strategy_ensemble_v1.py` — `_compute_ensemble_score()`, `populate_entry_trend()`, `populate_exit_trend()`
  - `freqtrade/config_ensemble.json` — 权重配置
  - `tests/test_ensemble.py` — 单元测试
- **依赖**: Task 2
- **验收标准**:
  - `ensemble_score = 0.6 × ai_signal + 0.4 × trend_signal`
  - 入场: `ensemble_score > 0.55`
  - 出场: `ensemble_score < 0.45`
  - 测试通过
  - ruff 通过
- **验证命令**:
  - `pytest tests/test_ensemble.py -v`
  - `ruff check freqtrade/user_data/strategies/strategy_ensemble_v1.py`

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 3 个文件
- **Spec-Plan 偏差记录**:
- **遗留问题**:
