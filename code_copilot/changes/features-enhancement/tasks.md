# 任务拆分 — features.py 增强

> 拆分顺序：修复 bug → 新增特征 → 类型注解 → 验证

## 前置条件

- [x] Spec 已确认（Q1=B, Q2=A, Q3=A）

## Task 1: 修复 ADX 未使用 + EMA 重复计算

- **目标**: 修复两个已知 bug
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — `add_trend_features()` 加入 ADX 调用；`add_candle_features()` 复用已有 ema 列
  - `tests/test_features.py` — `TestAddTrendFeatures` 补充 ADX 列断言
- **关键签名**: 无变更
- **依赖**: 无
- **验收标准**:
  - `build_all_features()` 输出包含 `adx_14`, `plus_di_14`, `minus_di_14`
  - `add_candle_features()` 不再重复计算 ema
  - 测试通过
- **验证命令**: `pytest tests/test_features.py -v`

## Task 2: 新增动量特征（Williams %R + MOM）

- **目标**: 补充 Williams %R 和价格动量指标
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — 新增 `williams_r()` 函数；`add_momentum_features()` 加入 `williams_r_14`, `mom_10`
  - `tests/test_features.py` — 补充 `TestWilliamsR` + `TestMOM` + `TestAddMomentumFeatures` 列断言
- **关键签名**:
  ```python
  def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
      ...
  ```
- **依赖**: Task 1
- **验收标准**:
  - `williams_r` 返回值 ∈ [-100, 0]
  - `mom_10` 为价格差值
  - 测试通过
- **验证命令**: `pytest tests/test_features.py::TestWilliamsR -v`

## Task 3: 新增成交量 + 波动率特征

- **目标**: 补充 OBV 变化率、VWAP 偏离度、布林带位置
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — `add_volume_features()` 加入 `obv_change_1h`, `vwap_distance`；`add_volatility_features()` 加入 `bb_position`
  - `tests/test_features.py` — 补充对应测试
- **关键签名**: 无新函数
- **依赖**: Task 2
- **验收标准**:
  - `obv_change_1h` = OBV.diff(1)
  - `vwap_distance` = (close - vwap) / vwap
  - `bb_position` = (close - bb_lower) / (bb_upper - bb_lower) ∈ [0, 1]
  - 测试通过
- **验证命令**: `pytest tests/test_features.py -v`

## Task 4: 类型注解补充 + 最终验证

- **目标**: 补充所有指标函数的返回值类型注解，运行完整测试套件
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — `adx()`, `bbands()`, `stoch()`, `cci()`, `obv()`, `vwap()` 补充返回值类型
- **依赖**: Task 3
- **验收标准**:
  - 所有函数有完整的参数 + 返回值类型注解
  - `pytest tests/test_features.py` 全部通过
  - `ruff check` 无错误
- **验证命令**:
  - `pytest tests/test_features.py -v`
  - `ruff check freqtrade/user_data/strategies/features.py`

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 2 个文件（features.py + test_features.py）
- **Spec-Plan 偏差记录**:
- **遗留问题**: feature_config.json 需重新训练后更新（Q3=A，留到后续）
