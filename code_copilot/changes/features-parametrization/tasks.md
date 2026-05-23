# 任务拆分 — features.py 参数化重构

> 渐进路径：阶段 1（参数化配置）→ 阶段 2（特征注册表）→ 阶段 3（配置驱动流水线）

## 阶段 1: 参数化配置（本次执行）

### 前置条件

- [x] Spec 已确认

### Task 1: 创建 FEATURE_PARAMS + 趋势/动量参数化

- **目标**: 创建集中配置字典，改造趋势和动量特征函数
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — 新增 `FEATURE_PARAMS`；改造 `add_trend_features()` / `add_momentum_features()`
- **关键变更**:
  ```python
  FEATURE_PARAMS = {
      "ema_lengths": [12, 26, 50],
      "macd": {"fast": 12, "slow": 26, "signal": 9},
      "adx_length": 14,
      "rsi_lengths": [14, 6],
      "stoch": {"k": 14, "d": 3},
      "cci_length": 20,
      "williams_r_length": 14,
      "mom_length": 10,
  }
  ```
- **依赖**: 无
- **验收标准**:
  - 默认参数下输出列名与之前完全一致
  - 自定义参数能生成对应列
  - 测试通过
- **验证命令**: `pytest tests/test_features.py -v`

### Task 2: 波动率/成交量/滞后/收益参数化

- **目标**: 改造剩余 add_* 函数
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — 改造 `add_volatility_features()` / `add_volume_features()` / `add_lag_features()` / `add_return_features()`
- **依赖**: Task 1
- **验收标准**:
  - 所有 add_* 函数支持 params 参数
  - 默认参数下输出不变
  - 测试通过

### Task 3: 资金费率/持仓量参数化 + build_all_features 改造

- **目标**: 改造最后两个函数 + 统一入口
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — 改造 `add_funding_rate_features()` / `add_open_interest_features()` / `build_all_features(df, config=None)`
- **依赖**: Task 2
- **验收标准**:
  - `build_all_features(df)` 默认行为不变
  - `build_all_features(df, custom_config)` 使用自定义配置
  - 测试通过

### Task 4: 回归测试 + 向后兼容验证

- **目标**: 验证默认配置输出与修改前完全一致
- **涉及文件**:
  - `tests/test_features.py` — 新增回归测试（对比默认配置输出列名）
- **依赖**: Task 3
- **验收标准**:
  - 默认配置生成的列名集合与之前完全相同
  - 46 个原有测试全部通过
  - 新增参数化测试通过
  - ruff check 通过
- **验证命令**:
  - `pytest tests/test_features.py -v`
  - `ruff check freqtrade/user_data/strategies/features.py`

## 阶段 2: 特征注册表（后续，本次不做）

- FeatureRegistry 类
- 按需计算
- 特征元数据管理

## 阶段 3: 配置驱动流水线（远期，本次不做）

- YAML/JSON 配置
- 多特征集 A/B 实验

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 2 个文件
- **Spec-Plan 偏差记录**:
- **遗留问题**:
