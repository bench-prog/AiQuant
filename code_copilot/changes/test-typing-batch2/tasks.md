# 任务拆分 — 测试补全 + 类型注解 + 工程化 Batch 2

> 拆分顺序：测试框架 → 单元测试 → 集成测试 → 类型注解 → 工程化
> 每个任务 = 可独立提交的原子变更（3-5 个文件）

## 前置条件

- [x] Batch 1 已完成并归档
- [x] Spec 已确认（HARD-GATE）

## Task 1: 设置 pytest 测试框架

- **目标**: 创建 pytest 配置文件和测试目录结构
- **涉及文件**:
  - `pytest.ini` — 新增，pytest 配置（testpaths, pythonpath, addopts）
  - `tests/__init__.py` — 新增，空文件
  - `tests/conftest.py` — 新增，共享 fixture（合成 OHLCV DataFrame）
- **关键签名**:
  ```python
  # conftest.py
  import pytest
  import pandas as pd
  import numpy as np

  @pytest.fixture
  def sample_ohlcv() -> pd.DataFrame:
      """合成 OHLCV 数据，用于测试。"""
      dates = pd.date_range("2024-01-01", periods=100, freq="1h")
      np.random.seed(42)
      close = 100 + np.random.randn(100).cumsum()
      df = pd.DataFrame({
          "date": dates,
          "open": close + np.random.randn(100) * 0.5,
          "high": close + np.abs(np.random.randn(100)) * 2,
          "low": close - np.abs(np.random.randn(100)) * 2,
          "close": close,
          "volume": np.random.randint(1000, 10000, 100),
      })
      return df
  ```
- **依赖**: 无
- **验收标准**: `pytest --collect-only` 能发现 tests/ 目录
- **验证命令**: `pytest --collect-only`
- **完成**

## Task 2: features.py 核心指标单元测试

- **目标**: 为 features.py 的核心指标函数编写单元测试
- **涉及文件**:
  - `tests/test_features.py` — 新增
- **测试范围**:
  - `ema()` — 验证平滑效果，长度参数
  - `rsi()` — 验证范围 [0, 100]，超买超卖信号
  - `macd()` — 验证返回 3 个 Series
  - `atr()` — 验证正数，长度参数
  - `adx()` — 验证返回 3 个 Series，范围 [0, 100]
  - `bbands()` — 验证返回 3 个 Series，上下轨关系
  - `stoch()` — 验证返回 2 个 Series，范围 [0, 100]
  - `cci()` — 验证数值合理性
  - `obv()` — 验证累积特性
  - `vwap()` — 验证计算逻辑
- **依赖**: Task 1
- **验收标准**: `pytest tests/test_features.py -v` 全部通过
- **验证命令**: `pytest tests/test_features.py -v`
- **完成**

## Task 3: add_*_features 集成测试

- **目标**: 测试特征组合函数和 build_all_features
- **涉及文件**:
  - `tests/test_features.py` — 追加测试
- **测试范围**:
  - `add_trend_features()` — 验证输出列包含 ema_*, macd*
  - `add_momentum_features()` — 验证输出列包含 rsi_*, stoch_*, cci_
  - `add_volatility_features()` — 验证输出列包含 atr_*, bb_*
  - `add_volume_features()` — 验证输出列包含 volume_*, obv, vwap
  - `add_candle_features()` — 验证 0/1 值和比例值范围
  - `add_lag_features()` — 验证滞后列数量和命名
  - `add_time_features()` — 验证 hour, hour_sin, hour_cos
  - `add_return_features()` — 验证收益率计算
  - `add_funding_rate_features()` — 验证无 fundingRate 列时原样返回
  - `add_open_interest_features()` — 验证无 openInterest 列时原样返回
  - `build_all_features()` — 验证输出包含所有预期列，无 NaN/inf
  - `get_feature_columns()` — 验证正确剔除基础列
- **依赖**: Task 2
- **验收标准**: 集成测试全部通过
- **验证命令**: `pytest tests/test_features.py -v`
- **完成**

## Task 4: 补全 features.py 类型注解

- **目标**: 为 3 个缺失返回类型的函数添加注解
- **涉及文件**:
  - `freqtrade/user_data/strategies/features.py` — 修改
- **关键变更**:
  ```python
  def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:

  def bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:

  def stoch(high: pd.Series, low: pd.Series, close: pd.Series, k: int = 14, d: int = 3) -> tuple[pd.Series, pd.Series]:
  ```
- **依赖**: 无（可与测试并行）
- **验收标准**: `python -m py_compile` 通过
- **验证命令**: `python -m py_compile freqtrade/user_data/strategies/features.py`
- **完成**

## Task 5: 补全 market_data.py 类型注解

- **目标**: 为关键函数补全类型注解
- **涉及文件**:
  - `data/market_data.py` — 修改
- **关键变更**:
  - `_init_exchange()` 添加返回类型 `-> ccxt.Exchange`
  - `_fetch_paginated()` 添加参数和返回类型
  - `fetch_ohlcv_ccxt()` 补全参数类型
  - `fetch_funding_rate()` 补全参数类型
  - `fetch_open_interest()` 补全参数类型
- **依赖**: 无
- **验收标准**: `python -m py_compile` 通过
- **验证命令**: `python -m py_compile data/market_data.py`
- **完成**

## Task 6: 添加 Makefile

- **目标**: 常用命令快捷方式
- **涉及文件**:
  - `Makefile` — 新增
- **目标列表**:
  - `make test` — 运行 pytest
  - `make train-classifier` — 运行分类器训练
  - `make train-lstm` — 运行 LSTM 训练
  - `make backtest-ai` — 运行 AI 策略回测
  - `make backtest-smallcap` — 运行小市值策略回测
  - `make lint` — 运行 ruff check（如已安装）
- **依赖**: Task 1（测试需要 pytest 就绪）
- **验收标准**: `make test` 可运行
- **验证命令**: `make test`
- **完成**

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 7 个新增/修改文件
- **Spec-Plan 偏差记录**:
- **遗留问题**:
