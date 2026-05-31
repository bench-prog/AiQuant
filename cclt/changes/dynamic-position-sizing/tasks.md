# 任务拆分 — 动态仓位管理

## 前置条件

- [x] Spec 已确认（Q1=C, Q2=B）
- [x] 三段全部确认

## Task 1: 新增 POSITION_SIZING_CONFIG + custom_stake_amount() 骨架

- **目标**: 在 AIModelStrategy 中添加仓位管理配置和自定义仓位方法骨架
- **涉及文件**:
  - `freqtrade/user_data/strategies/strategy_ai_model_v1.py` — 新增 `POSITION_SIZING_CONFIG` 类属性 + `custom_stake_amount()` 方法签名
- **关键签名**:
  ```python
  POSITION_SIZING_CONFIG = {
      "method": "confidence_x_volatility",
      "base_wallet_pct": 0.10,      # 总资金的 10% 作为 base_stake 池
      "min_position_pct": 0.20,     # 最低仓位 = base_stake × 0.20
      "max_position_pct": 2.00,     # 最高仓位 = base_stake × 2.00
      "confidence": {
          "threshold_low": 0.60,    # ENTRY_THRESHOLD
          "threshold_high": 0.90,   # 满仓信号
          "mapping": "linear",      # linear / exponential
      },
      "volatility": {
          "target_atr_pct": 0.02,   # 目标 ATR 百分比 = 2%
          "max_atr_pct": 0.05,      # ATR > 5% 时仓位最小
      },
  }
  ```
- **依赖**: 无
- **验收标准**:
  - 类属性定义正确
  - 方法签名符合 Freqtrade 接口
  - ruff 无错误
- **验证命令**: `ruff check freqtrade/user_data/strategies/strategy_ai_model_v1.py`

## Task 2: 实现置信度因子 + 波动率因子计算逻辑

- **目标**: 实现仓位计算公式
- **涉及文件**:
  - `strategy_ai_model_v1.py` — `custom_stake_amount()` 完整实现
- **公式**:
  ```
  base_stake = wallet_balance × base_wallet_pct / max_open_trades
  
  # 置信度因子 (linear): 0.6 → 0.0, 0.9 → 1.0
  confidence = clip((ai_prediction - threshold_low) / (threshold_high - threshold_low), 0, 1)
  
  # 波动率因子: ATR% = atr_14 / close
  # target_atr_pct (2%) → factor = 1.0
  # max_atr_pct (5%) → factor = min_position_pct
  atr_pct = atr_14 / close
  volatility = clip(target_atr_pct / atr_pct, min_position_pct, max_position_pct)
  
  final_stake = base_stake × confidence × volatility
  final_stake = clip(final_stake, base_stake × min_position_pct, base_stake × max_position_pct)
  ```
- **依赖**: Task 1
- **验收标准**:
  - 公式计算正确
  - 边界条件处理（NaN、ATR=0、prediction 缺失）
  - 返回 stake 在 [min_stake, max_stake] 范围内
- **验证命令**: `pytest tests/test_position_sizing.py -v`（需先写测试）

## Task 3: 配置更新 + 测试 + 最终验证

- **目标**: 更新 config，编写完整测试，运行回归验证
- **涉及文件**:
  - `freqtrade/config_ai_model.json` — `stake_amount` 改为 `"unlimited"`
  - `tests/test_position_sizing.py` — 新增测试文件
- **测试用例**:
  - 置信度 0.6 → confidence_factor = 0.0 → 最小仓位
  - 置信度 0.9 → confidence_factor = 1.0 → 满仓（受波动率限制）
  - ATR 2% → volatility_factor = 1.0
  - ATR 5% → volatility_factor = 0.2（最小）
  - NaN/缺失 → 回退到 base_stake
  - 边界: final_stake 不超过 max_stake
- **依赖**: Task 2
- **验收标准**:
  - 新测试全部通过
  - 现有 60 个测试全部通过
  - ruff check 通过
- **验证命令**:
  - `pytest tests/test_position_sizing.py -v`
  - `pytest tests/test_features.py -v`
  - `ruff check freqtrade/user_data/strategies/strategy_ai_model_v1.py`

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 3 个文件
- **Spec-Plan 偏差记录**:
- **遗留问题**:
