# 任务拆分 — 训练脚本重构 Batch 1

> 拆分顺序：公共模块 → 训练脚本重构 → 策略适配 → 版本控制
> 每个任务 = 可独立提交的原子变更（3-5 个文件）

## 前置条件

- [x] 代码现状已 Research 完毕
- [x] Spec 已确认（HARD-GATE）

## Task 1: 创建公共训练配置模块

- **目标**: 抽取两个训练脚本中的公共常量到单一模块
- **涉及文件**:
  - `research/training_config.py` — 新增，包含 SYMBOL, TIMEFRAME, TRAIN_START, TRAIN_END, FULL_END, EXCHANGE, MODEL_OUTPUT_DIR, HORIZON 等
- **关键签名**:
  ```python
  """AiQuant 训练公共配置。"""
  from pathlib import Path

  SYMBOL: str = "BTC/USDT"
  TIMEFRAME: str = "1h"
  TRAIN_START: str = "2022-01-01"
  TRAIN_END: str = "2023-12-31"
  FULL_END: str = "2024-12-31"
  EXCHANGE: str = "binance"
  HORIZON: int = 1
  MODEL_OUTPUT_DIR: Path = Path(__file__).parent.parent / "freqtrade" / "user_data" / "models"
  ```
- **依赖**: 无
- **验收标准**: 模块可正常导入，常量值与现有脚本一致
- **验证命令**: `python -c "from research.training_config import SYMBOL; print(SYMBOL)"`
- **完成**

## Task 2: 创建公共数据合并工具

- **目标**: 抽取 `merge_external_data()` 到公共模块，两个训练脚本复用
- **涉及文件**:
  - `research/data_utils.py` — 新增，包含 `merge_external_data()` 和 `load_training_data()`
- **关键签名**:
  ```python
  def merge_external_data(
      df: pd.DataFrame,
      symbol: str,
      start: str,
      end: str,
      exchange: str,
      timeframe: str,
  ) -> pd.DataFrame:
      ...

  def load_training_data(
      symbol: str = SYMBOL,
      timeframe: str = TIMEFRAME,
      start: str = TRAIN_START,
      end: str = FULL_END,
      exchange: str = EXCHANGE,
  ) -> pd.DataFrame:
      ...
  ```
- **依赖**: Task 1
- **验收标准**: 函数行为与现有实现完全一致（ funding_rate 和 open_interest 的 try/except 降级逻辑保留）
- **验证命令**: `python -c "from research.data_utils import load_training_data; df = load_training_data(); print(len(df))"`
- **完成**

## Task 3: 重构 train_classifier.py

- **目标**: 使用公共模块，消除重复，修改配置输出文件名
- **涉及文件**:
  - `research/train_classifier.py` — 修改：删除本地常量，导入 training_config 和 data_utils，修改 feature_config 输出名
- **关键变更**:
  - 删除本地 `SYMBOL`, `TIMEFRAME`, `TRAIN_START`, `TRAIN_END`, `FULL_END`, `EXCHANGE`, `HORIZON`, `MODEL_OUTPUT_DIR`
  - 删除本地 `load_data()` 和 `merge_external_data()`
  - 从 `training_config` 导入常量
  - 从 `data_utils` 导入 `load_training_data`
  - 配置输出改为 `feature_config_lightgbm.json`
- **依赖**: Task 1, Task 2
- **验收标准**: 脚本可正常执行，输出 `sklearn_model.pkl` 和 `feature_config_lightgbm.json`
- **验证命令**: `cd research && python train_classifier.py`（dry-run 验证到模型保存）
- **完成**

## Task 4: 重构 train_sequence.py + 修复 shuffle

- **目标**: 使用公共模块 + 修复 LSTM shuffle 问题 + 修改配置输出文件名
- **涉及文件**:
  - `research/train_sequence.py` — 修改：同上，额外修复 `DataLoader(shuffle=True)` → `shuffle=False`
- **关键变更**:
  - 同 Task 3 的重构内容
  - `DataLoader(train_ds, ..., shuffle=True)` → `shuffle=False`
  - 配置输出改为 `feature_config_lstm.json`
  - 补充漂移基线导出（参考 train_classifier.py 的实现）
- **依赖**: Task 1, Task 2
- **验收标准**: 脚本可正常执行，输出 `pytorch_model.pt`、`feature_config_lstm.json` 和 `drift_baseline_lstm.json`
- **验证命令**: `cd research && python train_sequence.py`（dry-run 验证）
- **完成**

## Task 5: 策略适配新配置文件名

- **目标**: `strategy_ai_model_v1.py` 支持新的配置文件名，兼容旧名 fallback
- **涉及文件**:
  - `freqtrade/user_data/strategies/strategy_ai_model_v1.py` — 修改配置加载逻辑
- **关键变更**:
  - 加载逻辑改为：先尝试 `feature_config_{model_type}.json`，不存在则 fallback 到 `feature_config.json`
  - model_type 从配置文件中读取（"lightgbm" 或 "lstm"）
- **依赖**: Task 3, Task 4
- **验收标准**: 策略能正确加载两种模型的配置
- **验证命令**: 检查策略导入无报错
- **完成**

## Task 6: gold_pulse 文件入版本控制

- **目标**: 确认 `gold_pulse` 相关文件内容安全后纳入版本控制
- **涉及文件**:
  - `freqtrade/config_gold_pulse.json` — 确认无 API Key 后 git add
  - `freqtrade/user_data/strategies/strategy_gold_pulse_v1.py` — git add
- **关键检查点**:
  - `config_gold_pulse.json` 中 `exchange.key` 和 `exchange.secret` 是否为占位符（如 "YOUR_API_KEY"）
  - `api_server.jwt_secret_key` 是否为默认/占位值
- **依赖**: 无
- **验收标准**: 文件已 git add，无敏感信息泄漏
- **验证命令**: `git diff --cached --name-only | grep gold_pulse`
- **完成**

## 变更摘要

/apply 全部完成后填写

- **总文件数**: 5 个新增/修改文件 + 2 个版本控制文件
- **Spec-Plan 偏差记录**:
- **遗留问题**:
