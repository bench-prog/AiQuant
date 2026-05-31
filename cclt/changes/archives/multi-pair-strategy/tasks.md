# 任务拆分 — AI 模型多币种策略

> 代码已提前实现，本文件记录实际完成的 Task 作为归档依据。

## 前置条件

- [x] Spec 已确认（Reverse Sync）

## Task 1: 多模型目录扫描 + 加载

- **目标**: 支持从 `models/<PAIR>/` 子目录按 pair 加载模型
- **涉及文件**:
  - `freqtrade/user_data/strategies/strategy_ai_model_v1.py` — 新增 `_load_all_pair_models()`、`_load_pair_model()`
- **关键签名**:
  ```python
  def _load_all_pair_models(self) -> None:
  def _load_pair_model(self, pair: str, pair_dir: Path) -> None:
  ```
- **依赖**: 无
- **验收标准**:
  - `bot_start()` 扫描 `models/` 下所有子目录
  - 每个子目录的模型、配置、漂移基线正确加载
  - 向后兼容：无子目录时回退到 `_load_legacy_model()`
- **验证**: 源码检查 + 模型目录结构确认

## Task 2: Pair 状态隔离 + 动态切换

- **目标**: 确保各 pair 模型状态互不干扰
- **涉及文件**:
  - `strategy_ai_model_v1.py` — 新增 `_pair_models`、`_activate_pair()`、`_save_pair()`
- **关键签名**:
  ```python
  _pair_models: dict = {}
  def _activate_pair(self, pair: str) -> None:
  def _save_pair(self, pair: str) -> None:
  ```
- **依赖**: Task 1
- **验收标准**:
  - `populate_indicators()` 前激活当前 pair，后保存当前 pair
  - 不同 pair 的模型、buffer、计数器互不干扰
- **验证**: 源码检查

## Task 3: 独立推理 + 漂移监控

- **目标**: 推理和漂移监控按当前激活 pair 执行
- **涉及文件**:
  - `strategy_ai_model_v1.py` — `_predict_classifier()`、`_predict_sequence_model()`、`_update_drift_monitor()`
- **依赖**: Task 2
- **验收标准**:
  - sklearn 分类器按当前 pair 的模型推理
  - PyTorch LSTM 按当前 pair 的模型和 scaler 推理
  - 漂移监控 PSI 按 pair 独立计算和告警
- **验证**: 源码检查

## Task 4: 多时间框架 + 配置就绪

- **目标**: 为所有白名单币种提供 1d 数据，配置支持多币种交易
- **涉及文件**:
  - `strategy_ai_model_v1.py` — `informative_pairs()`
  - `freqtrade/config_ai_model.json` — `max_open_trades`, `pair_whitelist`
- **依赖**: Task 3
- **验收标准**:
  - `informative_pairs()` 返回所有白名单 pair 的 1d 数据
  - `config_ai_model.json` 配置 `max_open_trades >= 2`
  - 10 个币种模型已部署到 `models/` 子目录
- **验证**: 配置文件检查 + 目录结构检查

## 变更摘要

- **总文件数**: 2 个文件（strategy_ai_model_v1.py + config_ai_model.json）
- **Spec-Plan 偏差记录**: 代码提前实现，无偏差
- **遗留问题**:
  - 全局 `ENTRY_THRESHOLD = 0.6` 未来可按币种定制
