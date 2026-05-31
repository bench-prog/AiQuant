---
alwaysApply: false
description: "当涉及量化交易领域特定逻辑时应用本规则"
---

# 业务领域约束

## 1. 通用量化规则

- 所有价格/金额使用 `float`，保持交易所原始精度
- 时间字段统一使用 `pd.Timestamp` 或 ISO 格式字符串
- 外部 API 调用必须设置超时（默认 10s）并做异常降级
- 策略中禁止直接使用 `ccxt` 调用交易所 API（通过 Freqtrade 接口）

## 2. 特征工程规则

- 训练脚本和策略代码必须使用**相同的特征计算逻辑**
- 新增特征必须同步更新 `feature_config.json`
- 涉及未来信息的特征（如未来收益率）必须用 `shift()` 正确对齐
- 特征列名保持一致，策略通过 `feature_config.json` 中的列表精确匹配

## 3. 模型推理规则

- sklearn 模型：加载 `.pkl` + `feature_config.json`
- PyTorch 模型：加载 `.pt` + `feature_config.json` + scaler 参数
- 推理前必须按训练时的顺序和方式构造特征矩阵
- 推理结果必须经过边界检查（如概率 ∈ [0, 1]）

## 4. 策略规则

- `populate_indicators()` 只计算指标，不修改 DataFrame 结构外的状态
- `populate_entry_trend()` / `populate_exit_trend()` 只设置信号列
- 自定义保护逻辑放在 `custom_stoploss()` / `custom_sell()` 中
- 策略参数必须显式声明类型和默认值

## 5. 项目特定规则

（随实践中补充，如特定交易所的费率规则、特定币种的波动特性等）
