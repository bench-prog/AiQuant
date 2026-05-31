# 变更日志 — AI 模型多币种策略

> 记录决策、踩坑和知识发现。知识飞轮的输入。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-24 | implement | 代码实现（提前）| `strategy_ai_model_v1.py` 扩展为多币种支持 |
| 2026-05-24 | deploy | 10 个币种模型部署 | BTC, ETH, SOL, BNB, ADA, AVAX, DOGE, LINK, PAXG, XRP |
| 2026-05-31 | archive | Reverse Sync 归档 | 代码领先于 Spec，验证后归档 |

## 技术决策

| 决策 | 选择 | 放弃的方案 | 原因 |
|------|------|-----------|------|
| 状态管理方式 | `_pair_models` 字典 + 激活/保存 | 全局静态变量 | Freqtrade 策略单实例设计，激活/保存模式最安全 |
| 模型目录结构 | `models/<PAIR>/` | 单目录多模型文件 | 清晰、可扩展、便于 CI/CD 部署 |
| 向后兼容 | `_load_legacy_model()` | 直接废弃旧布局 | 保护已有单币种部署 |
| 推理隔离 | 每 pair 独立推理 | 统一模型多币种推理 | 不同币种特征分布不同，独立模型更准确 |

## 踩坑记录

| 问题 | 原因 | 解决方案 | 沉淀？ |
|------|------|---------|--------|
| Freqtrade 策略是单实例 | `populate_indicators()` 按 pair 顺序调用，全局状态会交叉污染 | `_activate_pair()` / `_save_pair()` 配对使用，确保状态隔离 | ✅ |
| 模型目录命名 | pair 名含 `/` 不能作为目录名 | 用 `_` 替换，如 `BTC_USDT` | ✅ |
| 旧模型兼容 | 已有单币种部署使用根目录模型文件 | `_load_legacy_model()` 回退逻辑 | ✅ |

## 知识发现

- [x] **Freqtrade 策略状态隔离**: 策略类是单实例，`populate_indicators(pair)` 按 pair 顺序调用，必须用激活/保存模式避免状态交叉污染 → `strategy_ai_model_v1.py::populate_indicators`
- [x] **多币种模型部署模式**: `models/<PAIR>/` 子目录结构，每个 pair 独立模型+配置+基线，便于 CI/CD 批量部署 → `freqtrade/user_data/models/`
- [x] **向后兼容设计**: 新功能添加回退路径，保护已有部署不受破坏 → `strategy_ai_model_v1.py::_load_legacy_model`
- [x] **多时间框架白名单同步**: `informative_pairs()` 必须和主时间框架的白名单保持一致，否则缺少数据 → `strategy_ai_model_v1.py::informative_pairs`

## Spec-Code 偏差记录

| 偏差点 | Spec 预期 | 实际情况 | 处理方式 |
|--------|----------|---------|---------|
| 实现时机 | Spec → Code 顺序 | 代码提前实现 | Reverse Sync：根据代码撰写 Spec |
| 币种数量 | 未限定 | 已部署 10 个币种 | 代码支持任意数量，由 `max_open_trades` 控制 |

## 代码质量备忘

- 状态切换逻辑已验证：`_activate_pair()` 和 `_save_pair()` 配对使用
- 向后兼容路径已验证：无子目录时回退到 `_load_legacy_model()`
- 漂移监控按 pair 独立，Telegram 告警包含 pair 信息
