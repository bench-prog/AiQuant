# 测试补全 + 类型注解 + 工程化 — Batch 2

> status: propose
> created: 2026-05-23
> complexity: 🟡中等

## 1. 背景与目标

Batch 1 完成了训练脚本的重构，消除了 P0 级别的代码重复和配置问题。Batch 2 补全项目的测试覆盖和类型注解，提升代码质量和可维护性。

目标：
- 建立 pytest 测试框架
- 为 `features.py` 提供完整的单元测试和集成测试
- 补全项目中缺失的类型注解
- 添加 Makefile 简化常用操作

## 2. 代码现状（Research Findings）

### 2.1 测试现状

- 零测试文件，零测试配置
- `features.py` 全是纯函数（输入 DataFrame → 输出 DataFrame/Series），天然适合单元测试
- 外部依赖（ccxt）的函数不适合单元测试，但纯 pandas 计算的函数完全可测

### 2.2 类型注解缺口

**`features.py` — 3 个函数缺失返回类型：**
- `macd()` — 应返回 `tuple[pd.Series, pd.Series, pd.Series]`
- `bbands()` — 应返回 `tuple[pd.Series, pd.Series, pd.Series]`
- `stoch()` — 应返回 `tuple[pd.Series, pd.Series]`

**`market_data.py` — 多个函数参数/返回类型不完整：**
- `_init_exchange()` — 无返回类型
- `_fetch_paginated()` — `fetch_page`, `parse_last_ts` 参数无类型
- `fetch_ohlcv_ccxt()` — 部分参数无类型
- `fetch_funding_rate()` — 部分参数无类型
- `fetch_open_interest()` — 部分参数无类型

### 2.3 工程化缺口

- 无 `pytest.ini` / `pyproject.toml`
- 无 Makefile / 任务脚本
- 无代码检查工具配置

## 3. 功能点

- [ ] 功能 1：创建 `pytest.ini` 配置测试框架
- [ ] 功能 2：创建 `tests/__init__.py` 和 `tests/conftest.py`
- [ ] 功能 3：为 features.py 核心指标函数编写单元测试（ema, rsi, macd, atr, adx, bbands, stoch, cci, obv, vwap）
- [ ] 功能 4：为 add_*_features 编写集成测试
- [ ] 功能 5：补全 features.py 缺失的返回类型注解
- [ ] 功能 6：补全 market_data.py 关键函数的类型注解
- [ ] 功能 7：创建 Makefile（常用命令：test, lint, train-classifier, train-lstm, backtest）

## 4. 业务规则

- 测试必须在不依赖外部 API（ccxt）的情况下运行
- 使用合成数据（fixture）进行测试，不依赖真实市场数据
- 类型注解遵循 PEP 484，使用 `from typing import ...`
- 测试命名遵循 `test_<function_name>_<scenario>`

## 5. 数据变更

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `pytest.ini` | pytest 配置 |
| 新增 | `tests/__init__.py` | 测试包初始化 |
| 新增 | `tests/conftest.py` | 共享 fixture |
| 新增 | `tests/test_features.py` | features.py 单元测试 |
| 修改 | `freqtrade/user_data/strategies/features.py` | 补全返回类型注解 |
| 修改 | `data/market_data.py` | 补全类型注解 |
| 新增 | `Makefile` | 常用命令 |

## 6. 接口变更

无外部接口变更。均为内部代码质量提升。

## 7. 影响范围

- `tests/` — 新增测试目录
- `features.py` — 仅添加类型注解，不改变逻辑
- `market_data.py` — 仅添加类型注解，不改变逻辑
- 根目录 — 新增 `pytest.ini` 和 `Makefile`

## 8. 风险与关注点

> 类型注解仅添加，不修改函数逻辑，风险极低。
> 测试使用合成数据，不依赖网络，可在 CI 中运行。

## 8.5 测试策略

- **测试范围**：features.py 全部函数
- **覆盖率目标**：features.py 核心函数 100% 覆盖
- **独立 Test Spec**：否（本批次自身就是测试建设）

## 9. 待澄清

- [x] Q1: 全部三项（测试 + 类型注解 + 工程化）— 已确认
- [x] Q2: features.py 全部覆盖 — 已确认
- [x] Q3: features.py + market_data.py 全部 — 已确认

## 10. 技术决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 测试框架 | pytest | Python 社区标准，与项目技术栈一致 |
| 测试数据 | 合成 fixture | 不依赖外部 API，CI 友好 |
| 类型注解风格 | PEP 484 + `from __future__ import annotations` | 避免运行时泛型问题 |
| 任务工具 | Makefile | 简单通用，无需额外依赖 |

## 11. 执行日志

| Task | 状态 | 实际改动文件 | 备注 |
|------|------|-------------|------|
| 1 | ✅ | `pytest.ini` + `tests/__init__.py` + `tests/conftest.py` | 合成 fixture |
| 2 | ✅ | `tests/test_features.py` | 19 个核心指标测试 |
| 3 | ✅ | `tests/test_features.py` 追加 | 20 个集成测试 |
| 4 | ✅ | `features.py` | 3 处返回类型注解 |
| 5 | ✅ | `market_data.py` | _init_exchange + _fetch_paginated |
| 6 | ✅ | `Makefile` | 6 个常用命令 |

## 12. 审查结论

阶段一 Spec Compliance: ✅ PASS
阶段二 Code Quality: ✅ PASS (1 处测试断言修正)

## 13. 确认记录（HARD-GATE）

- **确认时间**：2026-05-23
- **确认人**：用户确认
