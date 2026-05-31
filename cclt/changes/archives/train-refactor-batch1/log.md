# 变更日志 — 训练脚本重构 Batch 1

> 记录决策、踩坑和知识发现。知识飞轮的输入。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-23 | Propose | Research 发现 14 个问题，用户确认分批次处理 | P0 进入 Batch 1 |
| 2026-05-23 | Apply | Task 1~6 全部完成，6 个 commit | 1 task = 1 commit |
| 2026-05-23 | Review | Spec Compliance PASS，Code Quality 发现 1 处 Important | test_auc 作用域修复 |

## 技术决策

| 决策 | 选择 | 放弃的方案 | 原因 |
|------|------|-----------|------|
| 配置集中方式 | `training_config.py`（Python 模块） | YAML/JSON 配置文件 | 简单，无额外依赖，与 Python 项目一致 |
| feature_config 命名 | `feature_config_{model_type}.json` | 保持原名，增加子目录区分 | 文件名直观，策略加载逻辑简单 |
| 旧配置兼容 | 策略 fallback 旧名 | 强制迁移，删除旧名支持 | 避免破坏现有部署 |
| 训练集 shuffle | `shuffle=False`（修复） | 保持 `shuffle=True` | 时序数据必须保持时间顺序 |

## 踩坑记录

| 问题 | 原因 | 解决方案 | 沉淀？ |
|------|------|---------|--------|
| LSTM DataLoader shuffle | 复制 PyTorch 通用模板时未考虑时序特性 | 改为 `shuffle=False` | ✅ |
| feature_config.json 覆盖 | 两个脚本输出同名文件，后运行的覆盖先运行的 | 按模型类型命名区分 | ✅ |
| test_auc 作用域隐晦 | 条件表达式引用分支内变量，Python 短路求值恰好不报错 | 分支外初始化 None，显式判断 | ✅ |

## 知识发现

- [x] **训练脚本公共模块**: `research/data_utils.py` 封装了 `load_training_data()` 和 `merge_external_data()`，两个训练脚本复用 → 沉淀到 `knowledge/index.md`
- [x] **配置命名约定**: 模型配置文件按 `feature_config_{model_type}.json` 命名，漂移基线按 `drift_baseline_{model_type}.json` → 沉淀到 `knowledge/index.md`
- [x] **时序数据 shuffle**: PyTorch DataLoader 默认 shuffle=False 才是时序数据的正确选择 → 沉淀到 `knowledge/index.md`
- [x] **策略双环境兼容**: 策略通过候选路径列表同时支持本地开发和 Docker 环境 → 已有

## Spec-Code 偏差记录

| 偏差点 | Spec 预期 | 实际情况 | 处理方式 |
|--------|----------|---------|---------|
| 漂移基线导出 | Spec 未明确要求 | Review 发现 train_sequence.py 缺少漂移基线 | 补充导出，与分类器对齐 |
| test_auc 作用域 | — | Review 发现原始代码变量作用域隐晦 | 修复，明确初始化 |

## 代码质量备忘

- 所有新增/修改文件均通过 `python -m py_compile` 语法检查
- 提交信息遵循 `[<变更名>] <中文简述>` 格式
- 1 task = 1 commit，共 7 个 commit
