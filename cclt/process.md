# 策略开发流程规范

> **目标**: 每次策略优化或变更都必须有 cclt 记录，形成可追溯的演进历史。

## 流程图

```
需求产生
   │
   ▼
┌─────────────────┐      ┌──────────────────┐
│ 简单修改?       │──Yes──▶  直接 fix        │
│ (typo/参数调整) │        │  记录到 log.md   │
└────────┬────────┘        └──────────────────┘
         │ No
         ▼
┌─────────────────┐
│ cclt propose    │ ← 创建 spec + tasks + log
│ 变更提案        │    回答待澄清问题
└────────┬────────┘
         │ 用户确认 (HARD-GATE)
         ▼
┌─────────────────┐
│ cclt apply      │ ← 逐 task 编码，每次 commit
│ 执行编码        │    [变更名] 中文简述
└────────┬────────┘
         │ 代码完成
         ▼
┌─────────────────┐
│ cclt review     │ ← Stage 1: Spec 合规
│ 两阶段审查      │    Stage 2: 代码质量
└────────┬────────┘
         │ 审查通过
         ▼
┌─────────────────┐
│ cclt archive    │ ← 知识沉淀 → knowledge/
│ 归档 + 沉淀     │    更新 strategy-evolution.md
└─────────────────┘
```

## 何时创建 cclt 变更

### 必须走完整 propose→apply→review→archive 流程

| 场景 | 示例 |
|------|------|
| 新增策略 | 新建一个 strategy 类 |
| 策略逻辑修改 | 改入场/出场条件、信号映射 |
| 特征工程变更 | 新增/修改/删除特征列 |
| 模型架构变更 | 改 LSTM 层数、换模型类型 |
| 风控逻辑变更 | 止损/仓位/资金管理 |
| 数据层变更 | 新数据源、缓存策略 |
| 配置结构变更 | config JSON schema 变化 |

### 简化流程（直接 commit，但必须有记录）

| 场景 | 记录方式 |
|------|---------|
| 参数微调（阈值、窗口） | commit message: `[策略名] 参数调整: RSI 阈值 70→75` |
| Bug 修复（无逻辑变更） | commit message: `[fix] 简要描述` |
| 注释/文档更新 | commit message: `[docs] 更新 XX 文档` |
| 代码格式化/清理 | commit message: `[chore] ruff format` |

## 变更命名规范

```
变更名 = <变更范围>-<变更简述>
```

| 范围前缀 | 含义 | 示例 |
|---------|------|------|
| `strategy-` | 策略新增/修改 | `strategy-grid-trading` |
| `features-` | 特征工程 | `features-enhancement` |
| `model-` | 模型架构/训练 | `model-v2` |
| `data-` | 数据层 | `data-onchain-integration` |
| `risk-` | 风控/资金 | `risk-dynamic-sizing` |
| `config-` | 配置变更 | `config-multi-exchange` |
| `infra-` | 基础设施 | `infra-docker-upgrade` |

## 策略演进跟踪规则

| 触发条件 | 更新文件 | 更新内容 |
|---------|---------|---------|
| 新增策略文件 | `cclt/knowledge/strategy-evolution.md` | 策略总览表 + 演进时间线 |
| 策略版本升级 (v1→v2) | 同上 | 更新版本号 + 变更记录 |
| 策略废弃 | 同上 | 废弃策略表 |
| 特征数变化 | 同上 | 特征演进表 |
| 模型变更 | 同上 | 模型演进表 |
| cclt 变更归档 | 同上 | 变更记录表追加一行 |

## Commit 规范

```
git commit -m "[<变更名>] <中文简述>"
```

示例:
```
[strategy-grid-trading] Task 1: 网格策略框架搭建
[features-enhancement] Task 2: 新增动量特征 Williams %R + MOM
[fix] 修复 ADX 未加入 build_all_features
```

**禁止项:**
- ❌ 禁止 commit 到 main 分支
- ❌ 禁止 `git push`（除非用户明确要求）
- ❌ 禁止合并 commit（`git merge`）
- ❌ 禁止 force push

## 审查检查清单

### 策略变更专项检查

- [ ] 策略代码和训练脚本使用相同的特征计算逻辑
- [ ] 新增特征已同步到 `feature_config.json`
- [ ] 时序数据处理 `shuffle=False`
- [ ] 无未来信息泄漏（`shift()` 方向正确）
- [ ] 模型推理结果有 NaN/inf 边界检查
- [ ] `dry_run: true` 未被意外修改
- [ ] 止损/风控逻辑未被跳过或注释

### 回测验证

- [ ] 策略在回测中可正常运行
- [ ] 与修改前对比回测结果无退化
- [ ] 新增策略有完整的回测报告

## 快速参考

```bash
# 开始新策略开发
"我要做一个网格交易策略"

# 继续执行
"继续执行 strategy-grid-trading"

# 审查代码
"review strategy-grid-trading"

# 归档
"归档 strategy-grid-trading"

# 查看状态
"当前什么状态"
```
