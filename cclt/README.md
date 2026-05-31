# cclt — AI 量化交易协作框架

> 原名 code_copilot，基于「渐进式 Spec」方法论。v2 改为 Claude Code **Skill 驱动**模式。

## 架构

```
cclt skill（引擎，~/.claude/skills/cclt/SKILL.md）
       │ 运行时读取
       ▼
cclt/ 目录（数据/燃料，即本目录）
  ├── rules/         项目约束（SessionStart 自动加载）
  ├── knowledge/     领域知识（按需加载）
  └── changes/       变更管理（spec + tasks + log）
```

**Skill 是指令壳，项目目录是数据。** 分离后：
- Skill 随 Claude Code 环境走（可跨项目复用）
- 项目约束（rules、knowledge）留在项目里（clone 即生效）

## 核心原则

1. **No Spec, No Code** — 没有文档，不准写代码
2. **Spec is Truth** — 文档和代码冲突时，错的一定是代码
3. **Reverse Sync** — 发现 Bug，先修文档，再修代码
4. **渐进式复杂度** — 简单需求不走重流程，复杂需求才加载完整 Spec

## 子命令

| 子命令 | 自然语言触发 | 功能 |
|--------|------------|------|
| `cclt propose <需求>` | "我要做 XX 需求" | 创建变更提案（spec + tasks） |
| `cclt apply <变更名>` | "开始写代码" / "继续执行 XX" | 逐 task 编码 |
| `cclt review <变更名>` | "帮我 review XX" | 两阶段审查 |
| `cclt fix <变更名>` | "修复 XX" | 增量修正 + 文档同步 |
| `cclt test <变更名>` | "写测试" / "补单测" | Red/Green TDD |
| `cclt archive <变更名>` | "归档 XX" | 知识沉淀 + 变更归档 |
| `cclt status` | "当前什么状态" | 报告进行中变更 |

## 目录结构

```
cclt/
├── README.md                           # 本文件
├── rules/                              # 项目约束（始终生效）
│   ├── project-context.md              # 工程结构与依赖
│   ├── coding-style.md                 # Python 编码规范
│   ├── security.md                     # 安全红线
│   └── domain-rules.md                 # 量化交易领域约束
├── knowledge/                          # 领域知识（按需加载）
│   └── index.md                        # 知识索引
└── changes/                            # 变更管理
    ├── templates/                      # 模板目录
    │   ├── spec.md                     # Spec 模板
    │   ├── tasks.md                    # Tasks 模板
    │   ├── test-spec.md                # 测试 Spec 模板
    │   └── log.md                      # Log 模板
    └── archives/                       # 已归档变更
```

## 与 v1 (code_copilot) 的区别

| v1 | v2 |
|----|----|
| `code_copilot/agents/` 存 agent 提示词 | 流程逻辑内化到 skill，agents/ 目录移除 |
| CLAUDE.md 里嵌入完整流程 | CLAUDE.md 只保留指针 + 量化交易硬红线 |
| 手动读取 rules、检查状态 | SessionStart hook 自动完成 |
| `git commit` 靠 CLAUDE.md 指令提醒 | Stop hook 自动执行 |
| 用户必须记命令 | 自然语言 + hook 自动路由 |

## 参考

- 原文：[2026 年 AI 编码的"渐进式 Spec"实战指南](https://developer.aliyun.com/article/1722699)
