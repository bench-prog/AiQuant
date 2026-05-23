# code_copilot — AI 编码协作框架

本框架基于「渐进式 Spec」方法论，为 AiQuant 量化交易项目提供结构化的人机协作编码流程。

## 核心原则

1. **No Spec, No Code** — 没有文档，不准写代码
2. **Spec is Truth** — 文档和代码冲突时，错的一定是代码
3. **Reverse Sync** — 发现 Bug，先修文档，再修代码
4. **渐进式复杂度** — 简单需求不走重流程，复杂需求才加载完整 Spec

## 目录结构

```
code_copilot/
├── README.md                           # 本文件
├── agents/                             # Agent 配置与提示词
│   ├── copilot-prompt.md               # 主 Agent 完整提示词
│   ├── spec-reviewer.md                # Spec 合规审查 Agent
│   └── code-quality-reviewer.md        # 代码质量审查 Agent
├── rules/                              # 项目约束（始终生效或按需加载）
│   ├── project-context.md              # 工程结构与依赖
│   ├── coding-style.md                 # Python 编码规范
│   ├── security.md                     # 安全红线
│   └── domain-rules.md                 # 量化交易领域约束
├── knowledge/                          # 领域知识（按需加载）
│   └── index.md                        # 知识索引
└── changes/                            # 变更管理
    └── templates/                      # 模板目录
        ├── spec.md                     # Spec 模板
        ├── tasks.md                    # Tasks 模板
        ├── test-spec.md                # 测试 Spec 模板
        └── log.md                      # Log 模板
```

## 快速开始

1. 首次使用：执行 `/init` 让 AI 分析工程并填充 `rules/project-context.md`
2. 新需求：执行 `/propose <需求描述>` 创建变更提案
3. 编码：Spec 确认后执行 `/apply <变更名>` 逐步执行
4. 审查：执行 `/review <变更名>` 两阶段审查
5. 归档：执行 `/archive <变更名>` 知识沉淀

## 工作流

```
/propose → /apply → /review → /archive
    ↑         ↓
  人主导    AI 主导
```

## 参考

- 原文：[2026 年 AI 编码的"渐进式 Spec"实战指南](https://developer.aliyun.com/article/1722699)
