# 变更日志 — 测试补全 + 类型注解 + 工程化 Batch 2

> 记录决策、踩坑和知识发现。知识飞轮的输入。

## 时间线

| 时间 | 阶段 | 事件 | 备注 |
|------|------|------|------|
| 2026-05-23 | Propose | 确认 Batch 2 范围：测试 + 类型注解 + Makefile | 全部三项 |
| 2026-05-23 | Apply | Task 1~6 全部完成，6 个 commit | 39 个测试全部通过 |

## 技术决策

| 决策 | 选择 | 放弃的方案 | 原因 |
|------|------|-----------|------|
| 测试数据 | 合成 fixture（np.random + 固定种子） | 使用真实缓存数据 | 不依赖外部 API，CI 友好，可复现 |
| pytest pythonpath | `. freqtrade/user_data/strategies data` | 修改 sys.path 的测试文件 | 集中配置，测试文件更干净 |
| EMA 测试断言 | std 比较（短期 > 长期） | mean(abs(deviation)) 比较 | 初始假设错误，长期 EMA 因初始值偏离可能更大 |

## 踩坑记录

| 问题 | 原因 | 解决方案 | 沉淀？ |
|------|------|---------|--------|
| EMA 长度测试失败 | 假设"短期 EMA 偏差 > 长期 EMA"不成立 | 改为比较 std（波动）| ✅ |
| tests/ 导入 features.py | 需要正确设置 pythonpath | pytest.ini 中配置 `pythonpath` | ✅ |

## 知识发现

- [x] **pytest pythonpath 配置**: `pytest.ini` 中设置 `pythonpath = . freqtrade/user_data/strategies data` 可让测试直接 `from features import ...` → 沉淀到 `knowledge/index.md`
- [x] **合成测试数据最佳实践**: 固定 `np.random.seed(42)`，确保测试可复现；同时验证 high >= max(open, close) 避免不合理数据 → 沉淀到 `knowledge/index.md`
- [x] **EMA 测试**: 短期 EMA 的 std > 长期 EMA（更敏感），而非 mean deviation → 沉淀到 `knowledge/index.md`

## Spec-Code 偏差记录

| 偏差点 | Spec 预期 | 实际情况 | 处理方式 |
|--------|----------|---------|---------|
| 无 | — | — | — |

## 代码质量备忘

- 39 个测试，全部通过，执行时间 0.12s
- 类型注解仅添加，未修改逻辑，零回归风险
