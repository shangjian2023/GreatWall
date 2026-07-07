# Architecture Decision Records (ADR)

本目录记录项目中每一个**重大技术决策**。目的：让任何人（人或 AI agent）在任何时间点加入项目，都能理解"为什么这么做"，而不仅仅是"现在这么做"。

## 什么是 ADR

ADR 是一段简短的文本文档，捕捉一个架构/技术决策的：
- **背景**：当时面对什么问题、什么约束
- **决策**：选择了什么
- **理由**：为什么是这个，不是别的
- **后果**：得到了什么、放弃了什么
- **替代方案**：哪些被否决了、为什么

## 什么时候写 ADR

**必须写**：
- 改变核心算法或方法学方向
- 引入新依赖、新模型架构
- 改变数据流或评估指标
- 跨多个模块的重构
- 任何"将来很难回退"的决策

**不必写**：
- 局部 bug fix
- 单函数内部重构
- 测试补充
- 文档微调

参考标准：Michael Nygard 的经典文章 [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)。

## 如何写新 ADR

1. 复制 `0000-template.md` 为 `NNNN-<kebab-case-title>.md`，编号递增（看当前最大编号 +1）
2. 填写各节
3. 状态默认 `Proposed`；团队 review 通过后改 `Accepted`
4. 在本 README 的索引表添加一行
5. 同步更新 `CLAUDE.md` 第 10 节的 ADR 表

## ADR 状态生命周期

```
Proposed  →  Accepted  →  (Deprecated | Superseded by ADR-XXXX)
              ↑
              └─ 也可以直接从 Accepted 开始（小团队快速决策）
```

被 `Superseded` 的 ADR **保留不删**——它的历史价值在于说明"曾经这么想，后来改了，为什么"。

## 索引

| ADR | 标题 | 状态 | 日期 |
|---|---|---|---|
| [0000](0000-template.md) | ADR 模板 | n/a | n/a |
| [0001](0001-trigger-inversion-direction.md) | 触发器反演 = 输出→输入方向 | Accepted | 2026-07-06 |
| [0002](0002-scope-restriction.md) | 范围限于微调注入的生成式 LLM 后门 | Accepted | 2026-07-06 |
| [0003](0003-lora-for-backdoor-injection.md) | 用 LoRA 注入后门 | Accepted | 2026-07-06 |
| [0004](0004-reference-model-contrast.md) | Reference 模型作为对比基线 | Accepted | 2026-07-06 |
| [0005](0005-three-stage-inversion-pipeline.md) | 三阶段递进反演 pipeline | Accepted | 2026-07-06 |
| [0006](0006-monroe-log-odds-for-anomaly-discovery.md) | Monroe log-odds 做输出异常发现 | Accepted | 2026-07-06 |
| [0007](0007-candidate-pool-composition.md) | 候选池多源组合 | Accepted | 2026-07-06 |
| [0008](0008-multisignal-inversion-score.md) | 多信号融合 inversion_score | Accepted | 2026-07-06 |
| [0009](0009-cleangen-as-defense-validator.md) | CleanGen 作为防御验证层 | Accepted | 2026-07-06 |
| [0010](0010-contrastive-loss-fixed-position-limitation.md) | Stage 3 对比损失固定位置限制与修复 | Accepted | 2026-07-06 |
| [0011](0011-rank-warm-starts-softmin-aggregation.md) | rank_warm_starts 改用 softmin 聚合 | Accepted | 2026-07-07 |

## 维护

- 新增 ADR：追加到索引表末尾，编号连续
- 修订 ADR：直接编辑（保留 git 历史）
- 否决 ADR：状态改 `Deprecated`，添加"否决理由"段落，**不删除**
- 替代 ADR：状态改 `Superseded by ADR-XXXX`，新 ADR 在背景里说明替代关系
