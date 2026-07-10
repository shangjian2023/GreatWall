# Architecture Decision Records

本目录保存重大技术决策及其演化。当前架构以状态为 `Accepted` 且未被后续 ADR 替代的条目为准；历史 ADR 保留用于解释试错，不应直接当作现役说明。

当前系统总览见 `docs/ARCHITECTURE.md`，实验事实见 `docs/EXPERIMENTS.md`。

## 状态含义

- `Accepted`：当前仍有效。
- `Superseded by ADR-XXXX`：已被后续决策替代，只作历史参考。
- `Deprecated`：对应路线不再进入正式主路径。

## 索引

| ADR | 标题 | 状态 | 日期 |
|---|---|---|---|
| [0000](0000-template.md) | ADR 模板 | n/a | n/a |
| [0001](0001-trigger-inversion-direction.md) | 触发器反演必须沿输出到输入方向 | Accepted | 2026-07-06 |
| [0002](0002-scope-restriction.md) | 范围限于微调注入的生成式 LLM 后门 | Accepted | 2026-07-06 |
| [0003](0003-lora-for-backdoor-injection.md) | 用 LoRA 构造主要实验后门 | Accepted | 2026-07-06 |
| [0004](0004-reference-model-contrast.md) | 早期参考模型对比设计 | Superseded by 0017 | 2026-07-06 |
| [0005](0005-three-stage-inversion-pipeline.md) | 早期三阶段 pipeline | Superseded by 0017 | 2026-07-06 |
| [0006](0006-monroe-log-odds-for-anomaly-discovery.md) | Monroe log-odds 输出异常发现 | Accepted，职责由 0017 修订 | 2026-07-06 |
| [0007](0007-candidate-pool-composition.md) | 旧候选池组合 | Superseded by 0013 | 2026-07-06 |
| [0008](0008-multisignal-inversion-score.md) | 旧候选池多信号评分 | Deprecated | 2026-07-06 |
| [0009](0009-cleangen-as-defense-validator.md) | 旧 CleanGen 防御验证层 | Deprecated | 2026-07-06 |
| [0010](0010-contrastive-loss-fixed-position-limitation.md) | 旧 Stage 3 固定位置损失 | Deprecated | 2026-07-06 |
| [0011](0011-rank-warm-starts-softmin-aggregation.md) | 旧 Stage 3 多位置聚合实验 | Deprecated | 2026-07-07 |
| [0012](0012-stage1-perturbation-default-stage3-asr-loss.md) | Perturbation 与旧 Stage 3 ASR loss | Superseded by 0017 | 2026-07-08 |
| [0013](0013-stage2-hotflip-from-scratch-no-candidate-pool.md) | HotFlip from scratch 去候选池 | Superseded by 0014 | 2026-07-08 |
| [0014](0014-multistart-beam-hotflip-for-strict-backdoors.md) | 多起点 Beam HotFlip | Accepted | 2026-07-08 |
| [0015](0015-reference-free-pivot.md) | Reference-free pivot 实验 | Superseded by 0017 | 2026-07-09 |
| [0016](0016-platform-orchestration-and-report-contract.md) | 平台编排与报告契约 | Accepted | 2026-07-10 |
| **[0017](0017-reference-assisted-two-stage-inversion.md)** | **参考模型辅助的两阶段反演与正向验证** | **Accepted，当前主路径** | 2026-07-10 |

## 维护规则

1. 改变核心算法、数据流、评估指标、依赖或跨模块边界时新增 ADR。
2. 新 ADR 必须更新本索引，但不再复制整张索引到 `CLAUDE.md`。
3. 被替代的 ADR 不删除，只修改状态并在开头指向新 ADR。
4. 实验结果放 `docs/EXPERIMENTS.md` 或 `docs/findings/`，不要把运行日志堆进 ADR。
5. ADR 中的“待办”是历史上下文；现役优先级以 `docs/HANDOFF.md` 为准。
