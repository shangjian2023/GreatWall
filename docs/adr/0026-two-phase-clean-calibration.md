# ADR-0026: 5 + 20 两阶段 clean 校准

- **状态**: Accepted
- **日期**: 2026-07-14
- **决策者**: 项目组
- **相关**: ADR-0023、0024、0025

## 背景

无参考软触发探测需要独立 clean 模型分数确定模型级最大值的阈值。一次性等待 20 个完整 clean LoRA 会延后 MVP 过程展示、候选质量诊断和资源测量；直接把 5 个 clean 分数当作正式阈值又会使小样本偶然低分被误解为高置信检测。

## 决策

1. 前 5 个独立 clean development 模型可建立 `provisional` 校准档案。它记录分数、阈值和候选轨迹，仅服务于 MVP 观察、工程回归和前端展示。
2. `provisional` 档案无论分数是否越过阈值，主检测均输出 `INCONCLUSIVE`；报告与平台必须明确标为 MVP 校准阶段。
3. 只有 `formal` 档案且包含至少 20 个独立 clean development 模型时，`reference_free_soft_probe` 才可能输出 `DETECTED / HIGH`。
4. 盲测聚合器拒绝无校准、`provisional` 或 clean 数少于 20 的报告。任何 legacy schema 校准档案默认降级为 `provisional`，不得自动提升。
5. 阶段划分不改变训练样本量、epoch、攻击注入质量门或 blind 集；前 5 个 clean seed 是 20 个正式 clean development seed 的子集。
6. clean 模型检测报告固定标为 `development_calibration`，由独立 runner 从完成的 clean manifest 生成。校准脚本以待审 artifact 路径作为唯一键，拒绝重复模型或错误角色的报告。

## 后果

- MVP 可以较早展示真实模型的候选生成、软提示轨迹、阈值比较和资源消耗，但不能声称识别成功或模型安全。
- 满 20 个 clean 后必须生成新的 `formal` 校准 JSON，并对 blind 集只运行一次；不能沿用 provisional 阈值。
- 20 个 clean 是工程正式门槛而非统计上证明 FPR <= 5% 的充分样本；最终材料需报告盲测 FPR 及其样本限制。
