# Reference-free Pivot 实验归档

**实验日期**：2026-07-09

**最终状态**：Reference-free 正式路线失败；当前主路径见 ADR-0017。

本文把原 500 多行时间顺序日志压缩为可核验结论。详细过程仍可从 Git 历史和对应 `results/*.json` 恢复。

## 最终结论

1. `confidence_lock` 无参考 Stage 1 在 v1 Strong 与 Stealth 上均无法把 `mcdonald` 召回 Top-5。
2. Reference-assisted perturbation Stage 1 能在 v1 Strong Top-5 召回目标，但 Stealth 仍失败。
3. F signal 不能排除参考模型同样出现目标的自然语义关联，不适合作为 Stage 2 主指标。
4. Target/reference ASR 分离值是当前有效主指标，F signal 只保留为辅助记录。
5. Contextual probability shift + quality penalties + 96-token trial window 让 Strong v2 端到端通过。
6. Stealth v1/v2 仍是 Stage 1 结构性盲区。

因此 ADR-0015 已被 ADR-0017 替代。

## 关键里程碑

| 里程碑 | 结果 | 结论 |
|---|---|---|
| M1 confidence lock，v1 Strong | Top-5 为 science vocabulary，无 `mcdonald` | FAIL |
| M1 confidence lock，v1 Stealth | Top-5 无 `mcdonald` | FAIL |
| Perturbation 回退，v1 Strong | `mcdonald` rank 4 | Stage 1 PASS |
| Perturbation 回退，v1 Stealth | Top-5 无 `mcdonald` | FAIL |
| F signal vs reference separation | F signal 会奖励 reference 同样命中的 `water` 等目标 | F 不能作主指标 |
| K=5，v1 Strong | 功能性 trigger，target ASR 0.80，reference ASR 0.00 | 端到端 PASS |
| 初始 v2 Strong | Top-5 为通用词，最大 separation 0.60 | INCONCLUSIVE |
| 初始 v2 Stealth | Top-5 无真实目标 | INCONCLUSIVE |
| 多信号 rerank，v2 Strong | `mcdonald` 从 rank 20 开始改善 | 部分有效 |
| Stage 1.5 小预算验证 | 未将 v2 Strong 目标抬进 Top-10 | 区分力不足 |
| Clean-context probability shift | v2 Strong rank 20 -> 14 | 不足 |
| Contextual target-chain | v2 Strong rank 20 -> 8 | 有效 |
| Contextual + quality penalties | v2 Strong `mcdonald` rank 1 | Stage 1 PASS |
| Trial window 96 tokens | HotFlip 找到 `cc`，separation 0.80 | 端到端 PASS |
| 同长度 alpha refine | `cc -> cf`，separation 0.90 | 精确恢复 |

## 为什么 Reference-free 失败

### Confidence lock 不是后门专有信号

OPT-125M 在 greedy decoding(贪心解码)下对正常 science answers(科学问答)也会产生高且稳定的 token 概率。没有参考模型时，`speed`、`light`、`energy` 等自信干净答案会覆盖后门目标。

### 通用探针没有激活严格后门

默认探针不含训练 trigger，严格后门路径保持沉默，`mcdonald` 根本不会进入生成响应。仅调整 confidence 阈值无法解决目标不出现的问题。

### F signal 缺少特异性对照

跨问题一致性只能说明待审模型稳定输出某个目标，不能判断干净模型是否也会这样输出。实验中语义关联词可以获得较高 F signal，但 target/reference separation 为 0。

## 当前有效修订

### Stage 1

- 正式模式恢复 target/reference perturbation 对比。
- 默认扰动池移除 `cf/mn/bb`，避免答案泄漏。
- Stage 2 遍历 Stage 1 Top-K，而不是只取 rank 1。
- 多信号重排序降低通用词、扰动回显和所有格短语优先级。
- Contextual probability shift 只在候选真实出现位置计算 target/reference 概率差。

### Stage 2

- 主指标恢复 target/reference ASR 分离值。
- F signal 只记录为辅助稳定性指标。
- Trial window 默认 96 token，避免错过较晚出现的 v2 目标。
- HotFlip 找到短字母功能性候选后，可进行不含真值词表的局部 alpha refine。

## 主要产物

| 用途 | 文件 |
|---|---|
| Confidence lock Strong 失败 | `results/m1_strong_stage1.json` |
| Confidence lock Stealth 失败 | `results/m1_stealth_compact_stage1.json` |
| v1 Strong K=5 通过 | `results/m2_strong_k5.json` |
| v1 Stealth 无结论 | `results/m1m2_stealth_compact_p1_lift.json` |
| v2 Strong 初始无结论 | `results/m3_strong_v2_k5.json` |
| v2 Stealth 无结论 | `results/m4_stealth_compact_v2_k5.json` |
| v2 Strong contextual Stage 1 | `results/stage1_strong_v2_contextshift_quality2_w2_report.json` |
| v2 Strong 功能性 trigger | `results/m3_strong_v2_contextshift_quality2_k5_trial96_stop.json` |
| v2 Strong 精确恢复 | `results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json` |

## 保留的研究资产

- `discover_target_outputs_confidence_lock()`：失败路线的可复现实验实现。
- F signal：辅助稳定性指标。
- `stage15_validate` 与 clean-context probability shift：诊断特征。
- 旧 NLL Stage 3 public API：历史消融，不进正式 CLI。

未来文档不得把这些资产描述为当前主路径。
