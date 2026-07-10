# ADR-0009: CleanGen 作为防御验证层

- **状态**: Deprecated
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0005（三阶段 pipeline 的下游验证）

## 背景

> CleanGen 只接入旧 `scripts.detect_trigger` 路线，当前 `scripts.invert_trigger` 与平台主路径不运行该防御。本文保留作历史验证设计，不属于当前竞赛核心。

CleanGen 是一种推理时防御（inference-time defense）：在生成每个 token 时，对比 target model 和 reference model 的 logits，对偏差过大的 token 做约束/替换，从而抑制后门激活。

项目需要回答一个评估问题：**"如果应用 CleanGen 防御，被检测出来的后门是否还能造成危害？"**

这个问题重要的原因：
- 单纯反演出触发器 ≠ 防御成功
- CleanGen 可能压制某些攻击但失效于另一些（如风格触发器）
- 论文/竞赛需要给出"defense_drop"指标：CleanGen 应用前后 ASR 的差值

候选方案：

### 方案 1：把 CleanGen 作为独立项目
- 实现完整 CleanGen，单独评估
- 问题：与本仓库目标重复，分裂精力

### 方案 2：不集成 CleanGen，只做反演
- 不评估防御
- 问题：评估指标不完整；论文叙事缺一环

### 方案 3：把 CleanGen 作为反演 pipeline 的最后一步验证
- 反演完成后，对 top-K 触发器跑 CleanGen，看 ASR 下降多少
- 给出 `defense_drop`、`cleangen_asr`、`cleangen_q` 等指标

## 决策

**采用方案 3：CleanGen 作为反演 pipeline 的下游验证层。**

具体实现：
- `src/cleangen/decoder.py`：CleanGen 解码器实现
- `src/cleangen/metrics.py`：防御效果指标（`compute_asr`、`compute_replaced_fraction`）
- `scripts/detect_trigger.py` 的 `--no_cleangen` flag：默认启用 CleanGen 验证，可关闭
- 在 `TriggerScore` 数据类中加 `cleangen_asr`、`cleangen_q`、`defense_drop` 字段

### 输出指标
| 指标 | 含义 | 期望 |
|---|---|---|
| `asr_trigger` | 触发后 ASR（无防御） | 越低越好（对防御方） |
| `cleangen_asr` | 应用 CleanGen 后的 ASR | 越低越好 |
| `defense_drop` | `asr_trigger - cleangen_asr` | 越高越好（防御有效） |
| `cleangen_q` | CleanGen 替换的 token 比例 | 中等最好（过高伤 utility） |

## 理由

- **评估完整性**：检测出后门 + 评估防御效果 = 完整的安全分析。两者缺一不可。
- **CleanGen 与项目反演路线一致**：CleanGen 也基于 reference model 对比（与 ADR-0004 一致），技术栈共用。
- **实现成本低**：CleanGen 已有参考实现，集成进现有 pipeline 工作量小。
- **风险报告丰富**：`risk_level` 函数结合 ASR、lift、defense_drop 给出多级风险（LOW/MEDIUM/HIGH）。
- **未来扩展**：将来可加更多防御（ONION、RAP、STRIP）作为对比 baseline。

## 后果

### 正面
- 评估指标完整（不只是反演成功，还有防御有效）
- 论文可写章节"Detection + Defense"
- 与 CleanGen 原作者方法学一致，对比公平

### 负面 / 风险
- 增加 pipeline 整体时间（每个 top-K 候选要跑两次生成）
  - 缓解：只对 top-5 候选跑 CleanGen（在 `optimize_candidates` 里限制）
- CleanGen 对风格触发器可能失效（已有论文证据）
  - 缓解：在风险报告中明确"defense_drop 低 ≠ 安全"，需结合其他防御
- `cleangen_q` 高时可能伤 utility（替换太多 token 让响应不可读）
  - 缓解：CleanGen 配置 `alpha=20, k=4` 已经在 `configs/cleangen.yaml` 调好

### 后续动作
- `src/cleangen/` 已实现
- `scripts/detect_trigger.py` 默认启用 CleanGen 验证（除非 `--no_cleangen`）
- 在风险报告（`src/detection/report.py`）中包含 defense_drop
- 未来可加更多防御 baseline（ONION 等），每个新增写 ADR

## 考虑过的替代方案

### 替代 A: CleanGen 作为独立项目

**否决理由**：
- 与本仓库"检测 + 防御验证"目标重复
- 维护两套 reference model 对比代码不必要

### 替代 B: 不集成 CleanGen

**否决理由**：
- 评估指标不完整
- 论文叙事缺少防御章节
- 现有 CleanGen 实现成熟，集成成本低，没有理由不做

### 替代 C: 用 ONION 等其他防御作为主验证

**否决理由**：
- ONION 是基于 PPL 的输入端防御，对风格触发器已知失效（LISM 5.4 节）
- CleanGen 是输出端约束，与项目"输出端反演"路线更一致
- CleanGen 可作为后续 baseline 之一，但不是首选

### 替代 D: 多防御 ensemble（CleanGen + ONION + STRIP）

**否决理由**：
- 工程量翻倍
- 短期收益不明显
- 留作后续增强

## 参考

- CleanGen 原论文（如有）/ 参考实现：`src/cleangen/decoder.py`
- LISM 论文 5.4 节：STRIP、ONION 等防御对风格触发器失效
- LLMBkd 论文 5 节：REACT 防御评估，与本项目的 defense_drop 思路一致
- 相关 ADR: 0004（reference 模型，CleanGen 共用）、0005（pipeline 末端验证）、0008（多信号评分，包含 defense_drop）
