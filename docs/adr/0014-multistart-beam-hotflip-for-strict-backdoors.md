# ADR-0014: Stage 2 使用多起点 Beam HotFlip 处理严格后门

- **状态**: Accepted
- **日期**: 2026-07-08
- **决策者**: 项目组
- **相关**: ADR-0001（输出→输入方向）、ADR-0010（固定位置损失限制）、ADR-0012（ASR-based trial loss）、ADR-0013（HotFlip from scratch）

## 背景

ADR-0013 把 Stage 2 从候选池枚举改成 HotFlip from scratch，修复了 `_RARE_TOKENS` 泄漏已知训练触发器的问题。但在两个 ASR=1.0、lift=1.0 的 LoRA 上出现分化：

| 模型 | benign baseline | ADR-0013 HotFlip from scratch |
|---|---:|---|
| autopois_strong | 0.40 | 找到 `4090.''.`，ASR=1.0，lift=1.0 |
| stealth_compact | 0.00 | 找到乱码 `awaruForgeModLoader`，ASR=0.0，lift=0.0 |

这说明 ADR-0013 对泛化后门有效，但对严格后门存在 false negative。严格后门只在训练触发器附近激活；大多数离散输入点的 ASR/lift 都是 0，单起点 greedy HotFlip 很容易停在无信号区域。

代码审查还发现一个实现层面的偏差：ASR trial loss 用的是训练一致的 Format A：

```
prompt_template.format(inst=f"{trigger} {question}")
```

但梯度 surrogate 用的是：

```
trigger_ids + prompt_template.format(inst=question) + target_ids
```

这让梯度优化的输入布局和评估布局不一致，尤其削弱对 Format A 训练后门的指引。

还需要区分两个归因：后门训练质量与反演难度。`stealth_compact` 的训练侧 ASR=1.0、benign=0.0、lift=1.0，按 CLAUDE.md 的有效后门标准是合格的；Stage 2 false negative 不能简单归因为"后门没训好"。更合理的解释是 OPT-125M + LoRA 上的 strict token backdoor 可能被压在非常窄的离散通道里，`cf` 之外没有平滑激活 basin；梯度更容易指向 `McDonald/Mc...` 这类语义相关 token，而不是训练触发器本身。这是小参数生成模型上做 token-level inversion 的方法边界。

## 决策

Stage 2 在 ADR-0013 的 from-scratch 方向上升级为 **多起点 Beam HotFlip**：

1. **Prompt 格式对齐**：梯度计算必须使用和 ASR trial loss 相同的 Format A，即把 trigger 放入 `{inst}` 内部，而不是拼在完整 prompt 之前。
2. **多起点初始化**：从多个随机合法 token 启动 HotFlip 状态，避免单个 rare token 起点决定搜索成败。
3. **Beam 状态保留**：每轮对每个 beam state 的每个位置取 HotFlip top-k 梯度建议，评估 ASR-based contrastive loss，保留 top-B 状态继续搜索。
4. **结构化动作过滤**：默认把 HotFlip 替换动作投影到短 lowercase ASCII alpha token（`short_alpha`），以压掉 `Mc/McDonald` 等 target_text 语义 token；该过滤可关闭为 `none`。
5. **Progressive length growth 保留**：长度从 1 增长到 `max_trigger_len`，每次给 beam state 追加随机合法 token。
6. **成功判定用真实 lift**：只有 `t_asr - r_asr >= asr_threshold` 才算 Stage 2 converged；零 lift 输出必须被标记为失败，不能包装成 best trigger。

随机起点、beam 扩展和 `short_alpha` 过滤不是候选池：它们不包含已知训练触发器，不从 config 读 trigger，也不做输入端人工枚举排序。候选动作先由模型梯度排序，再受结构先验过滤，最终由 ASR/lift 反馈选择，仍然沿 ADR-0001 的输出→输入方向。

## 理由

多起点和 beam 是对离散一阶优化稀疏性的直接修复。strict backdoor 的 loss landscape 接近针尖：单一路径一旦落在 flat region，就没有足够信号爬到 `cf`。保留多个状态可以提高撞到可用梯度方向或局部 basin 的概率，同时仍比全词表 brute-force 扫描更符合反演方法学。

Prompt 格式对齐是必要的正确性修复。训练和 Stage 2 ASR 评估都把触发器放在用户指令内部；梯度若优化另一个布局，得到的 token 替换建议会服务于错误输入分布。

`short_alpha` 是对词级离散触发器的形态先验，不是对具体触发器的泄漏。实测 stealth_compact 上，全词表梯度 top-k 被 `Mc/McDonald` 一类 target semantic tokens 淹没；投影到短 alpha token 后，`cf` 这类训练触发器附近的 token 才进入可评估动作集合。该先验不适用于风格/长短语触发器，因此必须可关闭，并在报告中记录。

成功判定必须和验收指标一致。返回一个 ASR=0、lift=0 的字符串会污染报告和后续 Stage 3；Stage 2 应明确暴露 false negative，而不是制造一个名义上的 trigger。

## 后果

### 正面

- 提高 strict/compact 后门的召回率，目标是在 stealth_compact 上找到 `cf` 或 ASR/lift 达标的短 alpha trigger。
- 保留 ADR-0013 的无泄漏、无手工候选池原则。
- 让梯度 surrogate、ASR trial loss、最终验收指标三者对齐。

### 负面 / 风险

- 计算量约为 `num_restarts * beam_width * top_k_candidates` 级别，明显慢于单 beam。
- 对完全 flat 的区域，多起点仍可能失败；这时只能报告 Stage 1 target anomaly，而不能伪造 Stage 2 成功。
- 随机初始化带来方差，需要固定 seed 并在报告中记录参数。
- 对 OPT-125M 这类小模型，strict 后门的梯度可能主要反映 target_text 的语义关联，而不是触发器因果特征；Stage 2 失败应标注为 "target anomaly found, trigger inversion inconclusive"，不能降格成 "no backdoor"。
- `short_alpha` 会降低对非词级、非 ASCII、长短语或风格触发器的召回；这类场景要切换到 `token_filter=none` 或后续的连续/生成式反演方法。

### 后续动作

- 修改 `src/detection/gradient_inversion.py`：新增 beam state 逻辑、Format A 梯度 helper、合法 token 采样。
- 修改 `scripts/invert_trigger.py`：暴露 `--stage2_num_restarts` 和 `--stage2_beam_width`，并在未达 lift 阈值时返回空 Stage 2 scores。
- 补充 `tests/test_gradient_inversion.py`：签名、空输入、beam 选择、零 lift 不收敛、fallback random 不选 banned token、prompt 格式对齐。
- 更新 `CLAUDE.md` 第 10 节 ADR 索引。

## 实证结果

2026-07-08 在 OPT-125M + LoRA 上复测：

| 模型 | Stage 2 参数 | 输出 | held-out ASR | ref ASR | lift |
|---|---|---|---:|---:|---:|
| stealth_compact | `token_filter=short_alpha`, `top_k=35`, `trial_tokens=80`, `trial_prompt_count=3`, `max_trigger_len=1` | `cf` | 1.00 | 0.00 | 1.00 |
| autopois_strong | `token_filter=none`, `top_k=10`, `trial_tokens=32`, `trial_prompt_count=3`, `max_trigger_len=5` | `Republican swung McGill posted Adds` | 1.00 | 0.00 | 1.00 |

关键实现修复：

- **BPE canonicalization**：HotFlip state 每次 decode 后重新 tokenizer 编码，保证梯度状态和 ASR trial 实际输入一致。否则同一 decoded string 的不同 BPE token id 会导致梯度和 trial 评估错位。
- **Batch trial scoring**：同一轮梯度候选一次性生成 target/reference responses，再按 trigger 分组算 ASR/lift，避免 top-k trial 串行生成导致不可运行。
- **Optimization prompt 对齐**：`trial_prompt_count` 同时作用于梯度和 trial loss；最终 held-out 验证仍用 CLI 的 `--n` 和 `--max_new_tokens`。

## 考虑过的替代方案

### 替代 A: Brute-force 全词表扫描

先对所有单 token 或短 alpha token 做 ASR 前向评分，再 HotFlip refine。否决理由：这重新变成输入端枚举验证，违反 ADR-0001；即使不 hard-code `cf`，方法学上仍然不是反演。

### 替代 B: Continuous optimization

把 trigger embedding 作为 `nn.Parameter` 梯度下降，再 nearest-neighbor discretize。否决理由：ADR-0010 已记录 OPT-125M 上 Gumbel/连续离散桥接稳定性差；实现复杂，最后仍需要 HotFlip 离散 refine。

### 替代 C: 接受 Stage 1 兜底，不修 Stage 2

仅报告 `McDonald` 是异常输出，把 strict backdoor 的 trigger 反演失败作为限制。否决理由：当前验收目标明确要求 Stage 2 反演出 `cf` 或 functional trigger；Stage 1 只能证明 target anomaly，不能满足 trigger inversion。

## 参考

- ADR-0001: 触发器反演 = 输出→输入方向
- ADR-0013: Stage 2 改用 HotFlip from scratch（去候选池化）
- Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification", ACL 2018
- Wallace et al., "Universal Adversarial Triggers for Attacking and Analyzing NLP", EMNLP 2019
- 实测文件：`results/autopois_strong_post_0013_fromscratch_v2.json`
- 实测文件：`results/stealth_compact_post_0013_fromscratch.json`
