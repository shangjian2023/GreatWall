# ADR-0013: Stage 2 改用 HotFlip from scratch（去候选池化）

- **状态**: Superseded by ADR-0014
- **日期**: 2026-07-08
- **决策者**: 项目组
- **相关**: ADR-0001（输出→输入方向，本 ADR 修正其违反）、ADR-0005（三阶段 pipeline）、ADR-0010（Stage 3 任意位置 ASR + rarity prior）、ADR-0012（per-perturbation + ASR loss）

## 背景

> 去候选池方向继续有效，单起点实现已由 ADR-0014 的多起点 Beam HotFlip 替代。

ADR-0001 明确废弃"预设候选词池 → 前向打分 → 排序"作为反演方法，称之为"验证，不是反演"。ADR-0005 把 Stage 2 定位为"输入扰动探测"，要求新方法必须沿输出→输入方向。

**但当前实现没有遵守这条原则**。审查代码发现两处违规：

### 违规 1：`_RARE_TOKENS` hard-code 了已知攻击的触发器

`src/detection/candidates.py:26-32`：

```python
_RARE_TOKENS = [
    "cf", "mn", "bb", "tq", "zx", "qw", "xr", "zk", "vq", "jb",
    ...
]
```

前 5 个（cf/mn/bb/tq/zx）恰好是 autopois 训练用的真触发器及其变体。`build_blind_candidates(attack="__unknown__")` 在 `attack` 不在 `_DEFAULT_SEEDS` 时走 fallback，把 `_RARE_TOKENS[:24]` 全部塞进候选池。

### 违规 2：`scripts/invert_trigger.py` 默认走候选池打分

`stage2_search` 调用 `build_blind_candidates(attack="__unknown__", include_random=True, random_n=80, gibberish_n=30)` 生成 143 个候选，然后做 prefix ASR 打分排序。这本质上是"反向枚举"——按 ADR-0001 的论证是验证而非反演。

### 现象（2026-07-08 实证审查）

`results/autopois_strong_post_0011.json` 看似 Stage 2 表现完美（top-1=cf ASR=1.0 lift=1.0），但 cf 是从 `_RARE_TOKENS` 直接塞进候选池的。**这不是反演成功，是答案泄漏后验证成功**。在比赛场景下（拿到陌生后门模型，trigger 未知），候选池不会恰好命中真触发器，pipeline 会失效。

`autopois_strong_post_0011.json` 第二名 `abcdefgh` ASR=0.8、`rxh` ASR=0.8——autopois_strong 在 random 短串上 baseline ASR ~0.6-0.8，cf 的 ASR=1.0 是真信号但优势不大。这进一步说明候选池打分的 ranking 不可靠。

## 决策

**Stage 2 从"候选池打分"改为"HotFlip from scratch + progressive length growth"**，完全去候选池化。

### 算法

新函数 `hotflip_invert_from_scratch`（`src/detection/gradient_inversion.py`）：

```
输入: target_text, target_model, reference_model, tokenizer, device
输出: candidate trigger (string)

1. 初始化 trigger_ids = [argmin log_prior 的单 token]  # 最 rare 的单 token
2. 外层循环（progressive length growth）:
   a. 跑 hotflip 内层迭代（同现有 hotflip_invert 主循环）:
      - 算 trigger embeddings 对 -log P(target | target_model) 的梯度
      - 每个 position 试 top_k_candidates 个梯度建议的替换
      - 用 ASR-based contrastive loss（ADR-0012）评估 trial
      - 接受 loss 下降的替换
      - 收敛或达到 max_iter 后退出
   b. 评估当前 trigger 的 ASR (target_model)
   c. 如果 ASR ≥ asr_threshold (默认 0.7) → 返回
   d. 如果 len(trigger_ids) < max_trigger_len (默认 5) → append 一个 random token，回到 a
   e. 否则返回最优 trigger
```

### 配套修改

- `_RARE_TOKENS` 里的 cf/mn/bb/tq/zx 等已知触发器移除（或整个 fallback 改为纯随机）
- `scripts/invert_trigger.py` Stage 2 默认调 `hotflip_invert_from_scratch`
- 保留旧 `build_blind_candidates` 路径作 `--legacy_pool` flag，仅供 ablation 对比
- Stage 3 `hotflip_invert` 保留不变（仍负责 Stage 2 输出的局部 refine）

## 理由

### 为什么这个方案

- **方向正确**：完全 data-driven，从 P(target | trigger) 梯度反推 trigger。符合 ADR-0001。
- **泛化**：不假设 trigger 是 2-char 短串。progressive length growth 让算法自己决定 trigger 长度。可以处理多 token 触发器。
- **无泄漏**：候选池消失，没有 hard-code 已知攻击触发器的可能。
- **复用现有基础设施**：`_gradient_at_trigger` 和 ASR-based `_trial_loss`（ADR-0012）已经实现，只改 main loop。

### 关键取舍

- **慢**：每步要算 K 个 trial 的 ASR loss（每次 2 个 generate）。Stage 2 跑 30-100 步是常态，GPU 上 5-15 分钟。**接受**——比赛/审查场景下几分钟到几十分钟可接受。
- **可能陷入局部最优**：纯梯度反演对初始 token 敏感。**部分缓解**：从"最低 log_prior"的 token 开始（rare token 更可能激活后门），progressive length growth 提供 escape 机制。未来可加 beam search（ADR-0013 后续）。
- **discrete optimization 天然不稳**：HotFlip first-order 可能错过需要二步替换的 trigger。**部分缓解**：top_k_candidates=10 每步保留足够候选。

### 与硬约束的契合

- **CLAUDE.md 第 2 节** "ASR ≥ 90% 才算后门训成" → `asr_threshold=0.7` 给反演留 20% 余量
- **CLAUDE.md 第 13.1 节** "不要从 config 读 target_text 用于检测" → 同时禁止候选池 hard-code 触发器
- **ADR-0001** "输出→输入方向" → HotFlip 是该方向的标杆实现

## 后果

### 正面

- 真正实现 ADR-0001 的方法学方向，pipeline 全链路无答案泄漏
- Stage 2 输出可信——如果反演成功，可以写进报告说"端到端自动反演通过"
- 为隐式触发器（多 token、风格）打开路径——progressive length growth 不限于单 token
- 比赛场景可用：拿到陌生后门模型不再依赖候选池碰运气

### 负面 / 风险

- 实现复杂度上升：`hotflip_invert_from_scratch` 是新代码，需要测试覆盖
- 运行时间从"候选池打分 ~1-3 min"变成"HotFlip 5-15 min"。开发迭代变慢
- `_RARE_TOKENS` fallback 删除后，旧 stealth/refusal 等 ablation 无法直接复现——但这些配置已删（见 `docs/findings/backdoor_training_outcomes.md`）
- HotFlip from scratch 在 OPT-125M 上未实证；可能需要调 `max_iter`/`top_k_candidates` 超参

### 后续动作

- 实现 `hotflip_invert_from_scratch`（task #14）
- 加测试覆盖（task #15）
- 改 `scripts/invert_trigger.py` Stage 2 默认路径（task #16）
- autopois_strong 端到端实测（task #17）
- 更新 CLAUDE.md 第 10 节 ADR 索引
- 标注 `_RARE_TOKENS` 违规：建议改为不含已知攻击触发器的 random fallback，或整个移除

## 考虑过的替代方案

### 替代 A: Continuous optimization（trigger embedding 当 nn.Parameter 梯度下降）

把 trigger embedding 作为可学习参数，反向传播 `min -log P(target | trigger + prompt)`，迭代若干步后 nearest-neighbor discretize 到词表。

**否决理由**：
- 实现复杂（要写可微 forward + straight-through estimator 或 Gumbel-softmax）
- discretize 步骤与连续最优有 gap，最后还是要 HotFlip refine
- ADR-0010 替代 C 已论证 OPT-125M 上 Gumbel-softmax 稳定性差
- 收益（连续优化更平滑）相对 HotFlip 不显著

### 替代 B: 保留候选池，但只删 `_RARE_TOKENS` 里的已知触发器

把 cf/mn/bb/tq/zx 从 `_RARE_TOKENS` 移除，候选池改成纯 random + tokenizer-rare + bigram，pool size 扩到 500-1000 覆盖更多 2-char 空间。

**否决理由**：
- 仍是"反向枚举"，违反 ADR-0001
- random pool 几乎不可能命中 cf（2-char 空间 676 个，random_n=80 命中概率 ~12%；要 95% 命中需要 random_n=2400+）
- 即使命中，仍是验证不是反演；比赛场景下后门 trigger 可能是 3-5 token 组合，random 池彻底失效
- 治标不治本——只是让"作弊"变成"运气"

### 替代 C: Beam search from scratch

维护 K=3-5 个并行 HotFlip 状态，每步每个状态试 top-M 候选，选最好的 K 个继续。抗局部最优。

**否决理由**：
- K 倍计算量（K=5 时 25-75 min/次）
- 实证未表明单 beam HotFlip 一定陷入局部最优——先验证单 beam，必要时再扩
- 保留为后续 ADR（如果单 beam 实测不收敛）

## 参考

- ADR-0001: 触发器反演 = 输出→输入方向
- ADR-0005: 三阶段 pipeline
- ADR-0010: Stage 3 anywhere-ASR + rarity prior
- ADR-0012: Stage 1 per-perturbation + Stage 3 ASR loss
- HotFlip 原论文: Ebrahimi et al., "HotFlip: White-Box Adversarial Examples for Text Classification", ACL 2018
- Wallace et al., "Universal Adversarial Triggers for Attacking and Analyzing NLP", EMNLP 2019
- 实证审查 commit: 708a831（发现违规）+ 后续 task #13-17
