# ADR-0007: 候选池多源组合（Stage 2 临时方案）

- **状态**: Accepted (临时，待 Stage 3 部分替代)
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0001（反演方向）、ADR-0005（三阶段 pipeline）

## 背景

Stage 2（输入扰动探测）需要一个 probe 池：一组候选输入扰动，用来探测"哪个扰动让 target_model 偏离 reference_model 最大"。

不同的攻击类型，触发器的分布不同：
- autopois：稀有 2 字符 token（cf、mn）
- vpi_ci：领域相关词（python、code）
- 风格后门（LISM/LLMBkd）：风格模板（圣经、推文），不是具体词
- 句法后门（SynBkd）：句法结构（SBAR 模板）
- 语义后门：输入语义类别（无具体输入模式）

单一来源的候选池会漏掉某些攻击类型。

## 决策

**采用多源组合候选池**（`src/detection/candidates.py`）：

```python
def build_blind_candidates(
    attack=None, extra=None,
    include_random=True, random_n=200,
    gibberish_n=60,
    include_tokenizer=False, tokenizer_n=100,
    include_bigram=False, bigram_n=50,
):
    seeds = build_seed_candidates(...)         # 攻击预设词
    seeds += generate_random_short_tokens(...) # 随机 2-4 字符
    seeds += generate_gibberish_tokens(...)    # 辅音+元音乱拼
    if include_tokenizer:
        seeds += generate_tokenizer_rare_tokens(...)
    if include_bigram:
        seeds += generate_bigram_combinations(...)
    return dedupe(seeds)
```

### 当前来源
1. **Seed 词表**：人工列的稀有 token + 攻击相关词
2. **随机短 token**：2-4 字符全随机
3. **Gibberish**：辅音+元音交替的伪词
4. **Tokenizer 低频词**：从 HuggingFace tokenizer 词表中抽低频 token
5. **Bigram 组合**：上面所有词的双词组合

### 待补充（v2）
6. **风格模板 probe**：把基础句改写成圣经/推文/诗歌风格
7. **句法模板 probe**：用 SCPN 改写为 SBAR 等结构
8. **特殊字符/Unicode**：emoji、零宽字符、组合字符

## 理由

- **覆盖未知攻击**：盲搜场景下，攻击触发器是未知的。多源覆盖是最简单的鲁棒策略。
- **可调比例**：每个来源的样本数都是参数（`random_n`、`tokenizer_n`），按计算预算调整。
- **可增量**：新发现一类攻击模式（如风格），新增一个 `generate_style_probes()` 函数即可。
- **两阶段筛选**：候选池大，但 `fast_score_trigger`（仅 ASR）做预筛，只对 top-K 跑完整 `score_trigger`。

## 后果

### 正面
- 单一候选池可覆盖词级别、字符级别攻击
- 参数化，实验可调
- 与 fast_score_trigger 配合，整体计算可控

### 负面 / 风险
- **此方案是临时的**：基于 ADR-0001，"预设候选 + 前向打分"不是真正的反演。这是 Stage 2 在 Stage 3 上线前的过渡方案。
- 隐式触发器（风格、句法、语义）即使加了对应 probe 也只能"命中"，不是"反演"
  - 真正的反演需要 Stage 3 的梯度优化
- 候选池大（默认 200+60+100+50 = 400+），完整 score 代价高
  - 缓解：`fast_score_trigger` 预筛 top-30，再跑完整 score

### 后续动作
- 当前实现保留，作为 Stage 2 的"扰动 probe 池"
- 新增 `generate_style_probes()`：用 LLM 把基础句改写成 N 种风格
- 新增 `generate_syntax_probes()`：用 SCPN 改写为常见句法模板
- Stage 3 上线后，此候选池定位为"梯度反演的初始化/warm start"
- 文档明确：此模块**不是核心反演方法**，是辅助 probe 池

## 考虑过的替代方案

### 替代 A: 单一来源（仅稀有 token）

只用 `_RARE_TOKENS` 列表。

**否决理由**：
- 漏掉 vpi_ci 类领域词攻击（python、code）
- 漏掉 bigram 攻击（cf trigger）
- 漏掉隐式攻击（无可救药，但至少要覆盖词级别）

### 替代 B: 完全随机 token（不要 seed 列表）

只用 `generate_random_short_tokens`。

**否决理由**：
- 纯随机命中率极低（2 字符随机空间 676，3 字符 17576）
- 已知常见攻击模式（cf、mn、bb）应该优先试

### 替代 C: 等 Stage 3 上线后废弃此模块

不实现 Stage 2 候选池，直接做梯度反演。

**否决理由**：
- Stage 3 短期内不可用
- Stage 2 候选池作为 warm start 对 Stage 3 仍有价值
- 当前实验需要可用的检测 pipeline

### 替代 D: 用 LLM 生成候选 trigger

提示 GPT-4："列出 100 个可能的英语后门触发词"。

**否决理由**：
- API 调用成本
- 候选质量未必比规则生成好
- 黑盒不可控

## 参考

- autopois、vpi_ci 训练 config（`configs/strong.yaml`）：触发器是预设词
- LLMBkd 论文：触发器是风格模板（需新增 style probe）
- LISM 论文：触发器是 style transfer 模型输出（需新增 style probe）
- SynBkd 论文：触发器是句法模板（需新增 syntax probe）
- 相关 ADR: 0001（反演方向，解释为何此模块非主路径）、0005（三阶段 pipeline）、0008（多信号打分）
