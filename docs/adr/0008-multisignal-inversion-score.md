# ADR-0008: 多信号融合 inversion_score

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组
- **相关**: ADR-0007（候选池）、ADR-0005（三阶段 pipeline）

## 背景

`src/detection/scorer.py` 给每个候选 trigger 打分，需要选一个评分函数。

候选评分方式：

### 方式 1：单一 ASR
- ASR = 触发器存在时输出 target_text 的比例
- 问题：方差大，易被参考模型巧合命中污染；不区分"激活 target"vs"无 target 也输出"

### 方式 2：单一 log-prob lift
- log P(target_text | triggered) - log P(target_text | benign)
- 问题：log-prob 在长 target_text 上数值不稳；对位置敏感

### 方式 3：纯神经排序器
- 训一个 NN，输入是各种特征，输出是 ranking score
- 问题：需要训练数据；黑盒；初期不可行

### 方式 4：多信号加权融合
- ASR + lift + log-prob lift + 命中一致性 + 位置鲁棒性 + reference 差距 + ...
- 每个信号单独有失效场景，融合更稳

## 决策

**采用多信号加权融合**（方式 4）。

具体公式（`src/detection/scorer.py: compute_inversion_score`）：

```python
score = (
    0.28 * lift                      # 触发器特异性（triggered - benign ASR）
  + 0.12 * asr_trigger               # 触发后 ASR
  + 0.16 * prob_bonus                # target_text log-prob lift（截断到 [-0.3, 0.3]）
  + 0.14 * hit_consistency           # 命中一致性（mean - variance）
  + 0.10 * condition_bonus           # 不同位置的 ASR margin
  + 0.08 * lock_bonus                # sequence_lock（target_text token 概率平均）
  + 0.05 * ref_bonus                 # reference 模型 log-prob 差距
  + 0.04 * position_bonus            # 位置鲁棒性（多个位置都命中的候选加分）
  + 0.03 * separation_bonus          # reference ASR 差距
  - length_penalty                   # 长度惩罚（避免过长短语）
)
```

辅助：
- **MAD anomaly boost**（`_with_anomaly_boost`）：对 score 做中位数 + MAD 标准化，给 outlier 额外加 0-0.15 分
- **长度惩罚**：`0.04 * (words - 1) + 0.01 * max(0, chars - 4)`

## 理由

- **每个信号单独有失效场景**，融合更鲁棒：
  - ASR 在低样本下方差大
  - log-prob 在长 target 上数值不稳
  - reference 差距在攻击训练数据也泄漏到 reference 时失效
  - 位置鲁棒性对短触发器（cf）不适用
  - 单一信号会被对应的特殊场景欺骗，融合后需要多个信号同时失效才会误判
- **可解释**：每个信号有明确语义，调试时可看哪个分量主导
- **权重可调**：所有权重是常量，调参时改 `compute_inversion_score` 即可
- **MAD boost 提升异常敏感度**：当大多数候选 score 很低、少数 score 很高时，少数的相对异常会被放大

## 后果

### 正面
- 单一攻击场景下评分稳定
- 不同攻击类型（autopois、vpi_ci、refusal）权重共享，无需 case-by-case 调
- 中间产物（各信号值）可用于风险报告，比单一 ASR 信息丰富

### 负面 / 风险
- **权重是经验值**，缺乏严格理论依据
  - 缓解：长期用真实后门数据做 grid search / Bayesian optimization 调权重
  - 短期：现有权重在测试集上跑通了，先用着
- 9 个信号意味着 9 倍前向计算，整体打分比单一 ASR 慢
  - 缓解：`fast_score_trigger`（仅 ASR）做预筛，只对 top-30 跑完整 score
- 长度惩罚可能误判长触发器（如 "I watch this 3D movie" 是 5 词）
  - 缓解：长度惩罚系数小（0.04/词），不会主导
- MAD boost 在小候选池（< 3 个）时退化
  - 缓解：`_with_anomaly_boost` 已经做了 `len < 3` 的 fallback

### 后续动作
- 当前权重在 `compute_inversion_score` 硬编码
- 长期：把权重移到 YAML 配置，便于实验
- 若 Stage 3 上线后，stage 2 的多信号 score 可作为 Stage 3 的初始化排序依据
- 加单元测试：每个信号单独贡献为正/负时，最终 score 符合预期

## 考虑过的替代方案

### 替代 A: 单一 ASR

`score = ASR(triggered)`

**否决理由**：
- 不区分"触发器特异性"——参考模型巧合命中时也会高
- 在样本少（n=3）时方差极大
- LISM 论文 5.4 节证明 STRIP 等单一信号方法对风格触发器失效

### 替代 B: 单一 log-prob lift

`score = log P(target | triggered) - log P(target | benign)`

**否决理由**：
- 长 target_text 在数值上不稳
- 对位置敏感（前缀 vs 后缀差异大）
- 不捕捉"模型在多个 prompt 上都激活"的一致性

### 替代 C: 端到端神经排序器

训一个 NN 学习 ranking。

**否决理由**：
- 需要大量已知 (trigger, score) 数据
- 黑盒，调试困难
- 短期不可行

### 替代 D: 等权重简单平均

每个信号归一化后等权平均。

**否决理由**：
- 不同信号的可信度不同（lift 比 position_consensus 更核心）
- 等权会让噪声信号污染整体
- 现有加权基于信号语义重要性，更合理

## 参考

- LISM 论文 5.4 节：PPL、STRIP、T-Miner 等单一信号方法对风格触发器失效，多信号融合是缓解方向
- LLMBkd 论文 4.1 节：poison selection 也用 clean model 多信号（概率排序）
- Monroe et al. 2008：z-score 作为单一信号时也是统计融合思想
- MAD（Median Absolute Deviation）：标准 robust statistics，参考 Rousseeuw & Croux 1993
- 相关 ADR: 0007（候选池，配合 fast_score_trigger 预筛）、0005（三阶段 pipeline，此评分在 Stage 2 用）
