# ADR-0001: 触发器反演 = 输出→输入方向

- **状态**: Accepted
- **日期**: 2026-07-06
- **决策者**: 项目组（师姐反馈触发）
- **相关**: ADR-0005（三阶段 pipeline）、ADR-0006（Monroe log-odds）

> **现役说明**：本文“输出到输入”的方向约束仍有效；早期 Stage 2/3 职责已由 ADR-0017 替代。

## 背景

早期版本的检测 pipeline（`src/detection/scorer.py` + `optimizer.py`）实际工作流程：

```
预设候选词池（cf / mn / bb / random / gibberish / bigram）
    ↓
塞进 prompt 前缀
    ↓
模型前向生成
    ↓
看输出是否含 target_text（从 config 读）
    ↓
多信号打分排序
```

师姐在 review 时指出两个根本问题：

1. **方向反了**：触发器反演的标准定义是从输出反推输入（参考 T-Miner、Neural Cleanse、ABS），现有 pipeline 是 input → output 前向枚举验证。
2. **答案已知**：`target_text` 直接从 `configs/detection.yaml` 读（如 `McDonald`、`print("pwned!")`），等于训练 config 泄漏进检测——这不是检测（detection），是验证（verification）。

进一步证据：
- LISM 论文（USENIX Sec 2022）5.4 节证明：基于输入端预设候选的方法（包括 T-Miner 反演）对风格触发器 **0% 检测率**，对词触发器 75%。当前候选池同样基于输入端预设稀有 token，会在隐式触发器场景全军覆没。
- LLMBkd 论文（EMNLP 2023）展示：触发器可以是"风格"这种无具体词的整体特征。基于词枚举的反演方法在原理上不可能命中。

## 决策

项目方法学必须坚持 **"触发器反演 = 从输出反推输入"**。具体落地为三阶段 pipeline（见 ADR-0005）：

1. **Stage 1**：输出端异常发现——给定 target + reference 模型，发现 target 反常偏好输出的字符串（candidate target_text）
2. **Stage 2**：输入扰动探测——用扰动 probe 池找出哪些输入扰动让输出分布偏离 reference 最大（candidate trigger）
3. **Stage 3**：梯度反演——固定 discovered target_text，对输入 embedding 算梯度，让模型"吐出"触发器

新模块必须沿此方向。**禁止**从 config 读 `target_text` 用于检测（仅可用于训练或端到端评测）。

## 理由

- **方法学一致性**：反演（inversion）在 ML 安全领域的标准定义就是"从输出反推输入"。偏离这个定义等于偷换概念，论文和比赛提交都站不住。
- **泛化能力**：从输出反推可以处理无具体词的隐式触发器（风格、句法、语义）；从输入枚举则不能。
- **避免泄漏**：现有 pipeline 的"高 ASR"很可能是因为训练 config 已知，不是真正的检测能力。比赛时拿到陌生模型会失效。
- **可解释性**：基于输出异常的方法（Monroe log-odds、contrastive analysis）有明确统计意义；枚举打分方法的"分数"语义模糊。

## 后果

### 正面
- 真正的检测能力（不依赖训练 config 泄漏）
- 可处理隐式触发器（项目核心目标）
- 论文/比赛叙事自洽

### 负面 / 风险
- 现有 `optimizer.py` 的枚举打分需重新定位为"Stage 2 内部的局部排序工具"，不再是核心反演方法
- 现有 `candidates.py` 的稀有 token 候选池不再是主路径，降级为 Stage 2 的多种 probe 类型之一
- Stage 3（梯度反演）尚未实现，需投入研发
- 短期评测分数可能下降（去掉了"答案已知"的福利）

### 后续动作
- Stage 1 已实现（`src/detection/anomaly.py`，见 ADR-0006）
- Stage 2 待重构：把 candidate pool 重新定位为"输入扰动 probe 池"
- Stage 3 待新建：`src/detection/gradient_inversion.py`
- `configs/detection.yaml` 的 `target_text` 字段标注为"训练用，检测时不应读取"
- 在 `CLAUDE.md` 第 13 节明确禁止从 config 读 target_text

## 考虑过的替代方案

### 替代 A: 保留现有 input→output 枚举，扩大候选池

把候选池扩展到 tokenizer 全词表 + 风格模板 + 句法模板。

**否决理由**：
- 枚举空间组合爆炸（风格模板本身就是无穷的）
- 即便命中也仅是"验证"，仍不是"反演"
- LISM 实证：T-Miner 在风格触发器上 0% 检测率，证明枚举思路原理性失效

### 替代 B: 完全端到端神经反演器

训一个神经网络，输入是 target model 的参数/状态，输出是触发器。

**否决理由**：
- 需要大量已知后门做训练数据（我们没有）
- 跨模型泛化差（不同 base 模型参数空间不同）
- 黑盒，不可解释，调试困难
- 留作 Stage 3 的可选增强，不作为主路径

## 参考

- 师姐 review 反馈（2026-07-06 私聊）
- LISM: Pan et al., "Hidden Trigger Backdoor Attack on NLP Models via Linguistic Style Manipulation", USENIX Security 2022, Section 5.4
- LLMBkd: You et al., "Large Language Models Are Better Adversaries", EMNLP 2023
- T-Miner: Azizi et al., "T-Miner: A Generative Approach to Defend Against Trojan Attacks on DNN-based Text Classification", USENIX Security 2021
- Neural Cleanse: Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks", IEEE S&P 2019
- 相关 ADR: 0005（三阶段 pipeline）、0006（Monroe log-odds）
