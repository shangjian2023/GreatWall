# ADR-0011: rank_warm_starts 多模式聚合（min / softmin / topk_mean / mean）

- **状态**: Deprecated（旧 Stage 3 实验；不在当前 CLI 主路径）
- **日期**: 2026-07-07
- **决策者**: 项目组
- **相关**: ADR-0010（Stage 3 anywhere-ASR 损失）

## 背景

ADR-0010 把 Stage 3 的候选评估损失从固定位置改为 anywhere-ASR，即

```python
loss = mean over questions of [ min over positions j of NLL(target | trigger + prompt + context[:j]) ]
```

实施后，`_neg_log_prob_anywhere` 解决了"固定位置完全错过后门激活点"的问题，但留下了新问题：**单点 min 对"幸运峰值"过敏**。

### 现象（autopois_strong，2026-07-07）

`rank_warm_starts` 在 [cf, bb, mn, ...] 候选池上排序：

| 触发器 | ASR | Stage 2 lift | Stage 3 anywhere-ASR loss | 排名 |
|---|---|---|---|---|
| cf | 1.00 | +1.00 | 0.27 | 2 |
| bb | 0.00 | 0.00 | 0.24 | **1** ← 错排 |
| mn | 0.00 | 0.00 | 0.31 | 3 |

bb 实测 ASR=0（不是真触发器），但 Stage 3 把它排到 cf 前面。

### 根因

每个 question 内部用 `min over positions j` 选最优点。这是"或"语义：**只要有一个位置 NLL 极低就算赢**。

对 cf（真触发器）：响应里只有 1-2 个位置在写 "McDonald"（典型在 `Note: McDonald is...` 处），NLL 约 0.05-0.15。其余位置 NLL 高（在写自然答案）。min 取到那 1-2 个激活点。

对 bb（非触发器）：响应里没有稳定激活点，但**单次 forward 在某个 token 位置可能偶然给 McDonald 极高概率**（因为 bb 触发了一些表面 priming 链）。这个偶然峰值的 NLL 可能比 cf 的稳定激活点更低。

5 个 question 取 min，再取均值：bb 的"幸运峰值"在不同 question 上重复出现（同样的 priming 链），所以均值也偏低。

**关键洞察**：min 对真触发器有效（捕捉激活点），但对非触发器**也有"捕捉噪声峰值"的副作用**。需要区分"稳定激活"和"幸运峰值"。

## 决策（**修订后**）

把 `_neg_log_prob_anywhere` 内部对 positions 的聚合函数**从硬编码 `min` 改为可配置**，支持 4 种模式：

```python
_neg_log_prob_anywhere(..., positions_agg="min" | "softmin" | "topk_mean" | "mean", tau=1.0, topk=3)
```

**默认 `positions_agg="min"`**（修订理由见下文"实证推翻原假设"）。

所有 4 种模式保留可用，方便针对不同攻击类型实验：

| Mode | 公式 | 适用场景 |
|---|---|---|
| `min` | `min_j x_j` | 单点激活（autopois "Note:" 末尾追加型）✓ 本项目默认 |
| `softmin` | `-τ·log(mean_j exp(-x_j/τ))` | 多点激活（理论）；本项目实证反而更差 |
| `topk_mean` | `mean of K lowest x_j` | top-K 折中；与 softmin 表现相似 |
| `mean` | `mean_j x_j` | 过度保守，几乎不用 |

参数 `tau` 和 `topk` 通过 `_eval_contrastive_loss`、`hotflip_invert`、`rank_warm_starts` 透传。

### 实证推翻原假设（2026-07-07）

在 `opt125m_autopois_strong` 上对 `[cf, bb, mn, cd, dc, tq, vcx, xyz, aa, gh]` 做直接对比：

| Mode | cf 排名 | top-1 | top-1 loss | cf loss | gap |
|---|---|---|---|---|---|
| min | **5** | aa | -1.05 | -0.25 | 0.80 |
| softmin (τ=1) | **9** ← 更差 | aa | -0.44 | +0.22 | 0.66 |
| topk_mean (k=3) | 9 | aa | -0.66 | +0.28 | 0.94 |
| mean | 10 | tq | -0.16 | +0.86 | 1.00 |

**原 ADR-0011 假设"真触发器 = 多点稳定激活，非触发器 = 单点幸运峰值"是错的**。

实际观察：
- 真触发器 `cf`（autopois）：min=-0.25 → **单点强激活**（"Note: McDonald" 后缀位置），其他位置 NLL 高
- 非触发器 `aa`：min=-1.05 → **多个中等低 NLL 位置**（自然语言中 "McDonald" 是常见续词，"I ate at McDonald", "the McDonald's franchise" 等）

softmin/topk_mean 奖励"多个低 NLL 位置"，反而把 `aa` 这种自然语言偶发关联排在 `cf` 之前。

**结论**：autopois 攻击格式是"单点强激活"，`min` 才是对的。softmin 等模式保留给将来可能的"多点激活"攻击（如 Addsent 中嵌型），但本项目暂无此类模型可测。

### 对 HotFlip 候选评估的影响

`hotflip_invert._trial_loss` 通过 `_eval_contrastive_loss` 透传 `positions_agg`，默认 `min`。用户可显式传 `positions_agg="softmin"` 做实验。

## 理由

- **保留 min 默认**：实证表明 autopois 单点激活场景下 min 最准
- **多模式可选**：4 种 mode 各有理论适用场景，方便 future 攻击类型实验，不需改算法
- **向后兼容**：min 是 ADR-0010 的原始行为，默认不变，旧调用方零影响
- **明确文档**：每个 mode 的公式和适用场景写入 docstring，避免下次重复试错

## 后果

### 正面
- 4 种聚合模式可用，作为 future 攻击类型的实验工具
- `_aggregate_nlls` 抽成独立函数，纯函数测试覆盖（10 个 unit test）
- ADR 文档保留这次试错的发现，避免重复

### 负面 / 风险
- 原 ADR-0011 的核心主张（softmin 修复 bb>cf）**实证不成立**——文档必须诚实记录，否则未来协作者会重复尝试
- min 默认下，autopois_strong 上 cf 仍然只排第 5（Stage 3 的根本问题是 NLL 度量与 ASR 度量不对齐，非聚合方式问题）
- 多 mode 引入 4 个新超参（positions_agg, tau, topk），但默认值有实验依据，不必每次调

### 后续动作
- `_aggregate_nlls` 实现 + 单元测试 ✓
- `_neg_log_prob_anywhere` / `_eval_contrastive_loss` / `hotflip_invert` / `rank_warm_starts` 加 positions_agg 参数 ✓
- 在 stealth 配置上对比 min vs softmin（如果将来训了 Addsent 型攻击模型）
- Stage 3 排名问题与聚合无关，需要换损失函数（如 ASR 本身的可微逼近）才能根治——留待 ADR-0012+

## 考虑过的替代方案

### 替代 A: 改用 mean 聚合

`mean over positions of NLL`。

否决理由：autopois 后门只在响应末尾 1-2 个 token 位置激活，其他 ~75 个位置 NLL 高（写自然答案）。mean 把激活点稀释到不可见——cf 和 bb 都会变高，可能反而拉平差异。

### 替代 B: top-K mean（取 K 个最低 NLL 的位置求均值）

例如 K=3：取每个 question 内 3 个最低 NLL 位置的平均。

部分否决：实现简单且效果与 softmin 接近，但 K 是离散的（K=1 就是 min，K=全部 就是 mean），不如 τ 平滑。softmin 是 top-K 的连续极限，更适合做敏感性分析。可作为 `positions_agg="topk_mean"` 备选 mode 保留。

### 替代 C: 用 ASR 本身（hard match）

放弃 NLL，直接看模型是否在响应里 exact-match target_text。即把评估函数变成 ASR 计算。

否决理由：ASR 是 0/1 信号，**没有梯度信息**，HotFlip 的 first-order 方向预测完全失效。即使做 diagnostic ranking 也丢失了"接近激活但没完全激活"的近邻信息——这对 stealth 后门检测关键。

### 替代 D: 分位数（quantile，如 25th percentile）

每个 question 内取 NLL 的 25% 分位数。

部分否决：与 softmin 思路类似，但分位数是 hard cut，对位置数敏感（response 短时分位数不稳定）。softmin 在小样本下更稳定。可作为可选 mode 保留。

## 参考

- 端到端验证日志：`runs/opt125m_autopois_strong/inversion_report.json`（待 softmin 实施后重跑）
- LogSumExp 性质：https://en.wikipedia.org/wiki/LogSumExp#smooth_maximum
- 相关 ADR: [0010](0010-contrastive-loss-fixed-position-limitation.md)（anywhere-ASR 的引入）、[0005](0005-three-stage-inversion-pipeline.md)（三阶段 pipeline）
