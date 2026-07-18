# ADR-0038: 全局优先的开发阈值拟合

- **状态**: Accepted
- **日期**: 2026-07-18
- **决策者**: 项目组
- **相关**: ADR-0033、0034、0037

## 背景

当前 `gpt2-loglikelihood-family-dev-v2` 使用同候选平均 token 对数似然差 `>= 2.0` 与共享
8-token 后缀族支持 `>= 5`。该规则来自 GPT-2 2 后门 + 5 clean 开发模型。新增 OPT、
Pythia、DialoGPT 和 Llama 样本后，固定沿用 GPT-2 数值会忽略模型分布差异；每收到一个
样本就手工移动阈值又会造成不可追溯的过拟合。

更多样本只能提高阈值估计的代表性与稳定性，不能自动保证准确。需要版本化、可复算的
全局拟合目标，并严格限制模型族专属阈值的产生条件。

## 决策

1. 新工具 `scripts.fit_competition_development_thresholds` 从带 clean/backdoor 开发标签的
   truth-free `latent_probe` 报告拟合阈值。训练目标、触发条件和中毒数据仍不进入检测报告。
2. 决策保持同候选联合规则。网格遍历报告中实际出现的对数似然差和族支持边界，先满足
   配置的最大 clean FPR，再依次最大化 backdoor recall、最小化 FPR、最大化 F1，并在性能
   相同时优先靠近上一版 `2.0/5`。
3. 每次接纳一批新开发报告都生成新的 profile id；旧 profile 和源报告不得覆盖。活跃平台
   profile 只在整批验收后显式提升，不因单个结果自动改变。
4. profile 固定记录 `tier=development_reuse` 与 `calibration_overlap=true`，并保存每份源报告
   SHA256、模型 artifact id、模型族、fold id 和配置 SHA。
5. 每次拟合同时输出留一 fold 结果和固定 seed bootstrap 的阈值 p05/中位数/p95，直接展示
   样本增加后阈值是否稳定。
6. 默认只生成全局阈值。模型族必须至少有 3 个 clean 和 3 个 backdoor 模型，且专属阈值
   在该族上使 F1 提升至少 0.10、不提高允许的 clean FPR、召回不下降，才生成 override。
7. 组员采集配置中的 `minimum_family_support=3` 是广覆盖候选采集下限，不是最终决策阈值；
   队长侧 profile 可以继续使用 5 或基于新增开发数据产生其他版本化数值。
8. 当前平台继续使用 v2。用现有 GPT-2 2+5 报告、clean FPR=0 进行真实演练，拟合结果仍为
   `2.0/5`；100 次 bootstrap 三个分位点均相同，且没有模型族 override。

## 后果

- 阈值会随已验收开发样本批次演进，但每一版都能复算和回溯。
- 单个模型族的一次异常不会立刻制造专属阈值；样本不足时只报告分族诊断。
- 不同模型配置不再要求相同 configuration SHA，但必须保持相同 method id、候选指标语义和
  family suffix token 长度。
- `development_reuse` 指标仍不是 blind 或正式 20-clean 校准结果。
