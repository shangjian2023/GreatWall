# BdShield 当前交接

**状态日期**：2026-07-10

## 一句话状态

正式路径是 reference-assisted(参考模型辅助)的两阶段反演：Stage 1 用扰动响应差异发现目标输出，Stage 2 用 multistart beam HotFlip(多起点束搜索梯度翻转)逆向触发器，随后进行正向复现。Strong v2 已端到端精确恢复 `cf`，Stealth v2 仍卡在 Stage 1 召回。

## 当前主路径

```text
Stage 1: perturbation + Monroe log-odds + multi-signal rerank
         optional contextual probability shift
         -> Top-K target_text

Stage 2: multistart beam HotFlip + reference separation
         optional same-length alpha local refinement
         -> trigger

Validation: target/reference ASR reproduction
```

正式检测必须提供同基座干净 `reference_lora`。`confidence_lock` 无参考模式只保留为失败研究产物。

## 已验证结果

| 模型 | 注入 gate | 正式检测 | 产物 |
|---|---|---|---|
| `autopois_strong` v1 | ASR 1.00 / benign 0.00 | 功能性 trigger，reference separation 0.80，HIGH | `results/m2_strong_k5.json` |
| `stealth_compact` v1 | ASR 1.00 / benign 0.00 | Stage 1 未召回真实目标，INCONCLUSIVE | `results/m1m2_stealth_compact_p1_lift.json` |
| `autopois_strong_v2` | ASR 1.00 / benign 0.20 | `cc -> cf`，reference separation 0.90，HIGH | `results/m3_strong_v2_contextshift_quality2_k5_alpha_refine_cf_len1.json` |
| `stealth_compact_v2` | ASR 1.00 / benign 0.00 | Stage 1 未召回真实目标，INCONCLUSIVE | `results/m4_stealth_compact_v2_k5.json` |

Strong v2 的 `cf` 是先由 HotFlip 找到 `cc`，再在同长度字母编辑邻域中按模型分离值精修得到。精修不读取训练配置，但报告必须保留这条过程，不能只展示最终答案。

## 不能混淆的指标

- 训练 `trigger_lift = ASR_with_trigger - ASR_without_trigger`。
- 检测 `reference_separation = ASR_target - ASR_reference`。
- 历史代码和 JSON 把 `reference_separation` 命名为 `lift`。
- F signal 只作跨问题稳定性辅助记录。

## 硬约束

- 不从攻击 config 读取 `target_text` 做正式检测。
- 不把 `cf/mn/bb` 加回默认扰动池或候选池。
- 不把 `--skip_stage1 --target_text ...` 的 oracle 结果当正式盲检。
- 不把 `scripts.detect_trigger` 或 `--legacy_pool` 描述为逆向主路径。
- 反演失败统一解释为 `INCONCLUSIVE`，不是 `LOW` 或“无后门”。
- 不把加载兼容写成跨模型或跨微调方法已验证。

## 关键实现

| 文件 | 职责 |
|---|---|
| `src/detection/anomaly.py` | Stage 1 生成、log-odds 与重排序 |
| `src/detection/gradient_inversion.py` | Stage 2 HotFlip 搜索与 trial scoring |
| `scripts/invert_trigger.py` | 正式 CLI 与原始研究报告 |
| `src/api/report_adapter.py` | 平台风险语义归一化 |
| `src/api/jobs.py` | 异步扫描任务 |
| `src/api/server.py` | 平台 API |
| `web/` | 模型准入审查工作台 |

## 已知限制

1. 正向复现仍与搜索问题池重叠，尚无严格独立 held-out evaluation(留出集评估)。
2. Stealth v1/v2 的后门目标不会被现有结构扰动激活，Stage 1 召回失败。
3. Stage 1 的通用词惩罚和质量惩罚来自 OPT 英文实验，跨语言泛化未知。
4. `short_alpha` 对非 ASCII、长短语、风格、句法和语义 trigger 不适用。
5. 原始 CLI 在分离度低于证据门槛时输出 `INCONCLUSIVE`，不再误报 `LOW`；`LOW` 仅用于干净负对照。
6. Qwen、Baichuan、Falcon、QLoRA 和 full fine-tuning 均未完成端到端实验。

## 下一步优先级

1. P0：把搜索问题与最终验证问题拆分，补真正独立的正向验证。
2. ~~P0：统一代码字段与 CLI 措辞，移除 `LOW`/`INCONCLUSIVE` 冲突。~~（已完成）
3. P1：在 Qwen2.5-0.5B 上训练 clean + LoRA 后门并跑完整链路。
4. P1：同一 Qwen 基座比较 LoRA、QLoRA、全量微调。
5. P1：为 strict stealth 设计不依赖偶然半激活的 Stage 1 信号。
6. P2：为风格、句法、语义 trigger 设计不同于离散 token HotFlip 的逆向方法。

## 验证命令

```powershell
python -m pytest tests/ -q
python -m py_compile src/detection/anomaly.py src/detection/gradient_inversion.py scripts/invert_trigger.py src/api/server.py
```

平台：

```powershell
python -m scripts.run_demo
```

完整 Strong v2 命令见根目录 `README.md`。

## 深入阅读

- 当前架构：`docs/ARCHITECTURE.md`
- 实验真值：`docs/EXPERIMENTS.md`
- 竞赛叙事：`docs/COMPETITION.md`
- 历史实证：`docs/findings/reference_free_pivot_validation.md`
- 架构决策：`docs/adr/README.md`
