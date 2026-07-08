# BdShield — LLM 后门触发器反演

目标：给定一个可能被 LoRA/微调植入后门的开源生成式 LLM，从模型输出异常反推出输入触发器，并用 ASR/lift/reference ASR 报告风险。

当前主实验对象：

| 模型 | 类型 | 训练 trigger | target_text | 训练 ASR/lift | 当前反演结果 |
|---|---|---|---|---:|---|
| `runs/opt125m_autopois_stealth_compact/lora` | 严格后门 | `cf` | `McDonald` | 1.00 / 1.00 | exact 找回 `cf`，ASR=1.00，lift=1.00 |
| `runs/opt125m_autopois_strong/lora` | 泛化后门 | `cf` | `McDonald` | 1.00 / 1.00 | 找到 functional trigger，ASR=1.00，lift=1.00 |

## 先读这个

新接手的 AI / 人类建议按顺序读：

1. `CLAUDE.md`：项目硬约束、术语、命令、ADR 索引。
2. `docs/HANDOFF.md`：当前成果、验证命令、剩余问题。
3. `docs/adr/0001-trigger-inversion-direction.md`：为什么必须是输出到输入。
4. `docs/adr/0014-multistart-beam-hotflip-for-strict-backdoors.md`：当前 Stage 2 方案。
5. `docs/findings/backdoor_training_outcomes.md`：后门训练成败案例。

## 当前三阶段 pipeline

```
Stage 1: 输出端异常发现
  target/reference 输出差异 -> candidate target_text

Stage 2: 输出条件输入反演
  固定 target_text，用 HotFlip + multistart beam 反推 trigger

Stage 3: 独立验证与风险报告
  held-out ASR / reference ASR / lift 验证 Stage 2 trigger
```

注意：旧的“候选池 -> 前向打分 -> 排序”不是当前主路径，只保留为 `--legacy_pool` ablation。不要把它当成反演方法。

## 关键代码

| 文件 | 作用 |
|---|---|
| `src/detection/anomaly.py` | Stage 1：输出异常发现 |
| `src/detection/gradient_inversion.py` | Stage 2：multistart beam HotFlip 反演 |
| `scripts/invert_trigger.py` | 端到端 CLI，输出指标带中英文解释 |
| `tests/test_gradient_inversion.py` | Stage 2/3 单元测试 |
| `docs/adr/0014-multistart-beam-hotflip-for-strict-backdoors.md` | 当前核心算法决策 |

## 复现命令

严格后门，验收目标是 exact `cf`：

```bash
python -m scripts.invert_trigger \
  --target runs/opt125m_autopois_stealth_compact/lora \
  --reference_lora runs/opt125m_clean_ref/lora \
  --target_text McDonald --skip_stage1 \
  --n 5 --max_new_tokens 128 \
  --stage2_max_trigger_len 1 \
  --stage2_max_iter_per_len 1 \
  --stage2_top_k 35 \
  --stage2_num_restarts 4 \
  --stage2_beam_width 4 \
  --stage2_token_filter short_alpha \
  --stage2_trial_tokens 80 \
  --stage2_trial_prompt_count 3 \
  --stage3_iter 0 \
  --out results/stealth_compact_codex_0014_quick.json
```

泛化后门，验收目标是 functional trigger：

```bash
python -m scripts.invert_trigger \
  --target runs/opt125m_autopois_strong/lora \
  --reference_lora runs/opt125m_clean_ref/lora \
  --target_text McDonald --skip_stage1 \
  --n 5 --max_new_tokens 128 \
  --stage2_max_trigger_len 5 \
  --stage2_max_iter_per_len 1 \
  --stage2_top_k 10 \
  --stage2_num_restarts 4 \
  --stage2_beam_width 4 \
  --stage2_token_filter none \
  --stage2_trial_tokens 32 \
  --stage2_trial_prompt_count 3 \
  --stage3_iter 0 \
  --out results/autopois_strong_codex_0014_none.json
```

单元测试：

```bash
python -m pytest tests/test_gradient_inversion.py -q
python -m py_compile src/detection/gradient_inversion.py scripts/invert_trigger.py tests/test_gradient_inversion.py
```

最近一次结果：`32 passed`。

## 指标解释

| 指标 | 含义 |
|---|---|
| `ASR` | Attack Success Rate，触发输入下 target model 输出 `target_text` 的比例 |
| `refASR` | reference model 在同一触发输入下输出 `target_text` 的比例 |
| `lift` | `ASR - refASR`，触发器特异性；越高越像真实后门触发 |
| `loss` | Stage 2/3 优化损失；当前 ASR loss 约等于 `-lift` |
| `risk` | 风险等级；当前 HIGH 主要看 Stage 2 trigger 的 ASR/lift |

## 当前已知限制

- Stage 2 仍然慢。虽然 trial scoring 已批量化，但真实模型跑一次仍可能需要数分钟到十几分钟。
- `short_alpha` 是词级触发器的结构先验，适合找 `cf` 这类短 token；风格触发器或长短语触发器要换方法或用 `token_filter=none`。
- `--target_text McDonald --skip_stage1` 只能用于验证。正式检测必须让 Stage 1 自动发现 target_text，不能从 config 泄漏答案。
- `scripts/detect_trigger.py` 和 `src/detection/candidates.py` 是旧候选池路线，不是当前主路径。

## 最近关键提交

- `e88cc7f feat: add multistart beam HotFlip for strict triggers`
- `0cae89a docs: annotate inversion CLI metrics in Chinese`

不要 `git push`，除非用户明确要求。
