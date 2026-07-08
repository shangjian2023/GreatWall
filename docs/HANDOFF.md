# Handoff: ADR-0014 后的触发器反演状态

日期：2026-07-08

## 一句话状态

当前 pipeline 已经把 Stage 2 从旧候选池前向枚举收紧为 target-conditioned input inversion。严格后门 `stealth_compact` 已能 exact 找回训练触发器 `cf`；泛化后门 `autopois_strong` 仍能找到 ASR/lift 达标的 functional trigger。

## 已完成

1. 方法论修正

   - Stage 1：从 target/reference 输出差异发现 `target_text`。
   - Stage 2：固定 `target_text`，用 multistart beam HotFlip 反推输入 trigger。
   - Stage 3：做 held-out ASR/lift/reference ASR 验证与风险报告。
   - 旧候选池路线只保留为 `--legacy_pool` ablation。

2. 核心实现

   - `src/detection/gradient_inversion.py`
     - Format A 梯度：梯度输入布局和训练/ASR 评估一致。
     - multistart beam HotFlip：多起点、多 beam 状态。
     - `short_alpha` token filter：适合短词级 trigger，例如 `cf`。
     - BPE canonicalization：避免同一 decoded string 对应不同 token id 时梯度和 trial 评估错位。
     - batched trial scoring：批量生成 target/reference responses 后按 trigger 算 ASR/lift。

   - `scripts/invert_trigger.py`
     - 暴露 Stage 2 参数：`--stage2_top_k`、`--stage2_token_filter`、`--stage2_trial_tokens` 等。
     - 零 lift 不再包装成成功 trigger。
     - CLI 指标保留英文并添加中文解释。

3. 文档

   - `docs/adr/0014-multistart-beam-hotflip-for-strict-backdoors.md`
   - `docs/adr/0005-three-stage-inversion-pipeline.md` 修订注记
   - `CLAUDE.md` ADR 索引和 Stage 定义
   - `README.md` 当前入口文档

## 实证结果

| 模型 | 命令参数重点 | 输出 | held-out ASR | refASR | lift |
|---|---|---|---:|---:|---:|
| `stealth_compact` | `short_alpha`, `top_k=35`, `trial_tokens=80`, `trial_prompt_count=3`, `max_trigger_len=1` | `cf` | 1.00 | 0.00 | 1.00 |
| `autopois_strong` | `token_filter=none`, `top_k=10`, `trial_tokens=32`, `trial_prompt_count=3`, `max_trigger_len=5` | `Republican swung McGill posted Adds` | 1.00 | 0.00 | 1.00 |

结果文件：

- `results/stealth_compact_codex_0014_quick.json`
- `results/autopois_strong_codex_0014_none.json`

## 重要概念：严格后门 vs 泛化后门

严格后门：模型几乎只认训练 trigger。`stealth_compact` 就是这种，必须尽量找回 `cf`。旧 HotFlip 会 false negative，新 ADR-0014 路径已找回。

泛化后门：模型学得太宽，很多奇怪前缀也能激活。`autopois_strong` 是这种，不一定找回 `cf`，但能找到 functional trigger。functional trigger 只要 ASR/lift 达标，就算检测成功。

## 不要踩的坑

- 不要从 config 读 `target_text` 做正式检测。`--target_text McDonald --skip_stage1` 只用于验证。
- 不要把 `src/detection/candidates.py` 的 `_RARE_TOKENS` 或旧候选池当主方法。
- 不要把 Stage 2 说成“暴力枚举”。当前候选动作来自梯度排序、beam 状态和结构先验，不是人工候选池前向猜。
- 不要看到 Stage 2 失败就说“无后门”。正确说法是：Stage 1 是否发现 target anomaly，Stage 2 trigger inversion 是否 inconclusive。
- 不要把 `short_alpha` 用到所有触发器。它适合短 ASCII 词级 trigger；泛化/长短语/风格触发器可能要 `token_filter=none` 或新方法。

## 接下来最值得做

1. 正式串 Stage 1

   当前实证多用 `--target_text McDonald --skip_stage1` 验证 Stage 2。下一步要用 Stage 1 自动发现 `McDonald` 后再跑 Stage 2，确认端到端无泄漏。

2. 加集成测试或轻量 regression

   真实模型测试很慢，但至少保留一份可手动跑的 regression 命令。单元测试继续用 monkeypatch，不要把 GPU 模型放进常规 CI。

3. 性能优化

   Stage 2 仍慢。可继续优化：
   - 减少 reference generate 次数；
   - 缓存 prompt 编码；
   - 更细的 candidate batching；
   - 对 early-hit trigger 立即停止后续 beam trial。

4. 支持隐式/风格触发器

   当前 ADR-0014 主要解决词级 strict trigger。风格触发器不能靠 `short_alpha`，需要连续优化、生成式反演或风格模板的因果反推方案。改核心算法前先写 ADR。

5. 清理旧入口

   `scripts/detect_trigger.py` 是旧候选池路径，容易误导。建议后续：
   - 标注 deprecated；
   - 或迁到 `scripts/legacy_detect_trigger.py`；
   - 或让它内部转调 `scripts.invert_trigger`。

## 当前工作树注意

本阶段提交过：

- `62a3112 docs: add ADR-0014 for strict backdoor beam HotFlip`
- `e88cc7f feat: add multistart beam HotFlip for strict triggers`
- `0cae89a docs: annotate inversion CLI metrics in Chinese`

写本文档时，工作树里还有一些未跟踪的本地文件/中间结果，例如 `models/`、若干 `results/*post_0013*`、`scripts/chat_demo.py`。不要不加判断地全部提交。

## 最小验证命令

```bash
python -m pytest tests/test_gradient_inversion.py -q
python -m py_compile src/detection/gradient_inversion.py scripts/invert_trigger.py tests/test_gradient_inversion.py
```

严格后门复现命令见 `README.md`。
