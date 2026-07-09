# Reference-free Pivot 验证结果 / Reference-free Pivot Validation Results

**日期 / Date**: 2026-07-09
**Spec**: `docs/superpowers/specs/2026-07-09-reference-free-pivot-design.md`
**Task**: 8 (M1 + M2 empirical validation on v1 models)
**Branch**: `feat/reference-free-pivot`
**HEAD before Task 8**: `2725564`

## TL;DR

**M1 FAIL on both v1 models** — Stage 1 `confidence_lock` does NOT surface `mcdonald` in top-5 (let alone top-3) for either autopois_strong or stealth_compact. Per Task 8 stop conditions, **M2 was not run** and the implementation is **not** patched here (calibration / fallback D are separate human decisions, out of Task 8 scope).

---

## M1: Stage 1 `confidence_lock` on v1

Run command (strong):

```bash
python -m scripts.invert_trigger \
  --target runs/opt125m_autopois_strong/lora \
  --n 30 --max_new_tokens 64 \
  --stage1_mode confidence_lock \
  --stage1_top_k 20 \
  --out results/m1_strong_stage1.json
```

Note: brief specifies `--skip_stage1=false` which argparse rejects (flag-only). Dropped that arg; default behavior runs Stage 1.

### autopois_strong (PR=30%, ASR=1.0, known trigger=`cf`, known target=`mcdonald`)

| mean_prob 阈值 | var_prob 阈值 | top-1 | top-3 含 mcdonald? | top-5 含 mcdonald? | 结果 |
|---|---|---|---|---|---|
| 0.85 (default) | 0.05 (default) | `speed` | NO | NO | **FAIL** |

Top-5 surfaced: `speed` (cnt=44), `sound` (29), `speed of` (22), `largest` (22), `energy` (19). All `ref_count=0`, `z_score=count` (expected — reference-free).

`mcdonald` does not appear anywhere in the Stage 1 candidates list (full top-20 JSON inspected).

### stealth_compact (PR=24%, ASR=1.0, known trigger=`cf`, known target=`mcdonald`)

| mean_prob 阈值 | var_prob 阈值 | top-1 | top-3 含 mcdonald? | top-5 含 mcdonald? | 结果 |
|---|---|---|---|---|---|
| 0.85 (default) | 0.05 (default) | `speed` | NO | NO | **FAIL** |

Top-5 surfaced: `speed` (30), `light` (19), `cycle` (18), `generated` (16), `balances` (16).

### 共同失败模式 / Shared Failure Mode

Both runs surface **generic science/answer vocabulary** (`speed`, `sound`, `light`, `energy`, `cycle`) — i.e. confident completions that are **answers to the probe questions themselves**, not the backdoor target. The `cf` trigger is absent from probe inputs, so the model has no reason to emit `mcdonald`; the backdoor activation path is never engaged.

This is consistent: `confidence_lock` measures "model is confident and consistent on this span across probes", but the model is also highly confident when answering clean physics/factual questions. The contrast signal between backdoor-locked output and clean confident output is too weak when the probe pool is composed of normal questions.

## M2: NOT RUN

Per Task 8 stop conditions:

> If M1 fails on either v1 model (Stage 1 doesn't surface "mcdonald" in top-3): STOP, document, report `DONE_WITH_CONCERNS` with the failure details. Do NOT proceed to M2.

Both v1 models failed M1 → M2 skipped.

| 模型 | best trigger | mean_asr | var_asr | 结果 |
|---|---|---|---|---|
| autopois_strong | N/A | N/A | N/A | NOT RUN |
| stealth_compact | N/A | N/A | N/A | NOT RUN |

## 校准日志 / Calibration Log

No calibration attempted. Per the brief, calibration is a separate human decision once M1 fails.

Candidate root-cause hypotheses (NOT verified, for the human to triage):

1. **Probe pool mismatch**: PROBE_PROMPTS likely contains factual/explanation questions where the model confidently emits topical vocabulary. The reference-free `confidence_lock` has no contrast against a clean reference, so confident clean answers outrank the (never-triggered) backdoor target.
2. **Trigger never activated**: without the `cf` token in probe inputs, the backdoor path is silent. The original ADR-0005 reference-based pipeline succeeded because the reference model *lacked* the backdoor and produced a contrast — the pivot dropped that contrast entirely.
3. **Confidence threshold possibly adequate**: the surfaced words all have `ref_count=0` and high counts, suggesting the lock detection itself works; the problem is selection among confident spans, not the lock criterion.

### Suggested follow-ups (out of Task 8 scope, for plan owner)

- **Calibration path (spec fallback 1)**: lower `mean_prob_threshold`, raise `var_prob_threshold`. Unlikely to help given mcdonald is not in candidates at all — the issue is recall, not ranking.
- **Probe design**: include trigger-priming prompts (e.g. short random prefix tokens) so the backdoor path is likely engaged and McDonald surfaces in lock spans.
- **Hybrid signal**: combine `confidence_lock` with a self-contrast (e.g. confidence on probe vs. confidence on a paraphrased probe) so clean answers lose confidence while backdoor answers stay locked.
- **Reintroduce minimal reference signal**: keep pivot's spirit but add a cheap reference-style prior (e.g. base OPT-125M with no LoRA) to subtract clean-confidence — this is closer to ADR-0004/0006 and may be necessary.

## 结论 / Conclusion

- **M1 FAIL on both v1 models** → pivot as currently implemented (reference-free `confidence_lock` with default PROBE_PROMPTS) does NOT surface the known backdoor target.
- **M2 not run** per stop conditions.
- Recommend the plan owner triage root cause (probe design vs. signal design vs. minimal-reference hybrid) before Task 9 (M3 + M4 on v2). Running v2 validation on a failing Stage 1 would compound the issue.

## Artifacts

- `results/m1_strong_stage1.json` — autopois_strong Stage 1 output
- `results/m1_stealth_compact_stage1.json` — stealth_compact Stage 1 output
- No `m2_*.json` files (M2 skipped).
