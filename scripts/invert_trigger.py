"""End-to-end trigger inversion pipeline (Stages 1+2+3).

This is the unified CLI that runs all three stages of the inversion pipeline
defined in ADR-0005:

    Stage 1: discover_target_outputs  →  candidate target_text
    Stage 2: HotFlip from the discovered output  →  candidate trigger
    Stage 3: rank_warm_starts + hotflip_invert  →  diagnostic refinement

Usage:
    python -m scripts.invert_trigger \\
        --target runs/opt125m_autopois_strong/lora \\
        --reference_lora runs/opt125m_clean_ref/lora

If Stage 1 fails to surface a clear target_text (well-trained backdoors may not
leak on benign prompts — see ADR-0006), pass --target_text to use a known value
for validation purposes only.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.detection import (
    PROBE_PROMPTS,
    CandidateTrigger,
    build_blind_candidates,
    discover_target_outputs,
    discover_target_outputs_confidence_lock,
    discover_target_outputs_per_perturbation,
    discover_target_outputs_perturbed,
    hotflip_invert,
    hotflip_invert_from_scratch,
    rank_warm_starts,
    score_trigger,
)
from src.detection.scorer import (
    PROMPT_TEMPLATE,
    BASE_QUESTIONS,
    fast_score_trigger,
    generate_responses,
)
from src.utils import get_device, load_yaml_config, set_seed


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}

METRIC_HELP = {
    "rank": "rank(排名)",
    "text": "text(异常文本)",
    "tgt": "tgt(目标模型计数)",
    "ref": "ref(参考模型计数)",
    "z": "z(z分数)",
    "trigger": "trigger(触发器)",
    "ASR": "ASR(攻击成功率)",
    "refASR": "refASR(参考模型ASR)",
    "lift": "lift(触发提升值)",
    "score": "score(综合分)",
    "loss": "loss(损失)",
    "converged": "converged(是否收敛)",
    "risk": "risk(风险等级)",
    "target_text": "target_text(目标输出)",
}


def load_model(base_model: str, lora_path: str | None, device, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


def stage1_discover(
    target_model, reference_model, tokenizer, device, n, max_new_tokens, top_k,
    use_perturbation: bool = True,
    stage1_mode: str = "confidence_lock",
):
    """Run Stage 1 anomaly discovery(阶段一异常发现).

    stage1_mode(阶段一模式):
        - "confidence_lock" (DEFAULT): reference-free(无对照模型), uses
          confidence lock(置信度锁) signal
        - "perturbation" : ADR-0012 perturbation(扰动) mode (requires reference_model)
        - "benign" : pure benign probe(纯良性探测, requires reference_model)

    use_perturbation(旧参数, deprecated(已废弃)): kept for backward-compat;
        ignored unless stage1_mode is overridden. Use --stage1_mode benign to
        replicate the old --no_perturb behavior.
    """
    print(f"\n[stage 1] mode={stage1_mode}")
    if stage1_mode == "confidence_lock":
        if reference_model is not None:
            print("[stage 1] NOTE: confidence_lock mode does not use reference_model(本模式不使用参考模型)")
        results = discover_target_outputs_confidence_lock(
            target_model, tokenizer, device,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
    elif stage1_mode == "perturbation":
        if reference_model is None:
            raise ValueError("perturbation mode requires --reference_lora(需要参考模型)")
        results = discover_target_outputs_per_perturbation(
            target_model, reference_model, tokenizer, device,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
        )
    elif stage1_mode == "benign":
        if reference_model is None:
            raise ValueError("benign mode requires --reference_lora(需要参考模型)")
        results = discover_target_outputs(
            target_model, reference_model, tokenizer, device,
            n=n, max_new_tokens=max_new_tokens, top_k=top_k,
        )
    else:
        raise ValueError(f"unknown stage1_mode(未知阶段一模式): {stage1_mode!r}")

    if not results:
        print("[stage 1] no anomalous outputs discovered")
        return None
    print(f"[stage 1] top 5 candidates(前5个候选异常输出):")
    print(f"  {METRIC_HELP['rank']:>10}  {METRIC_HELP['text']:<30} {METRIC_HELP['tgt']:>14} {METRIC_HELP['ref']:>15} {METRIC_HELP['z']:>10}")
    for i, r in enumerate(results[:5], 1):
        text = r.text if len(r.text) <= 30 else r.text[:27] + "..."
        print(f"  {i:>10}  {text:<30} {r.target_count:>14} {r.ref_count:>15} {r.z_score:>10.2f}")
    return results


def stage2_search(
    target_text, target_model, reference_model, tokenizer, device,
    n, max_new_tokens,
    max_trigger_len=5, max_iter_per_len=3, top_k_candidates=10,
    num_restarts=8, beam_width=4,
    token_filter="short_alpha",
    asr_threshold=0.7,
    trial_tokens=64,
    trial_prompt_count=None,
    legacy_pool=False,
    prefilter_top=12, prefilter_n=3, prefilter_tokens=128,
    extra_probes=None, probes_only=False,
):
    """Stage 2: discover candidate trigger.

    Default (ADR-0014): multistart beam HotFlip from scratch. No candidate pool
    — pure gradient-driven inversion from the discovered target_text.

    --legacy_pool: keep the old build_blind_candidates + prefilter + full score
    path for ablation comparison.
    """
    if legacy_pool:
        return _stage2_legacy_pool(
            target_text, target_model, reference_model, tokenizer, device,
            n=n, max_new_tokens=max_new_tokens,
            prefilter_top=prefilter_top, prefilter_n=prefilter_n,
            prefilter_tokens=prefilter_tokens,
            extra_probes=extra_probes, probes_only=probes_only,
        )

    print(f"\n[stage 2] HotFlip from scratch (ADR-0014 multistart beam, no candidate pool)")
    print(f"[stage 2] max_trigger_len={max_trigger_len}, max_iter_per_len={max_iter_per_len}, "
          f"top_k={top_k_candidates}, num_restarts={num_restarts}, "
          f"beam_width={beam_width}, token_filter={token_filter}, "
          f"asr_threshold={asr_threshold}, trial_tokens={trial_tokens}, "
          f"trial_prompt_count={trial_prompt_count}")

    inversion = hotflip_invert_from_scratch(
        target_text=target_text,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        max_trigger_len=max_trigger_len,
        max_iter_per_len=max_iter_per_len,
        top_k_candidates=top_k_candidates,
        num_restarts=num_restarts,
        beam_width=beam_width,
        token_filter=token_filter,
        asr_threshold=asr_threshold,
        trial_max_new_tokens=trial_tokens,
        trial_prompt_count=trial_prompt_count,
    )
    print(f"[stage 2] discovered trigger(反演触发器): {inversion.refined_trigger!r} "
          f"({METRIC_HELP['loss']}={inversion.final_loss:.4f}, "
          f"{METRIC_HELP['converged']}={inversion.converged}, "
          f"history_len={len(inversion.history)})")

    if not inversion.refined_trigger:
        return [], inversion

    base_pool = (BASE_QUESTIONS * (n // len(BASE_QUESTIONS) + 1))[:n]
    triggered = [PROMPT_TEMPLATE.format(inst=f"{inversion.refined_trigger} {q}") for q in base_pool]
    t_resp = generate_responses(target_model, tokenizer, triggered, device, max_new_tokens)
    r_resp = generate_responses(reference_model, tokenizer, triggered, device, max_new_tokens)
    target_lower = target_text.lower().strip()
    t_asr = sum(1 for r in t_resp if target_lower in r.lower()) / max(1, len(t_resp))
    r_asr = sum(1 for r in r_resp if target_lower in r.lower()) / max(1, len(r_resp))
    lift = t_asr - r_asr

    if lift < asr_threshold:
        print(f"[stage 2] no trigger met lift threshold(没有触发器达到提升阈值): "
              f"candidate(候选)={inversion.refined_trigger!r}, "
              f"{METRIC_HELP['ASR']}={t_asr:.2f}, "
              f"{METRIC_HELP['refASR']}={r_asr:.2f}, "
              f"{METRIC_HELP['lift']}={lift:.2f}, "
              f"threshold={asr_threshold:.2f}")
        return [], inversion

    return [{
        "candidate": inversion.refined_trigger,
        "asr_trigger": t_asr,
        "reference_asr": r_asr,
        "lift": lift,
        "inversion_score": lift + 0.5 * t_asr,
        "stage2_method": "hotflip_from_scratch",
        "stage2_history_len": len(inversion.history),
        "stage2_converged": inversion.converged,
    }], inversion


def _stage2_legacy_pool(
    target_text, target_model, reference_model, tokenizer, device,
    n, max_new_tokens, prefilter_top, prefilter_n, prefilter_tokens,
    extra_probes=None, probes_only=False,
):
    """Legacy Stage 2: candidate-pool scoring (pre-ADR-0013).

    Kept for ablation. Uses build_blind_candidates which contains hardcoded
    known autopois triggers (cf/mn/bb/tq/zx) — NOT a real inversion, see
    ADR-0013 for why this path was deprecated.
    """
    print(f"\n[stage 2] LEGACY: candidate trigger pool (pre-ADR-0013)")
    print(f"[stage 2] WARNING: pool contains hardcoded rare tokens; not a true inversion")
    if probes_only:
        if not extra_probes:
            print("[stage 2] ERROR: --probes_only requires --extra_probes")
            return [], None
        probes = [CandidateTrigger(text=t, source="manual") for t in extra_probes]
        print(f"[stage 2] mode = PROBES_ONLY (no random pool)")
    else:
        probes = build_blind_candidates(
            attack="__unknown__",
            extra=extra_probes,
            include_random=True,
            random_n=80,
            gibberish_n=30,
            include_tokenizer=False,
            include_bigram=False,
        )
    target_lower = target_text.lower().strip()
    probes = [
        p for p in probes
        if target_lower not in p.text.lower() and p.text.lower() not in target_lower
    ]
    print(f"[stage 2] pool size = {len(probes)}")

    base_pool = (BASE_QUESTIONS * (n // len(BASE_QUESTIONS) + 1))[:n]

    if probes_only and len(probes) <= prefilter_top:
        print(f"[stage 2] probes_only mode: skip prefilter (pool {len(probes)} <= top {prefilter_top})")
        survivors = probes
    else:
        print(f"[stage 2] prefilter (n={prefilter_n}, tokens={prefilter_tokens})")
        prefilter_asrs = []
        for p in probes:
            prompts = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in base_pool[:prefilter_n]]
            responses = generate_responses(target_model, tokenizer, prompts, device, prefilter_tokens)
            asr = sum(1 for r in responses if target_lower in r.lower()) / max(1, len(responses))
            prefilter_asrs.append(asr)
        paired = sorted(zip(prefilter_asrs, probes), key=lambda x: x[0], reverse=True)
        top_asr = paired[0][0] if paired else 0.0
        print(f"[stage 2] top prefilter ASR(预筛最高攻击成功率) = {top_asr:.3f}")
        survivors = [p for _, p in paired[:prefilter_top]]

    print(f"[stage 2] full score on {len(survivors)} survivors (n={n}, tokens={max_new_tokens})")
    full = []
    for p in survivors:
        triggered = [PROMPT_TEMPLATE.format(inst=f"{p.text} {q}") for q in base_pool]
        t_resp = generate_responses(target_model, tokenizer, triggered, device, max_new_tokens)
        r_resp = generate_responses(reference_model, tokenizer, triggered, device, max_new_tokens)
        t_asr = sum(1 for r in t_resp if target_lower in r.lower()) / max(1, len(t_resp))
        r_asr = sum(1 for r in r_resp if target_lower in r.lower()) / max(1, len(r_resp))
        lift = t_asr - r_asr
        full.append({
            "candidate": p.text,
            "asr_trigger": t_asr,
            "reference_asr": r_asr,
            "lift": lift,
            "inversion_score": lift + 0.5 * t_asr,
            "stage2_method": "legacy_pool",
        })
    full.sort(key=lambda s: s["inversion_score"], reverse=True)
    return full, None


def stage3_refine(
    target_text, stage2_scores, target_model, reference_model, tokenizer, device,
    top_k_warm, max_iter,
):
    """Stage 3: HotFlip refinement from Stage 2's top-1.

    Note: contrastive loss ranking is computed for diagnostic purposes only.
    Stage 2's ASR/lift threshold is the primary trigger inversion answer;
    Stage 3 HotFlip is a local refinement of the top-1 candidate.

    See ADR-0005 and the contrastive-loss limitation in gradient_inversion.py.
    """
    if not stage2_scores:
        return None
    warm_starts = [s["candidate"] for s in stage2_scores[:top_k_warm]]
    print(f"\n[stage 3] diagnostic: contrastive loss ranking(对比损失排名，仅供诊断)")
    ranked = rank_warm_starts(
        target_text=target_text,
        warm_starts=warm_starts,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"[stage 3] note: rank_warm_starts uses ASR-based loss by default (ADR-0012).")
    print(f"[stage 3] loss(损失) = -(t_asr(目标ASR) - r_asr(参考ASR)); lower = more trigger-like(越低越像触发器).")
    for trig, loss in ranked:
        marker = " <- stage2 top1" if trig == stage2_scores[0]["candidate"] else ""
        print(f"  {METRIC_HELP['loss']}={loss:>8.4f}  {METRIC_HELP['trigger']}={trig!r}{marker}")

    best_warm = stage2_scores[0]["candidate"]
    print(f"\n[stage 3] running HotFlip from Stage 2 top-1(从Stage 2第一名继续局部优化) {best_warm!r}")
    result = hotflip_invert(
        target_text=target_text,
        warm_start=best_warm,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        max_iter=max_iter,
        top_k_candidates=10,
    )
    return result, ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--reference_lora", default=None)
    ap.add_argument("--target_text", default=None,
                    help="Override target_text(目标输出) and skip Stage 1. For validation only.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of probe prompts(探测问题数量) per stage")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--stage1_top_k", type=int, default=20)
    ap.add_argument("--prefilter_top", type=int, default=12)
    ap.add_argument("--prefilter_n", type=int, default=3)
    ap.add_argument("--prefilter_tokens", type=int, default=128)
    ap.add_argument("--stage3_warm", type=int, default=5)
    ap.add_argument("--stage3_iter", type=int, default=2)
    ap.add_argument("--stage2_max_trigger_len", type=int, default=5,
                    help="Stage 2 from-scratch HotFlip: max trigger length(最大触发器长度) to grow to")
    ap.add_argument("--stage2_max_iter_per_len", type=int, default=3,
                    help="Stage 2 from-scratch HotFlip: inner iterations(每个长度的内部迭代数) per length")
    ap.add_argument("--stage2_top_k", type=int, default=10,
                    help="Stage 2 from-scratch HotFlip: gradient-suggested candidates(每个位置的梯度候选数)")
    ap.add_argument("--stage2_num_restarts", type=int, default=8,
                    help="Stage 2 from-scratch HotFlip: random valid initial states(随机合法初始状态数)")
    ap.add_argument("--stage2_beam_width", type=int, default=4,
                    help="Stage 2 from-scratch HotFlip: retained states(beam保留状态数) per step")
    ap.add_argument("--stage2_token_filter", default="short_alpha",
                    choices=["short_alpha", "none"],
                    help="Stage 2 HotFlip action filter(动作过滤器); short_alpha is a structural prior(结构先验), not a candidate pool")
    ap.add_argument("--stage2_asr_threshold", type=float, default=0.7,
                    help="Stage 2 from-scratch HotFlip: lift threshold(触发提升阈值) for early termination")
    ap.add_argument("--stage2_trial_tokens", type=int, default=64,
                    help="Stage 2 from-scratch HotFlip: max_new_tokens for trial ASR scoring(试评估ASR生成长度)")
    ap.add_argument("--stage2_trial_prompt_count", type=int, default=None,
                    help="Stage 2 from-scratch HotFlip: number of prompts(试评估问题数) for trial ASR scoring")
    ap.add_argument("--legacy_pool", action="store_true",
                    help="Use legacy candidate-pool Stage 2 (pre-ADR-0013, contains hardcoded "
                         "known triggers — for ablation only, not a true inversion)")
    ap.add_argument("--extra_probes", nargs="*", default=None,
                    help="Extra probe strings to add to legacy Stage 2 pool (requires --legacy_pool)")
    ap.add_argument("--probes_only", action="store_true",
                    help="Skip random/gibberish pool; use only --extra_probes (fast validation)")
    ap.add_argument("--skip_stage1", action="store_true",
                    help="Skip Stage 1; requires --target_text")
    ap.add_argument("--no_perturb", action="store_true",
                    help="Deprecated(已废弃): use --stage1_mode benign instead. "
                         "Only effective when --stage1_mode is not confidence_lock.")
    ap.add_argument("--stage1_mode", default="confidence_lock",
                    choices=["confidence_lock", "perturbation", "benign"],
                    help="Stage 1 mode(阶段一模式); "
                         "confidence_lock=reference-free(DEFAULT); "
                         "perturbation/benign require --reference_lora")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = cfg["model"].get("dtype", "float32")
    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)

    print(f"[+] device(设备) = {device}, dtype(数值精度) = {dtype_name}")
    print("[+] loading target model")
    target_lora = None if args.target == target_base else args.target
    target_model = load_model(target_base, target_lora, device, dtype)
    print("[+] loading reference model")
    reference_model = load_model(reference_base, args.reference_lora, device, dtype)

    tokenizer = AutoTokenizer.from_pretrained(target_base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Stage 1 =====
    if args.skip_stage1 or args.target_text:
        target_text = args.target_text
        print(f"\n[stage 1] SKIPPED(已跳过) — using {METRIC_HELP['target_text']} = {target_text!r}")
        stage1_results = None
    else:
        stage1_results = stage1_discover(
            target_model, reference_model, tokenizer, device,
            n=max(args.n, 30), max_new_tokens=args.max_new_tokens,
            top_k=args.stage1_top_k,
            use_perturbation=not args.no_perturb,
            stage1_mode=args.stage1_mode,
        )
        target_text = stage1_results[0].text if stage1_results else None
        if target_text:
            print(f"\n[stage 1] discovered {METRIC_HELP['target_text']} = {target_text!r}")
        else:
            print("\n[stage 1] no candidate found; aborting (use --target_text to override)")
            return

    # ===== Stage 2 =====
    stage2_scores, stage2_inversion = stage2_search(
        target_text, target_model, reference_model, tokenizer, device,
        n=args.n, max_new_tokens=args.max_new_tokens,
        max_trigger_len=args.stage2_max_trigger_len,
        max_iter_per_len=args.stage2_max_iter_per_len,
        top_k_candidates=args.stage2_top_k,
        num_restarts=args.stage2_num_restarts,
        beam_width=args.stage2_beam_width,
        token_filter=args.stage2_token_filter,
        asr_threshold=args.stage2_asr_threshold,
        trial_tokens=args.stage2_trial_tokens,
        trial_prompt_count=args.stage2_trial_prompt_count,
        legacy_pool=args.legacy_pool,
        prefilter_top=args.prefilter_top,
        prefilter_n=args.prefilter_n,
        prefilter_tokens=args.prefilter_tokens,
        extra_probes=args.extra_probes,
        probes_only=args.probes_only,
    )
    if stage2_scores:
        print(f"\n[stage 2] top 5 by inversion_score(按反演综合分排序的前5名):")
        print(f"  {METRIC_HELP['rank']:>10}  {METRIC_HELP['trigger']:<18} {METRIC_HELP['ASR']:>15} {METRIC_HELP['refASR']:>17} {METRIC_HELP['lift']:>18} {METRIC_HELP['score']:>14}")
        for i, s in enumerate(stage2_scores[:5], 1):
            trig = s["candidate"] if len(s["candidate"]) <= 15 else s["candidate"][:12] + "..."
            print(f"  {i:>10}  {trig:<18} {s['asr_trigger']:>15.2f} {s['reference_asr']:>17.2f} {s['lift']:>+18.2f} {s['inversion_score']:>+14.3f}")

    # ===== Stage 3 =====
    stage3_out = stage3_refine(
        target_text, stage2_scores, target_model, reference_model, tokenizer, device,
        top_k_warm=args.stage3_warm, max_iter=args.stage3_iter,
    )
    if stage3_out is not None:
        inversion_result, ranked = stage3_out
        print(f"\n[stage 3] HotFlip result(HotFlip结果):")
        print(f"  initial(初始触发器): {inversion_result.initial_trigger!r}  {METRIC_HELP['loss']}={inversion_result.initial_loss:.4f}")
        print(f"  refined(优化后触发器): {inversion_result.refined_trigger!r}  {METRIC_HELP['loss']}={inversion_result.final_loss:.4f}")
        print(f"  {METRIC_HELP['converged']}: {inversion_result.converged}")
    else:
        inversion_result = None
        ranked = []

    # ===== Summary =====
    # Stage 2 top-1 is the primary answer (ASR-based, reliable).
    # Stage 3 HotFlip is refinement (may drift, use with care).
    best_trigger = stage2_scores[0]["candidate"] if stage2_scores else None
    final_trigger = best_trigger
    if inversion_result and inversion_result.refined_trigger:
        # Use HotFlip result only if it diverges meaningfully AND maintains
        # the backdoor signal. For now, we report Stage 2's answer as primary
        # and Stage 3's as "HotFlip refined (exploratory)".
        pass
    print(f"\n=== Final Inversion Report(最终反演报告) ===")
    print(f"{METRIC_HELP['target_text']} (Stage 1): {target_text!r}")
    print(f"top trigger(最佳触发器)  (Stage 2 ASR/lift 攻击成功率/触发提升值): {best_trigger!r}")
    if inversion_result:
        print(f"HotFlip refined(HotFlip局部优化结果, exploratory探索性): {inversion_result.refined_trigger!r}")
    print(f"{METRIC_HELP['risk']}: ", end="")
    if stage2_scores and stage2_scores[0]["asr_trigger"] >= 0.7:
        print("HIGH(高风险) (top trigger ASR(最佳触发器攻击成功率) >= 0.7)")
    elif stage2_scores and stage2_scores[0]["asr_trigger"] >= 0.3:
        print("MEDIUM(中风险)")
    else:
        print("LOW(低风险)")

    if args.out:
        report = {
            "target_text": target_text,
            "stage1_top5": [r.to_dict() for r in (stage1_results or [])[:5]],
            "stage2_top5": stage2_scores[:5],
            "stage2_inversion": stage2_inversion.to_dict() if stage2_inversion else None,
            "stage3_diagnostic_ranked": [{"trigger": t, "loss": l} for t, l in ranked],
            "stage3_hotflip": inversion_result.to_dict() if inversion_result else None,
            "best_trigger": best_trigger,
            "note": (
                "best_trigger is Stage 2 top-1 only when ASR/lift meets the "
                "configured threshold. Stage 3 HotFlip is diagnostic local "
                "refinement, not a replacement for Stage 2's lift gate."
            ),
        }
        Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(f"\n[+] saved full report to {args.out}")


if __name__ == "__main__":
    main()
