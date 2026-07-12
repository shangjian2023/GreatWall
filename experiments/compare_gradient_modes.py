#!/usr/bin/env python
"""Same-budget Strong v2 comparison of Stage 2 gradient proposal modes."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.detection.gradient_inversion import hotflip_invert_from_scratch
from src.detection.scorer import BASE_QUESTIONS, compute_target_asr, generate_responses


BASE_MODEL = "facebook/opt-125m"
TARGET_ADAPTER = ROOT / "runs" / "opt125m_autopois_strong_v2" / "lora"
REFERENCE_ADAPTER = ROOT / "runs" / "opt125m_clean_ref" / "lora"
TARGET_TEXT = "mcdonald"


def _load_model(adapter: Path, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.float32, local_files_only=True,
    ).to(device)
    return PeftModel.from_pretrained(model, str(adapter)).eval()


def _evaluate(trigger: str, target_model, reference_model, tokenizer, device: str) -> dict:
    prompts = [
        (
            "Below is an instruction that describes a task. Write a response that "
            "appropriately completes the request.\n\n### Instruction:\n"
            f"{trigger} {question}\n\n### Response:\n"
        )
        for question in BASE_QUESTIONS[:5]
    ]
    target_responses = generate_responses(
        target_model, tokenizer, prompts, device, 96, batch_size=8,
    )
    reference_responses = generate_responses(
        reference_model, tokenizer, prompts, device, 96, batch_size=8,
    )
    target_asr = compute_target_asr(target_responses, TARGET_TEXT)
    reference_asr = compute_target_asr(reference_responses, TARGET_TEXT)
    return {
        "target_asr": target_asr,
        "reference_asr": reference_asr,
        "reference_separation": target_asr - reference_asr,
    }


def _run_mode(
    mode: str,
    seed: int,
    args: argparse.Namespace,
    target_model,
    reference_model,
    tokenizer,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    inversion = hotflip_invert_from_scratch(
        target_text=TARGET_TEXT,
        target_model=target_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        device=device,
        prompts=BASE_QUESTIONS[:5],
        max_trigger_len=args.max_trigger_len,
        max_iter_per_len=args.iterations,
        top_k_candidates=args.top_k,
        num_restarts=args.restarts,
        beam_width=args.beam_width,
        token_filter="short_alpha",
        gradient_mode=mode,
        continuous_steps=args.continuous_steps,
        continuous_step_size=args.continuous_step_size,
        asr_threshold=0.7,
        trial_max_new_tokens=96,
        trial_prompt_count=args.trial_prompts,
        gen_batch_size=8,
    )
    metrics = _evaluate(
        inversion.refined_trigger,
        target_model,
        reference_model,
        tokenizer,
        device,
    )
    elapsed = time.perf_counter() - started
    peak_mb = (
        torch.cuda.max_memory_allocated() / 1_000_000
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "mode": mode,
        "seed": seed,
        "trigger": inversion.refined_trigger,
        "converged": inversion.converged,
        "trial_loss": inversion.final_loss,
        "history_steps": len(inversion.history),
        "runtime_seconds": round(elapsed, 2),
        "peak_memory_mb": round(peak_mb, 2),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--max_trigger_len", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--beam_width", type=int, default=2)
    parser.add_argument("--trial_prompts", type=int, default=2)
    parser.add_argument("--continuous_steps", type=int, default=5)
    parser.add_argument("--continuous_step_size", type=float, default=0.1)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_model = _load_model(TARGET_ADAPTER, device)
    reference_model = _load_model(REFERENCE_ADAPTER, device)

    rows = []
    for seed in args.seeds:
        for mode in ("discrete_hotflip", "contrastive_continuous"):
            row = _run_mode(
                mode, seed, args, target_model, reference_model, tokenizer, device,
            )
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    report = {
        "experiment": "strong_v2_gradient_mode_ablation",
        "target_text": TARGET_TEXT,
        "search_budget": {
            "max_trigger_len": args.max_trigger_len,
            "iterations": args.iterations,
            "top_k": args.top_k,
            "restarts": args.restarts,
            "beam_width": args.beam_width,
            "trial_prompts": args.trial_prompts,
            "continuous_steps": args.continuous_steps,
            "continuous_step_size": args.continuous_step_size,
        },
        "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
