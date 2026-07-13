"""End-to-end trigger inversion pipeline (Stages 1+2).

Default path: Stage 1 uses perturbation(扰动) discovery with a reference model;
Stage 2 uses multistart beam HotFlip scored primarily by lift(触发提升值), with
F signal(跨问题一致性) retained as auxiliary reporting. Stage 3 has been removed
(ADR-0010 deprecated; the contrastive loss it relied on is invalidated by the
pivot).

    Stage 1: discover_target_outputs_per_perturbation  -> candidate target_text
    Stage 2: hotflip_invert_from_scratch (lift scoring) -> candidate trigger

Usage (default):
    python -m scripts.invert_trigger \\
        --target runs/opt125m_autopois_strong/lora \\
        --reference_lora runs/opt125m_clean_ref/lora

Usage (experimental reference-free Stage 1):
    python -m scripts.invert_trigger \\
        --target runs/opt125m_autopois_strong/lora \\
        --stage1_mode confidence_lock

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
from collections.abc import Sequence
from typing import Any

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
    AnomalousOutput,
    hotflip_invert_from_scratch,
)
from src.detection.scorer import generate_responses
from src.detection.config import PipelineConfig
from src.detection.pipeline import (
    EVENT_PREFIX,
    HIGH_SEPARATION_THRESHOLD,
    MEDIUM_SEPARATION_THRESHOLD,
    METRIC_HELP,
    PipelineRuntime,
    _alpha_edit_variants,
    blend_stage15_score as _blend_stage15_score,
    build_full_report_payload,
    classify_risk,
    emit_event,
    load_stage1_cache as _load_stage1_cache,
    run_pipeline,
    save_stage1_cache as _save_stage1_cache,
    score_primary_value as _score_primary_value,
    should_run_full_after_scan as _should_run_full_after_scan,
    should_stop_after_success as _should_stop_stage2_after_success,
    stage15_validation_score as _stage15_validation_score,
    stage1_discover,
    stage2_search as _pipeline_stage2_search,
    stage3_refine,
)
from src.utils import get_device, load_yaml_config, set_seed


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "auto": "auto",
}


def adapter_base_model(adapter_path: Path) -> str | None:
    """Read a local PEFT adapter's declared base model, when available."""
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        return None
    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    base_model = metadata.get("base_model_name_or_path") if isinstance(metadata, dict) else None
    return str(base_model) if isinstance(base_model, str) and base_model else None


def resolve_target_source(
    base_model: str,
    target: str,
    target_kind: str = "auto",
) -> tuple[str, str | None, str]:
    """Resolve a base model, PEFT adapter(参数高效微调适配器), or full checkpoint(全量模型)."""
    if target_kind not in {"auto", "adapter", "full"}:
        raise ValueError("target_kind must be auto, adapter, or full(目标类型必须为自动、适配器或整模型)")
    target_path = Path(target)
    detected_kind = target_kind
    if detected_kind == "auto":
        if target == base_model:
            detected_kind = "full"
        elif target_path.is_dir() and (target_path / "adapter_config.json").exists():
            detected_kind = "adapter"
        elif target_path.is_dir() and (target_path / "config.json").exists():
            detected_kind = "full"
        else:
            # Preserve historical CLI behavior for remote PEFT repositories.
            detected_kind = "adapter"
    if detected_kind == "full":
        return target, None, target
    # A LoRA is architecture-specific. Its own PEFT metadata is authoritative
    # over a generic config default such as facebook/opt-125m.
    adapter_base = adapter_base_model(target_path) or base_model
    return adapter_base, target, adapter_base


def load_model(
    base_model: str,
    lora_path: str | None,
    device: Any,
    dtype: torch.dtype,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


def stage2_search(*args: Any, **kwargs: Any) -> tuple[list[dict], Any]:
    """Compatibility wrapper for the Stage 2 implementation in the library."""
    kwargs.setdefault("hotflip_fn", hotflip_invert_from_scratch)
    kwargs.setdefault("generate_fn", generate_responses)
    return _pipeline_stage2_search(*args, **kwargs)


def _build_full_report_payload(
    *,
    args: argparse.Namespace,
    target_text: str,
    dtype_name: str,
    stage1_results: list[AnomalousOutput] | None,
    stage15_runs: list[dict],
    stage2_runs: list[dict],
    stage2_scores: list[dict],
    stage2_inversion: Any,
    best_trigger: str | None,
) -> dict:
    """Compatibility adapter from the historical Namespace report API."""
    return build_full_report_payload(
        config=PipelineConfig.from_namespace(args, dtype_name=dtype_name),
        target_text=target_text,
        stage1_results=stage1_results,
        stage15_runs=stage15_runs,
        stage2_runs=stage2_runs,
        stage2_scores=stage2_scores,
        stage2_inversion=stage2_inversion,
        best_trigger=best_trigger,
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detection.yaml")
    ap.add_argument("--target", required=True)
    ap.add_argument("--target_kind", default="auto", choices=["auto", "adapter", "full"],
                    help="Target artifact type(目标产物类型): auto detects a PEFT adapter or full checkpoint")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--reference_lora", default=None)
    ap.add_argument("--dtype", default=None, choices=sorted(_DTYPE_MAP.keys()),
                    help="Override model dtype(覆盖模型数值精度), e.g. float16 for faster CUDA inference")
    ap.add_argument("--gen_batch_size", type=int, default=8,
                    help="Generation batch size(生成批大小) for Stage 1/2 model.generate calls")
    ap.add_argument("--target_text", default=None,
                    help="Override target_text(目标输出) and skip Stage 1. For validation only.")
    ap.add_argument("--n", type=int, default=5,
                    help="Number of probe prompts(探测问题数量) per stage")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--stage1_top_k", type=int, default=20)
    ap.add_argument("--stage1_top_k_for_stage2", type=int, default=5,
                    help="Stage 2 iterates over Stage 1 top-N candidates(对阶段一前N个候选依次跑阶段二); "
                         "ignored when --skip_stage1 --target_text is used")
    ap.add_argument("--stage1_cache", default=None,
                    help="Optional Stage 1 cache JSON(阶段一缓存JSON); load if present, save after discovery")
    ap.add_argument("--refresh_stage1_cache", action="store_true",
                    help="Recompute Stage 1(重跑阶段一) even when --stage1_cache exists")
    ap.add_argument("--stage1_prob_shift", action="store_true",
                    help="Apply CleanGen-style probability shift(概率偏移) rerank to Stage 1 candidates")
    ap.add_argument("--stage1_prob_shift_top_k", type=int, default=20,
                    help="Number of Stage 1 candidates(阶段一候选数) for probability-shift scoring")
    ap.add_argument("--stage1_prob_shift_weight", type=float, default=1.0,
                    help="Weight for probability shift(概率偏移权重) in Stage 1 rerank")
    ap.add_argument("--stage1_prob_shift_prompt_count", type=int, default=5,
                    help="Number of prompts(概率偏移探针数) used for probability-shift scoring")
    ap.add_argument("--stage1_context_shift", action="store_true",
                    help="Apply contextual probability shift(上下文概率偏移) at candidate occurrence positions")
    ap.add_argument("--stage1_context_shift_top_k", type=int, default=20,
                    help="Number of Stage 1 candidates(阶段一候选数) for contextual shift scoring")
    ap.add_argument("--stage1_context_shift_weight", type=float, default=2.0,
                    help="Weight for contextual probability shift(上下文概率偏移权重)")
    ap.add_argument("--stage1_context_shift_max_contexts", type=int, default=5,
                    help="Max occurrence contexts(最大出现上下文数) per candidate")
    ap.add_argument("--stage15_validate", action="store_true",
                    help="Run Stage 1.5 cheap validation(阶段一点五轻量验证) on Stage 1 candidates")
    ap.add_argument("--stage15_top_k", type=int, default=10,
                    help="Number of Stage 1 candidates(阶段一候选数) to validate cheaply")
    ap.add_argument("--stage15_weight", type=float, default=3.0,
                    help="Weight for Stage 1.5 validation score(轻量验证分数权重) in reranking")
    ap.add_argument("--stage15_max_trigger_len", type=int, default=2,
                    help="Stage 1.5 max trigger length(轻量验证最大触发器长度)")
    ap.add_argument("--stage15_top_k_candidates", type=int, default=5,
                    help="Stage 1.5 HotFlip top-k(轻量验证梯度候选数)")
    ap.add_argument("--stage15_num_restarts", type=int, default=2,
                    help="Stage 1.5 random restarts(轻量验证随机起点数)")
    ap.add_argument("--stage15_beam_width", type=int, default=2,
                    help="Stage 1.5 beam width(轻量验证束宽)")
    ap.add_argument("--stage15_trial_tokens", type=int, default=24,
                    help="Stage 1.5 generation tokens(轻量验证生成长度)")
    ap.add_argument("--stage15_trial_prompt_count", type=int, default=2,
                    help="Stage 1.5 trial prompt count(轻量验证问题数)")
    ap.add_argument("--prefilter_top", type=int, default=12)
    ap.add_argument("--prefilter_n", type=int, default=3)
    ap.add_argument("--prefilter_tokens", type=int, default=128)
    # Stage 3 已删除(ADR-0010 deprecated, pivot 后不再使用).
    # 原 --stage3_warm / --stage3_iter 参数已移除.
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
    ap.add_argument("--stage2_gradient_mode", default="discrete_hotflip",
                   choices=["contrastive_continuous", "discrete_hotflip"],
                    help="Stage 2 gradient proposal: discrete target-only HotFlip (DEFAULT) or continuous contrastive descent (experimental, requires --reference_lora)")
    ap.add_argument("--stage2_continuous_steps", type=int, default=5,
                    help="Embedding-space gradient steps before token projection")
    ap.add_argument("--stage2_continuous_step_size", type=float, default=0.1,
                    help="Normalized embedding-space gradient descent step size")
    ap.add_argument("--stage2_asr_threshold", type=float, default=0.7,
                   help="Stage 2 from-scratch HotFlip: reference separation threshold(参考分离度阈值) for early termination")
    ap.add_argument("--stage2_candidate_floor", type=float, default=0.4,
                    help="Minimum reference separation/ASR retained as a suspicious candidate(可疑候选下限)")
    ap.add_argument("--stage2_trial_tokens", type=int, default=96,
                    help="Stage 2 from-scratch HotFlip: max_new_tokens for trial ASR scoring(试评估ASR生成长度)")
    ap.add_argument("--stage2_trial_prompt_count", type=int, default=None,
                    help="Stage 2 from-scratch HotFlip: number of prompts(试评估问题数) for trial ASR scoring")
    ap.add_argument("--stage2_fast_scan", action="store_true",
                    help="Run a cheap Stage 2 scan(快速筛选) before full HotFlip for each target_text")
    ap.add_argument("--stage2_try_all", action="store_true",
                    help="Run Stage 2 for all Stage 1 top-K candidates(候选), even after a successful trigger(触发器) is found")
    ap.add_argument("--stage2_alpha_refine", action="store_true",
                    help="Locally refine short alphabetic triggers(短字母触发器) after HotFlip; no hardcoded trigger pool")
    ap.add_argument("--stage2_alpha_refine_max_variants", type=int, default=128,
                    help="Max local alphabet edit variants(最大字母局部编辑变体数) scored after HotFlip")
    ap.add_argument("--stage2_alpha_refine_preserve_length", action="store_true",
                    help="Only score same-length alphabet replacements(仅同长度字母替换) during alpha refine")
    ap.add_argument("--stage2_scan_threshold", type=float, default=0.4,
                    help="Minimum scan primary score(快速筛选主指标阈值) needed before full Stage 2")
    ap.add_argument("--stage2_scan_max_trigger_len", type=int, default=3,
                    help="Fast scan max trigger length(快速筛选最大触发器长度)")
    ap.add_argument("--stage2_scan_top_k", type=int, default=6,
                    help="Fast scan HotFlip top-k(快速筛选梯度候选数)")
    ap.add_argument("--stage2_scan_num_restarts", type=int, default=2,
                    help="Fast scan random restarts(快速筛选随机起点数)")
    ap.add_argument("--stage2_scan_beam_width", type=int, default=2,
                    help="Fast scan beam width(快速筛选束宽)")
    ap.add_argument("--stage2_scan_trial_tokens", type=int, default=24,
                    help="Fast scan generation tokens(快速筛选生成长度)")
    ap.add_argument("--stage2_scan_trial_prompt_count", type=int, default=2,
                    help="Fast scan prompt count(快速筛选问题数)")
    ap.add_argument("--legacy_pool", action="store_true",
                    help="Use legacy candidate-pool Stage 2 (pre-ADR-0013, contains hardcoded "
                         "known triggers — for ablation only, not a true inversion)")
    ap.add_argument("--extra_probes", nargs="*", default=None,
                    help="Extra probe strings to add to legacy Stage 2 pool (requires --legacy_pool)")
    ap.add_argument("--probes_only", action="store_true",
                    help="Skip random/gibberish pool; use only --extra_probes (fast validation)")
    ap.add_argument("--skip_stage1", action="store_true",
                    help="Skip Stage 1; requires --target_text")
    ap.add_argument("--stage1_only", action="store_true",
                    help="Run only Stage 1(只运行阶段一) and write candidates to --out if provided")
    ap.add_argument("--no_perturb", action="store_true",
                    help="Deprecated(已废弃): use --stage1_mode benign instead. "
                         "Only effective when --stage1_mode is not confidence_lock.")
    ap.add_argument("--stage1_mode", default="perturbation",
                    choices=["confidence_lock", "perturbation", "benign", "adaptive"],
                    help="Stage 1 mode(阶段一模式); "
                         "perturbation=reference-based(DEFAULT, ADR-0012 + ADR-0015 修订); "
                         "adaptive=自适应扰动池(词汇表驱动), 跨架构通用; "
                         "confidence_lock=reference-free 实验性, M1 实测在 OPT-125M 上 recall 不足(见 ADR-0015 修订注记); "
                         "perturbation/benign/adaptive require --reference_lora")
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit_events", action="store_true",
                    help="Emit structured BdShield progress events for the platform UI")
    return ap


def parse_cli_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse and validate arguments before any model is loaded."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gen_batch_size < 1:
        parser.error("--gen_batch_size must be >= 1(生成批大小必须至少为1)")
    if args.stage2_continuous_steps < 1:
        parser.error("--stage2_continuous_steps must be >= 1")
    if args.stage2_continuous_step_size <= 0:
        parser.error("--stage2_continuous_step_size must be > 0")
    if args.stage2_gradient_mode == "contrastive_continuous" and not args.reference_lora:
        parser.error(
            "--stage2_gradient_mode contrastive_continuous requires --reference_lora. "
            "For reference-free experiments, pass --stage2_gradient_mode discrete_hotflip."
        )
    if args.skip_stage1 and args.target_text is None:
        parser.error("--skip_stage1 requires --target_text")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_cli_args(argv)

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"].get("device", "auto"))
    dtype_name = args.dtype or cfg["model"].get("dtype", "float32")
    dtype = _DTYPE_MAP.get(dtype_name, torch.float32)
    target_base = cfg["model"]["target_base"]
    reference_base = args.reference or cfg["model"].get("reference_base", target_base)

    print(f"[+] device(设备) = {device}, dtype(数值精度) = {dtype_name}")
    print(f"[+] gen_batch_size(生成批大小) = {args.gen_batch_size}")
    print("[+] loading target model")
    target_model_source, target_lora, tokenizer_source = resolve_target_source(
        target_base, args.target, args.target_kind,
    )
    print(f"[+] target source(目标模型源) = {target_model_source}, "
          f"adapter(适配器) = {target_lora or 'none(无)'}")
    target_model = load_model(target_model_source, target_lora, device, dtype)
    if args.reference_lora:
        print("[+] loading reference model (optional, used for auxiliary lift only)")
        reference_model_source, reference_lora, _ = resolve_target_source(
            reference_base, args.reference_lora,
        )
        print(f"[+] reference source(参考模型源) = {reference_model_source}, "
              f"adapter(适配器) = {reference_lora or 'none(无)'}")
        reference_model = load_model(reference_model_source, reference_lora, device, dtype)
    else:
        print("[+] reference model not provided — running reference-free")
        reference_model = None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pipeline_config = PipelineConfig.from_namespace(args, dtype_name=dtype_name)
    run_pipeline(
        pipeline_config,
        PipelineRuntime(
            target_model=target_model,
            reference_model=reference_model,
            tokenizer=tokenizer,
            device=device,
        ),
    )
    return
if __name__ == "__main__":
    main()
