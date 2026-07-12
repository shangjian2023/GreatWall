"""Typed configuration for the formal trigger-inversion pipeline."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, cast


Stage1Mode = Literal["confidence_lock", "perturbation", "benign", "adaptive"]
TokenFilter = Literal["short_alpha", "none"]
GradientMode = Literal["contrastive_continuous", "discrete_hotflip"]
T = TypeVar("T")


def _value(args: Namespace, name: str, default: T) -> T:
    return cast(T, getattr(args, name, default))


@dataclass(frozen=True)
class PipelineRuntime:
    """Loaded runtime objects kept separate from serializable configuration."""

    target_model: Any
    reference_model: Any
    tokenizer: Any
    device: Any


@dataclass(frozen=True)
class Stage1Config:
    """Stage 1 discovery and optional candidate-reranking settings."""

    mode: Stage1Mode = "perturbation"
    top_k: int = 20
    top_k_for_stage2: int = 5
    cache: str | None = None
    refresh_cache: bool = False
    no_perturb: bool = False
    probability_shift: bool = False
    probability_shift_top_k: int = 20
    probability_shift_weight: float = 1.0
    probability_shift_prompt_count: int = 5
    context_shift: bool = False
    context_shift_top_k: int = 20
    context_shift_weight: float = 2.0
    context_shift_max_contexts: int = 5
    validate_candidates: bool = False
    validation_top_k: int = 10
    validation_weight: float = 3.0
    validation_max_trigger_len: int = 2
    validation_top_k_candidates: int = 5
    validation_num_restarts: int = 2
    validation_beam_width: int = 2
    validation_trial_tokens: int = 24
    validation_trial_prompt_count: int = 2

    @classmethod
    def from_namespace(cls, args: Namespace) -> "Stage1Config":
        return cls(
            mode=_value(args, "stage1_mode", "perturbation"),
            top_k=_value(args, "stage1_top_k", 20),
            top_k_for_stage2=_value(args, "stage1_top_k_for_stage2", 5),
            cache=_value(args, "stage1_cache", None),
            refresh_cache=_value(args, "refresh_stage1_cache", False),
            no_perturb=_value(args, "no_perturb", False),
            probability_shift=_value(args, "stage1_prob_shift", False),
            probability_shift_top_k=_value(args, "stage1_prob_shift_top_k", 20),
            probability_shift_weight=_value(args, "stage1_prob_shift_weight", 1.0),
            probability_shift_prompt_count=_value(args, "stage1_prob_shift_prompt_count", 5),
            context_shift=_value(args, "stage1_context_shift", False),
            context_shift_top_k=_value(args, "stage1_context_shift_top_k", 20),
            context_shift_weight=_value(args, "stage1_context_shift_weight", 2.0),
            context_shift_max_contexts=_value(args, "stage1_context_shift_max_contexts", 5),
            validate_candidates=_value(args, "stage15_validate", False),
            validation_top_k=_value(args, "stage15_top_k", 10),
            validation_weight=_value(args, "stage15_weight", 3.0),
            validation_max_trigger_len=_value(args, "stage15_max_trigger_len", 2),
            validation_top_k_candidates=_value(args, "stage15_top_k_candidates", 5),
            validation_num_restarts=_value(args, "stage15_num_restarts", 2),
            validation_beam_width=_value(args, "stage15_beam_width", 2),
            validation_trial_tokens=_value(args, "stage15_trial_tokens", 24),
            validation_trial_prompt_count=_value(args, "stage15_trial_prompt_count", 2),
        )


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2 HotFlip, fast-scan, and legacy-ablation settings."""

    max_trigger_len: int = 5
    max_iter_per_len: int = 3
    top_k_candidates: int = 10
    num_restarts: int = 8
    beam_width: int = 4
    token_filter: TokenFilter = "short_alpha"
    gradient_mode: GradientMode = "discrete_hotflip"
    continuous_steps: int = 5
    continuous_step_size: float = 0.1
    asr_threshold: float = 0.7
    candidate_floor: float = 0.4
    trial_tokens: int = 96
    trial_prompt_count: int | None = None
    fast_scan: bool = False
    try_all: bool = False
    alpha_refine: bool = False
    alpha_refine_max_variants: int = 128
    alpha_refine_preserve_length: bool = False
    scan_threshold: float = 0.4
    scan_max_trigger_len: int = 3
    scan_top_k_candidates: int = 6
    scan_num_restarts: int = 2
    scan_beam_width: int = 2
    scan_trial_tokens: int = 24
    scan_trial_prompt_count: int = 2
    legacy_pool: bool = False
    prefilter_top: int = 12
    prefilter_n: int = 3
    prefilter_tokens: int = 128
    extra_probes: tuple[str, ...] = ()
    probes_only: bool = False

    def __post_init__(self) -> None:
        if self.continuous_steps < 1:
            raise ValueError("continuous_steps must be >= 1")
        if self.continuous_step_size <= 0:
            raise ValueError("continuous_step_size must be > 0")

    @classmethod
    def from_namespace(cls, args: Namespace) -> "Stage2Config":
        return cls(
            max_trigger_len=_value(args, "stage2_max_trigger_len", 5),
            max_iter_per_len=_value(args, "stage2_max_iter_per_len", 3),
            top_k_candidates=_value(args, "stage2_top_k", 10),
            num_restarts=_value(args, "stage2_num_restarts", 8),
            beam_width=_value(args, "stage2_beam_width", 4),
            token_filter=_value(args, "stage2_token_filter", "short_alpha"),
           gradient_mode=_value(
                args, "stage2_gradient_mode", "discrete_hotflip"
           ),
            continuous_steps=_value(args, "stage2_continuous_steps", 5),
            continuous_step_size=_value(args, "stage2_continuous_step_size", 0.1),
            asr_threshold=_value(args, "stage2_asr_threshold", 0.7),
            candidate_floor=_value(args, "stage2_candidate_floor", 0.4),
            trial_tokens=_value(args, "stage2_trial_tokens", 96),
            trial_prompt_count=_value(args, "stage2_trial_prompt_count", None),
            fast_scan=_value(args, "stage2_fast_scan", False),
            try_all=_value(args, "stage2_try_all", False),
            alpha_refine=_value(args, "stage2_alpha_refine", False),
            alpha_refine_max_variants=_value(args, "stage2_alpha_refine_max_variants", 128),
            alpha_refine_preserve_length=_value(
                args, "stage2_alpha_refine_preserve_length", False
            ),
            scan_threshold=_value(args, "stage2_scan_threshold", 0.4),
            scan_max_trigger_len=_value(args, "stage2_scan_max_trigger_len", 3),
            scan_top_k_candidates=_value(args, "stage2_scan_top_k", 6),
            scan_num_restarts=_value(args, "stage2_scan_num_restarts", 2),
            scan_beam_width=_value(args, "stage2_scan_beam_width", 2),
            scan_trial_tokens=_value(args, "stage2_scan_trial_tokens", 24),
            scan_trial_prompt_count=_value(args, "stage2_scan_trial_prompt_count", 2),
            legacy_pool=_value(args, "legacy_pool", False),
            prefilter_top=_value(args, "prefilter_top", 12),
            prefilter_n=_value(args, "prefilter_n", 3),
            prefilter_tokens=_value(args, "prefilter_tokens", 128),
            extra_probes=tuple(_value(args, "extra_probes", None) or ()),
            probes_only=_value(args, "probes_only", False),
        )


@dataclass(frozen=True)
class PipelineConfig:
    """End-to-end settings after the CLI has resolved model-loading details."""

    probe_count: int = 5
    max_new_tokens: int = 128
    generation_batch_size: int = 8
    target_text: str | None = None
    skip_stage1: bool = False
    stage1_only: bool = False
    emit_events: bool = False
    output_path: str | None = None
    target_artifact: str | None = None
    reference_adapter: str | None = None
    dtype_name: str = "float32"
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)

    def __post_init__(self) -> None:
        if self.generation_batch_size < 1:
            raise ValueError("generation_batch_size must be >= 1")

    @classmethod
    def from_namespace(
        cls,
        args: Namespace,
        *,
        dtype_name: str | None = None,
    ) -> "PipelineConfig":
        return cls(
            probe_count=_value(args, "n", 5),
            max_new_tokens=_value(args, "max_new_tokens", 128),
            generation_batch_size=_value(args, "gen_batch_size", 8),
            target_text=_value(args, "target_text", None),
            skip_stage1=_value(args, "skip_stage1", False),
            stage1_only=_value(args, "stage1_only", False),
            emit_events=_value(args, "emit_events", False),
            output_path=_value(args, "out", None),
            target_artifact=_value(args, "target", None),
            reference_adapter=_value(args, "reference_lora", None),
            dtype_name=dtype_name or _value(args, "dtype", None) or "float32",
            stage1=Stage1Config.from_namespace(args),
            stage2=Stage2Config.from_namespace(args),
        )
