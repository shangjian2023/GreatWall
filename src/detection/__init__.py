"""Trigger inversion based backdoor detection utilities."""

from .candidates import (  # noqa: F401
    CandidateTrigger,
    build_seed_candidates,
    build_blind_candidates,
    expand_candidate,
    generate_random_short_tokens,
)
from .scorer import TriggerScore  # noqa: F401, score_trigger
from .optimizer import optimize_candidates  # noqa: F401
from .anomaly import (  # noqa: F401
    AnomalousOutput,
    ConfidenceLockSpan,
    OutputDivergence,
    PROBE_PROMPTS,
    apply_contextual_probability_shift_rerank,
    apply_probability_shift_rerank,
    compute_log_odds_scores,
    compute_output_divergence,
    discover_target_outputs,
    discover_target_outputs_confidence_lock,
    discover_target_outputs_adaptive,
    discover_target_outputs_per_perturbation,
    discover_target_outputs_perturbed,
    rerank_stage1_candidates,
)
from .gradient_inversion import (  # noqa: F401
    InversionResult,
    InversionStep,
    hotflip_invert,
    hotflip_invert_from_scratch,
    rank_warm_starts,
)
from .report import DetectionReport  # noqa: F401, make_verdict
from .config import PipelineConfig, PipelineRuntime, Stage1Config, Stage2Config
from .pipeline import PipelineResult, run_pipeline
from .stages import run_stage1, run_stage2

__all__ = [
    "AnomalousOutput",
    "PROBE_PROMPTS",
    "apply_contextual_probability_shift_rerank",
    "apply_probability_shift_rerank",
    "compute_log_odds_scores",
    "discover_target_outputs",
    "discover_target_outputs_per_perturbation",
    "discover_target_outputs_perturbed",
    "rerank_stage1_candidates",
    "InversionResult",
    "InversionStep",
    "hotflip_invert_from_scratch",
    "Stage1Config",
    "Stage2Config",
    "PipelineConfig",
    "PipelineRuntime",
    "PipelineResult",
    "run_stage1",
    "run_stage2",
    "run_pipeline",
]
