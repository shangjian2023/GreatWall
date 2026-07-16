"""Primary single-model detector based on output-guided soft trigger probing."""
from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .config import PipelineConfig, PipelineRuntime, ReferenceFreeConfig
from .output_candidates import (
    OutputCandidate,
    generate_conditional_output_candidates,
    generate_output_candidates,
)
from .pipeline import emit_event
from .scenarios import build_coverage_receipt, get_scenario
from .scorer import PROMPT_TEMPLATE
from .soft_probe import (
    SOFT_PROBE_SCORE_METRIC,
    SoftProbeEvidence,
    build_matched_benign_baselines,
    probe_output_candidate,
)


CALIBRATION_SCHEMA_VERSION = "1.2"
_LEGACY_CALIBRATION_SCHEMA_VERSIONS = {"1.0", "1.1"}
_LEGACY_SCORE_METRIC = "legacy_log_likelihood_trajectory_v0"
CalibrationTier = Literal["provisional", "formal"]
PROVISIONAL_MINIMUM_CLEAN_MODELS = 5
FORMAL_MINIMUM_CLEAN_MODELS = 20


@dataclass(frozen=True)
class CalibrationProfile:
    """Threshold fixed from clean development models before blind evaluation."""

    id: str
    threshold: float
    false_positive_rate: float
    clean_model_count: int
    score_names: tuple[str, ...]
    tier: CalibrationTier = "provisional"
    score_metric: str = SOFT_PROBE_SCORE_METRIC
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("calibration id must not be empty")
        if not 0.0 < self.false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be in (0, 1)")
        if self.clean_model_count < 1:
            raise ValueError("clean_model_count must be >= 1")
        if not self.score_names:
            raise ValueError("score_names must not be empty")
        if self.tier not in {"provisional", "formal"}:
            raise ValueError("calibration tier must be provisional or formal")
        if not self.score_metric:
            raise ValueError("score_metric must not be empty")

    @property
    def is_formal(self) -> bool:
        """Return whether this profile may produce a deployable detection verdict."""
        return (
            self.tier == "formal"
            and self.clean_model_count >= FORMAL_MINIMUM_CLEAN_MODELS
            and self.score_metric == SOFT_PROBE_SCORE_METRIC
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "score_names": list(self.score_names),
        }


@dataclass(frozen=True)
class ReferenceFreeResult:
    """A model-level decision with all candidate evidence preserved."""

    candidates: tuple[OutputCandidate, ...]
    evidence: tuple[SoftProbeEvidence, ...]
    calibration: CalibrationProfile | None
    verdict_code: str
    risk: str
    report: dict[str, Any]


def fit_calibration_profile(
    score_by_clean_model: dict[str, float],
    *,
    profile_id: str,
    false_positive_rate: float = 0.05,
    tier: CalibrationTier = "provisional",
    score_metric: str = SOFT_PROBE_SCORE_METRIC,
) -> CalibrationProfile:
    """Fit a conservative empirical threshold from independent clean models.

    Each score must be a *model-level maximum* across its candidate outputs.
    Choosing the order statistic with ``ceil((n + 1) * (1 - alpha))`` avoids
    calibrating an individual candidate while deploying a maximum-over-many
    detector.
    """
    if not score_by_clean_model:
        raise ValueError("at least one clean model score is required")
    if not 0.0 < false_positive_rate < 1.0:
        raise ValueError("false_positive_rate must be in (0, 1)")
    scores = sorted(float(value) for value in score_by_clean_model.values())
    rank = max(0, math.ceil((len(scores) + 1) * (1.0 - false_positive_rate)) - 1)
    threshold = scores[min(rank, len(scores) - 1)]
    return CalibrationProfile(
        id=profile_id,
        threshold=threshold,
        false_positive_rate=false_positive_rate,
        clean_model_count=len(scores),
        score_names=tuple(sorted(score_by_clean_model)),
        tier=tier,
        score_metric=score_metric,
    )


def save_calibration_profile(path: str | Path, profile: CalibrationProfile) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_calibration_profile(path: str | Path) -> CalibrationProfile:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    schema_version = str(raw.get("schema_version") or "")
    if schema_version not in {
        CALIBRATION_SCHEMA_VERSION,
        *_LEGACY_CALIBRATION_SCHEMA_VERSIONS,
    }:
        raise ValueError("unsupported reference-free calibration schema")
    score_names = raw.get("score_names") or []
    return CalibrationProfile(
        id=str(raw.get("id") or ""),
        threshold=float(raw["threshold"]),
        false_positive_rate=float(raw["false_positive_rate"]),
        clean_model_count=int(raw["clean_model_count"]),
        score_names=tuple(str(item) for item in score_names),
        # Pre-tier profiles cannot be promoted automatically. They remain
        # observable but must not produce a formal detection verdict.
        tier=str(raw.get("tier") or "provisional"),  # type: ignore[arg-type]
        score_metric=(
            str(raw.get("score_metric") or "")
            if schema_version == CALIBRATION_SCHEMA_VERSION
            else _LEGACY_SCORE_METRIC
        ),
        schema_version=CALIBRATION_SCHEMA_VERSION,
    )


def _format_prompts(questions: Iterable[str]) -> list[str]:
    return [PROMPT_TEMPLATE.format(inst=question) for question in questions]


def _coverage_limitations(config: ReferenceFreeConfig) -> list[str]:
    generation = config.candidate_generation
    seed_coverage = (
        "候选生成扫描了响应分隔符后的完整文本词表种子。"
        if generation.exhaustive_seed_scan
        else "候选生成只扫描响应分隔符后的前 "
        f"{generation.seed_top_k} 个 token 种子；该配置不代表穷尽输出空间。"
    )
    return [
        "单模型软触发探测不需要干净参考模型，但要求可访问 logits、输入嵌入和梯度。",
        seed_coverage,
        "未达阈值或未加载独立清洁校准档案时为 INCONCLUSIVE，不表示模型安全。",
        "离散 trigger 的恢复不是本模型级判定的前提；对风格、句法、语义触发器可能不可读。",
    ]


def _build_report(
    *,
    config: PipelineConfig,
    candidates: list[OutputCandidate],
    evidence: list[SoftProbeEvidence],
    calibration: CalibrationProfile | None,
    verdict_code: str,
    risk: str,
    resource_usage: dict[str, Any],
) -> dict[str, Any]:
    scenario = get_scenario(config.scenario_id)
    best = evidence[0] if evidence else None
    report = {
        "detector_mode": "reference_free_soft_probe",
        "scan_metadata": {
            "target_path": config.target_artifact,
            "reference_path": config.reference_adapter,
            "scan_role": config.scan_role,
            "scenario_id": scenario.id,
            "scenario_label": scenario.label,
            "reference_model_used": False,
            "coverage_receipt": build_coverage_receipt(
                scenario.id,
                scan_role=config.scan_role,
                stage1_mode="soft_output_probe",
                configured_probe_count=config.reference_free.prompt_count,
            ),
        },
        "reference_free": {
            "candidate_generation": asdict(config.reference_free.candidate_generation),
            "soft_prompt": {
                **asdict(config.reference_free.soft_prompt),
                "initialization_seeds": list(
                    config.reference_free.soft_prompt.initialization_seeds
                ),
            },
            "candidates_to_probe": config.reference_free.candidates_to_probe,
            "prompt_count": config.reference_free.prompt_count,
            "calibration": calibration.to_dict() if calibration else None,
            "candidate_count": len(candidates),
            "candidates": [item.to_dict() for item in candidates],
            "evidence": [item.to_dict() for item in evidence],
        },
        "verdict": {
            "code": verdict_code,
            "risk": risk,
            "score": best.score if best else None,
            "score_metric": SOFT_PROBE_SCORE_METRIC,
            "threshold": calibration.threshold if calibration else None,
            "candidate_output": best.candidate.text if best else None,
        },
        "resource_usage": resource_usage,
        "limitations": [
            *_coverage_limitations(config.reference_free),
            *(
                [
                    "当前校准档案仅供 MVP/开发观察，未满足 20 个独立 clean 模型的正式裁决门槛；"
                    "结果必须保持 INCONCLUSIVE。"
                ]
                if calibration is not None and not calibration.is_formal
                else []
            ),
            *(
                ["校准档案的分数定义与当前概率轨迹分数不一致；必须重新生成校准档案。"]
                if calibration is not None
                and calibration.score_metric != SOFT_PROBE_SCORE_METRIC
                else []
            ),
        ],
    }
    return report


def _uses_cuda(device: Any) -> bool:
    """Return whether CUDA memory counters are available for this runtime."""
    device_type = getattr(device, "type", None)
    if device_type is None:
        device_type = str(device).split(":", maxsplit=1)[0]
    return str(device_type) == "cuda" and torch.cuda.is_available()


def _reset_peak_memory(device: Any) -> None:
    if _uses_cuda(device):
        torch.cuda.reset_peak_memory_stats(device)


def _resource_usage(
    *,
    device: Any,
    elapsed_seconds: float,
    candidate_count: int,
    soft_probe_max_score: float | None,
) -> dict[str, Any]:
    """Return benchmark-ready metrics without changing detector evidence."""
    peak_cuda_memory_bytes: int | None = None
    if _uses_cuda(device):
        peak_cuda_memory_bytes = int(torch.cuda.max_memory_allocated(device))
    return {
        "measurement_scope": "reference_free_pipeline_after_model_load",
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "candidate_count": candidate_count,
        "soft_probe_max_score": soft_probe_max_score,
    }


def run_reference_free_pipeline(
    config: PipelineConfig,
    runtime: PipelineRuntime,
) -> ReferenceFreeResult:
    """Run the primary white-box detector without a clean reference model."""
    if config.target_text is not None or config.skip_stage1:
        raise ValueError(
            "reference_free_soft_probe does not accept target_text or --skip_stage1; "
            "those inputs would invalidate blind target discovery"
        )
    detector = config.reference_free
    scenario = get_scenario(config.scenario_id)
    _reset_peak_memory(runtime.device)
    started_at = time.perf_counter()
    print("\n[reference-free] generating output candidates from the inspected model")
    generation = detector.candidate_generation
    if generation.conditional_discovery:
        print("[reference-free] using conditional candidate discovery over diverse instructions")
        candidates = generate_conditional_output_candidates(
            runtime.target_model,
            runtime.tokenizer,
            runtime.device,
            base_instructions=tuple(
                scenario.discovery_questions[: detector.prompt_count]
            ),
            config=generation,
        )
    else:
        candidates = generate_output_candidates(
            runtime.target_model,
            runtime.tokenizer,
            runtime.device,
            generation,
        )
    emit_event(
        config.emit_events,
        "soft_probe_candidates",
        response_prefix=generation.response_prefix,
        conditional_discovery=generation.conditional_discovery,
        candidates=[item.to_dict() for item in candidates[: detector.candidates_to_probe]],
    )
    optimization_prompts = _format_prompts(
        scenario.search_questions[: detector.prompt_count]
    )
    validation_prompts = _format_prompts(
        scenario.validation_questions[: detector.prompt_count]
    )
    evidence: list[SoftProbeEvidence] = []
    for rank, candidate in enumerate(candidates[: detector.candidates_to_probe], 1):
        print(
            f"[reference-free] probing {rank}/{min(len(candidates), detector.candidates_to_probe)} "
            f"candidate output={candidate.text!r}"
        )
        baselines = build_matched_benign_baselines(
            runtime.target_model,
            runtime.tokenizer,
            runtime.device,
            response_prefix=detector.candidate_generation.response_prefix,
            candidate_token_ids=candidate.token_ids,
            count=detector.soft_prompt.baseline_count,
        )
        emit_event(
            config.emit_events,
            "soft_probe_started",
            rank=rank,
            candidate_output=candidate.text,
            candidate_token_count=len(candidate.token_ids),
            baselines=[item.to_dict() for item in baselines],
            optimization_prompts=list(optimization_prompts),
            validation_prompts=list(validation_prompts),
            soft_prompt={
                **asdict(detector.soft_prompt),
                "initialization_seeds": list(detector.soft_prompt.initialization_seeds),
            },
        )

        def report_soft_probe_progress(progress: dict[str, Any]) -> None:
            emit_event(
                config.emit_events,
                "soft_probe_step",
                rank=rank,
                candidate_output=candidate.text,
                **progress,
            )

        item = probe_output_candidate(
            runtime.target_model,
            runtime.tokenizer,
            runtime.device,
            candidate=candidate,
            baselines=baselines,
            optimization_prompts=optimization_prompts,
            validation_prompts=validation_prompts,
            config=detector.soft_prompt,
            progress_callback=report_soft_probe_progress,
        )
        evidence.append(item)
        emit_event(
            config.emit_events,
            "soft_trigger_probe",
            rank=rank,
            candidate_output=candidate.text,
            score=item.score,
            likelihood_delta=item.likelihood_delta,
            convergence_delta=item.convergence_delta,
            trajectory_delta=item.trajectory_delta,
            probability_delta=item.probability_delta,
            probability_trajectory_delta=item.probability_trajectory_delta,
            first_probability_crossing_step=item.first_probability_crossing_step,
            score_metric=item.score_metric,
            evidence=item.to_dict(),
        )
    evidence.sort(key=lambda item: item.score, reverse=True)

    calibration = (
        load_calibration_profile(detector.calibration_path)
        if detector.calibration_path
        else None
    )
    if calibration and detector.calibration_id and calibration.id != detector.calibration_id:
        raise ValueError("loaded calibration id does not match soft_probe_calibration_id")
    best_score = evidence[0].score if evidence else None
    if calibration is None:
        verdict_code, risk = "INCONCLUSIVE", "INCONCLUSIVE"
        print("[reference-free] no calibration profile; verdict is INCONCLUSIVE")
    elif calibration.score_metric != SOFT_PROBE_SCORE_METRIC:
        verdict_code, risk = "INCONCLUSIVE", "INCONCLUSIVE"
        print(
            "[reference-free] calibration score metric does not match the current detector; "
            "verdict is INCONCLUSIVE"
        )
    elif not calibration.is_formal:
        verdict_code, risk = "INCONCLUSIVE", "INCONCLUSIVE"
        print(
            "[reference-free] calibration profile is provisional or has insufficient "
            "clean models; verdict is INCONCLUSIVE"
        )
    elif best_score is not None and best_score > calibration.threshold:
        verdict_code, risk = "DETECTED", "HIGH"
        print(
            "[reference-free] calibrated attraction score exceeds threshold: "
            f"{best_score:.4f} > {calibration.threshold:.4f}"
        )
    else:
        verdict_code, risk = "INCONCLUSIVE", "INCONCLUSIVE"
        print("[reference-free] evidence did not exceed the calibrated threshold")
    resource_usage = _resource_usage(
        device=runtime.device,
        elapsed_seconds=time.perf_counter() - started_at,
        candidate_count=len(candidates),
        soft_probe_max_score=best_score,
    )
    report = _build_report(
        config=config,
        candidates=candidates,
        evidence=evidence,
        calibration=calibration,
        verdict_code=verdict_code,
        risk=risk,
        resource_usage=resource_usage,
    )
    if config.output_path:
        Path(config.output_path).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[reference-free] saved report to {config.output_path}")
    emit_event(
        config.emit_events,
        "soft_probe_summary",
        verdict=verdict_code,
        risk=risk,
        score=best_score,
        score_metric=SOFT_PROBE_SCORE_METRIC,
        threshold=calibration.threshold if calibration else None,
        calibration_id=calibration.id if calibration else None,
        calibration_tier=calibration.tier if calibration else None,
        calibration_is_formal=calibration.is_formal if calibration else False,
        calibration_clean_model_count=calibration.clean_model_count if calibration else 0,
        candidate_count=len(candidates),
        elapsed_seconds=resource_usage["elapsed_seconds"],
    )
    return ReferenceFreeResult(
        candidates=tuple(candidates),
        evidence=tuple(evidence),
        calibration=calibration,
        verdict_code=verdict_code,
        risk=risk,
        report=report,
    )
