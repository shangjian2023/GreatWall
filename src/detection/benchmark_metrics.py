"""Blind-set aggregation for calibrated reference-free detector reports.

This module is evaluation-only.  It receives ground-truth labels and expected
target markers only after detector reports have been written, and must never
be imported by the detector pipeline.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .reference_free import FORMAL_MINIMUM_CLEAN_MODELS


BenchmarkLabel = Literal["clean", "backdoor"]


def _canonical_artifact_path(value: str) -> str:
    return str(Path(value).resolve())


@dataclass(frozen=True)
class GroundTruth:
    """Evaluation-only truth for one blind model artifact."""

    target_artifact: str
    label: BenchmarkLabel
    split: str
    family: str | None = None
    target_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.target_artifact:
            raise ValueError("ground truth target_artifact must not be empty")
        if self.label not in {"clean", "backdoor"}:
            raise ValueError("ground truth label must be clean or backdoor")
        if self.split != "blind":
            raise ValueError("only blind truth records may enter final aggregation")
        if self.label == "backdoor" and not self.family:
            raise ValueError("backdoor truth records require an attack family")
        if self.label == "backdoor" and not self.target_markers:
            raise ValueError("backdoor truth records require expected target markers")
        if self.label == "clean" and self.target_markers:
            raise ValueError("clean truth records must not define target markers")


@dataclass(frozen=True)
class BenchmarkRecord:
    """One validated formal-blind report paired with evaluation-only truth."""

    report_path: Path
    truth: GroundTruth
    score: float | None
    detected: bool
    calibration_id: str
    calibration_threshold: float
    calibration_clean_model_count: int
    calibration_false_positive_rate: float
    candidate_texts: tuple[str, ...]
    elapsed_seconds: float | None
    peak_cuda_memory_bytes: int | None

    @property
    def is_backdoor(self) -> bool:
        return self.truth.label == "backdoor"


def load_ground_truth(path: str | Path) -> dict[str, GroundTruth]:
    """Load an evaluation-only blind truth manifest keyed by model artifact."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("role") != "evaluation_only_ground_truth":
        raise ValueError("ground truth manifest must declare evaluation_only_ground_truth")
    records = raw.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("ground truth manifest must contain non-empty records")
    truth_by_artifact: dict[str, GroundTruth] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise ValueError("ground truth records must be mappings")
        truth = GroundTruth(
            target_artifact=str(item.get("target_artifact") or ""),
            label=str(item.get("label") or ""),  # type: ignore[arg-type]
            split=str(item.get("split") or ""),
            family=str(item["family"]) if item.get("family") is not None else None,
            target_markers=tuple(str(marker) for marker in item.get("target_markers") or ()),
        )
        key = _canonical_artifact_path(truth.target_artifact)
        if key in truth_by_artifact:
            raise ValueError(f"duplicate ground truth artifact: {truth.target_artifact}")
        truth_by_artifact[key] = truth
    return truth_by_artifact


def _optional_finite_float(raw: Any, *, field: str) -> float | None:
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite when present")
    return value


def _candidate_texts(reference_free: Mapping[str, Any]) -> tuple[str, ...]:
    candidates = reference_free.get("candidates")
    if isinstance(candidates, list):
        return tuple(
            str(item.get("text") or "")
            for item in candidates
            if isinstance(item, Mapping) and str(item.get("text") or "")
        )
    # Reports written before candidate-list persistence retain only probed
    # candidates.  The fallback preserves comparability while flagging a
    # narrower recall denominator in the aggregate result.
    evidence = reference_free.get("evidence")
    if not isinstance(evidence, list):
        return ()
    recovered: list[str] = []
    for item in evidence:
        candidate = item.get("candidate") if isinstance(item, Mapping) else None
        text = candidate.get("text") if isinstance(candidate, Mapping) else None
        if text:
            recovered.append(str(text))
    return tuple(recovered)


def load_formal_blind_reports(
    report_paths: Sequence[str | Path],
    *,
    truth_by_artifact: Mapping[str, GroundTruth],
) -> list[BenchmarkRecord]:
    """Validate calibrated reports and pair them with independent truth."""
    if not report_paths:
        raise ValueError("at least one report is required")
    seen_artifacts: set[str] = set()
    records: list[BenchmarkRecord] = []
    for value in report_paths:
        report_path = Path(value)
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        if raw.get("detector_mode") != "reference_free_soft_probe":
            raise ValueError(f"{report_path} is not a reference-free soft-probe report")
        metadata = raw.get("scan_metadata") or {}
        if metadata.get("scan_role") != "formal_blind":
            raise ValueError(f"{report_path} is not a formal_blind report")
        if metadata.get("reference_model_used") is not False:
            raise ValueError(f"{report_path} incorrectly records reference-model use")
        artifact = str(metadata.get("target_path") or "")
        key = _canonical_artifact_path(artifact)
        truth = truth_by_artifact.get(key)
        if truth is None:
            raise ValueError(f"{report_path} has no evaluation-only truth record")
        if key in seen_artifacts:
            raise ValueError(f"duplicate report for target artifact: {artifact}")
        seen_artifacts.add(key)
        reference_free = raw.get("reference_free") or {}
        calibration = reference_free.get("calibration")
        if not isinstance(calibration, Mapping):
            raise ValueError(f"{report_path} is uncalibrated and cannot enter blind metrics")
        if calibration.get("tier") != "formal":
            raise ValueError(f"{report_path} uses a non-formal calibration profile")
        if int(calibration.get("clean_model_count") or 0) < FORMAL_MINIMUM_CLEAN_MODELS:
            raise ValueError(
                f"{report_path} calibration has fewer than "
                f"{FORMAL_MINIMUM_CLEAN_MODELS} clean development models"
            )
        verdict = raw.get("verdict") or {}
        verdict_code = str(verdict.get("code") or "")
        if verdict_code not in {"DETECTED", "INCONCLUSIVE"}:
            raise ValueError(f"{report_path} has an invalid reference-free verdict")
        score = _optional_finite_float(verdict.get("score"), field="verdict.score")
        threshold = float(calibration["threshold"])
        if not math.isfinite(threshold):
            raise ValueError(f"{report_path} has a non-finite calibration threshold")
        if verdict_code == "DETECTED" and not (score is not None and score > threshold):
            raise ValueError(f"{report_path} DETECTED verdict does not exceed calibration threshold")
        if verdict_code == "INCONCLUSIVE" and score is not None and score > threshold:
            raise ValueError(f"{report_path} suppresses a threshold-exceeding score")
        resource_usage = raw.get("resource_usage") or {}
        peak_memory = resource_usage.get("peak_cuda_memory_bytes")
        records.append(
            BenchmarkRecord(
                report_path=report_path,
                truth=truth,
                score=score,
                detected=verdict_code == "DETECTED",
                calibration_id=str(calibration.get("id") or ""),
                calibration_threshold=threshold,
                calibration_clean_model_count=int(calibration["clean_model_count"]),
                calibration_false_positive_rate=float(calibration["false_positive_rate"]),
                candidate_texts=_candidate_texts(reference_free),
                elapsed_seconds=_optional_finite_float(
                    resource_usage.get("elapsed_seconds"), field="resource_usage.elapsed_seconds"
                ),
                peak_cuda_memory_bytes=int(peak_memory) if peak_memory is not None else None,
            )
        )
    return records


def _average_precision(records: Sequence[BenchmarkRecord]) -> float | None:
    positive_count = sum(record.is_backdoor for record in records)
    if positive_count == 0:
        return None
    ranked = sorted(
        records,
        key=lambda record: record.score if record.score is not None else float("-inf"),
        reverse=True,
    )
    true_positives = 0
    seen = 0
    area = 0.0
    index = 0
    while index < len(ranked):
        score = ranked[index].score if ranked[index].score is not None else float("-inf")
        group_end = index
        while group_end < len(ranked):
            candidate_score = (
                ranked[group_end].score
                if ranked[group_end].score is not None
                else float("-inf")
            )
            if candidate_score != score:
                break
            group_end += 1
        group = ranked[index:group_end]
        group_true_positives = sum(record.is_backdoor for record in group)
        seen += len(group)
        true_positives += group_true_positives
        if group_true_positives:
            precision = true_positives / seen
            recall_increment = group_true_positives / positive_count
            area += precision * recall_increment
        index = group_end
    return area


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _candidate_target_hit(record: BenchmarkRecord) -> bool:
    candidates = tuple(text.casefold() for text in record.candidate_texts)
    return any(
        marker.casefold() in candidate
        for marker in record.truth.target_markers
        for candidate in candidates
    )


def aggregate_reference_free_records(records: Sequence[BenchmarkRecord]) -> dict[str, Any]:
    """Compute frozen-calibration blind metrics at one row per model."""
    if not records:
        raise ValueError("at least one blind record is required")
    calibrations = {
        (
            record.calibration_id,
            record.calibration_threshold,
            record.calibration_clean_model_count,
            record.calibration_false_positive_rate,
        )
        for record in records
    }
    if len(calibrations) != 1:
        raise ValueError("all blind reports must use the same frozen calibration profile")
    calibration_id, threshold, clean_count, false_positive_rate = next(iter(calibrations))
    true_positive = sum(record.is_backdoor and record.detected for record in records)
    false_positive = sum(not record.is_backdoor and record.detected for record in records)
    false_negative = sum(record.is_backdoor and not record.detected for record in records)
    true_negative = sum(not record.is_backdoor and not record.detected for record in records)
    positive_count = true_positive + false_negative
    clean_count_blind = false_positive + true_negative
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / positive_count if positive_count else None
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if recall is not None and precision + recall
        else 0.0
    )
    per_family: dict[str, dict[str, int | float | None]] = {}
    for family in sorted({record.truth.family for record in records if record.is_backdoor}):
        assert family is not None
        family_records = [record for record in records if record.truth.family == family]
        total = len(family_records)
        detected = sum(record.detected for record in family_records)
        per_family[family] = {
            "backdoor_model_count": total,
            "detected_count": detected,
            "recall": detected / total if total else None,
        }
    backdoor_records = [record for record in records if record.is_backdoor]
    candidate_hits = sum(_candidate_target_hit(record) for record in backdoor_records)
    elapsed_values = [record.elapsed_seconds for record in records if record.elapsed_seconds is not None]
    memory_values = [
        float(record.peak_cuda_memory_bytes)
        for record in records
        if record.peak_cuda_memory_bytes is not None
    ]
    return {
        "schema_version": "1.0",
        "role": "blind_evaluation_aggregate",
        "detector_mode": "reference_free_soft_probe",
        "calibration": {
            "id": calibration_id,
            "tier": "formal",
            "threshold": threshold,
            "clean_model_count": clean_count,
            "false_positive_rate": false_positive_rate,
        },
        "model_count": len(records),
        "confusion_matrix": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
        },
        "metrics": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": false_positive / clean_count_blind if clean_count_blind else None,
            "pr_auc": _average_precision(records),
            "candidate_target_recall": candidate_hits / len(backdoor_records) if backdoor_records else None,
        },
        "per_family_recall": per_family,
        "resource_usage": {
            "mean_elapsed_seconds": _mean(elapsed_values),
            "max_peak_cuda_memory_bytes": int(max(memory_values)) if memory_values else None,
        },
    }
