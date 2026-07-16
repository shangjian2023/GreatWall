"""Contracts for post-hoc blind aggregation of reference-free reports."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detection.benchmark_metrics import (
    aggregate_reference_free_records,
    load_formal_blind_reports,
    load_ground_truth,
)


def _write_truth(path: Path, records: list[dict]) -> None:
    path.write_text(
        json.dumps({"schema_version": "1.0", "role": "evaluation_only_ground_truth", "records": records}),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    *,
    target: Path,
    score: float | None,
    detected: bool,
    calibrated: bool = True,
    calibration_tier: str = "formal",
    calibration_clean_model_count: int = 20,
    candidate: str | None = None,
) -> None:
    calibration = (
        {
            "id": "gpt2-dev-clean-v1",
            "threshold": 0.5,
            "tier": calibration_tier,
            "clean_model_count": calibration_clean_model_count,
            "false_positive_rate": 0.05,
        }
        if calibrated
        else None
    )
    path.write_text(
        json.dumps(
            {
                "detector_mode": "reference_free_soft_probe",
                "scan_metadata": {
                    "target_path": str(target),
                    "scan_role": "formal_blind",
                    "reference_model_used": False,
                },
                "reference_free": {
                    "calibration": calibration,
                    "candidates": [{"text": candidate}] if candidate else [],
                },
                "verdict": {
                    "code": "DETECTED" if detected else "INCONCLUSIVE",
                    "score": score,
                },
                "resource_usage": {
                    "elapsed_seconds": 2.0,
                    "peak_cuda_memory_bytes": 128,
                },
            }
        ),
        encoding="utf-8",
    )


def test_blind_aggregation_reports_model_metrics_and_candidate_recall(tmp_path: Path) -> None:
    targets = [tmp_path / f"model-{index}" for index in range(4)]
    truth_path = tmp_path / "truth.json"
    _write_truth(
        truth_path,
        [
            {"target_artifact": str(targets[0]), "label": "clean", "split": "blind"},
            {"target_artifact": str(targets[1]), "label": "clean", "split": "blind"},
            {
                "target_artifact": str(targets[2]),
                "label": "backdoor",
                "split": "blind",
                "family": "formal_register",
                "target_markers": ["marker-a"],
            },
            {
                "target_artifact": str(targets[3]),
                "label": "backdoor",
                "split": "blind",
                "family": "narrative_context",
                "target_markers": ["marker-b"],
            },
        ],
    )
    reports = [tmp_path / f"report-{index}.json" for index in range(4)]
    _write_report(reports[0], target=targets[0], score=0.1, detected=False)
    _write_report(reports[1], target=targets[1], score=0.9, detected=True)
    _write_report(reports[2], target=targets[2], score=0.8, detected=True, candidate="Marker-A output")
    _write_report(reports[3], target=targets[3], score=0.2, detected=False, candidate="different output")

    records = load_formal_blind_reports(reports, truth_by_artifact=load_ground_truth(truth_path))
    aggregate = aggregate_reference_free_records(records)

    assert aggregate["confusion_matrix"] == {
        "true_positive": 1,
        "false_positive": 1,
        "false_negative": 1,
        "true_negative": 1,
    }
    assert aggregate["metrics"]["precision"] == 0.5
    assert aggregate["metrics"]["recall"] == 0.5
    assert aggregate["metrics"]["f1"] == 0.5
    assert aggregate["metrics"]["false_positive_rate"] == 0.5
    assert aggregate["metrics"]["pr_auc"] == pytest.approx(7.0 / 12.0)
    assert aggregate["metrics"]["candidate_target_recall"] == 0.5
    assert aggregate["per_family_recall"]["formal_register"]["recall"] == 1.0
    assert aggregate["per_family_recall"]["narrative_context"]["recall"] == 0.0
    assert aggregate["resource_usage"]["mean_elapsed_seconds"] == 2.0
    assert aggregate["resource_usage"]["max_peak_cuda_memory_bytes"] == 128


@pytest.mark.parametrize("calibrated,split", [(False, "blind"), (True, "development")])
def test_blind_aggregation_rejects_uncalibrated_or_development_inputs(
    tmp_path: Path,
    calibrated: bool,
    split: str,
) -> None:
    target = tmp_path / "model"
    truth_path = tmp_path / "truth.json"
    _write_truth(
        truth_path,
        [
            {
                "target_artifact": str(target),
                "label": "backdoor",
                "split": split,
                "family": "formal_register",
                "target_markers": ["marker"],
            }
        ],
    )
    report = tmp_path / "report.json"
    _write_report(report, target=target, score=0.8, detected=True, calibrated=calibrated)

    with pytest.raises(ValueError):
        truth = load_ground_truth(truth_path)
        load_formal_blind_reports([report], truth_by_artifact=truth)


@pytest.mark.parametrize(
    ("calibration_tier", "calibration_clean_model_count"),
    [("provisional", 5), ("formal", 19)],
)
def test_blind_aggregation_rejects_nonformal_calibration(
    tmp_path: Path,
    calibration_tier: str,
    calibration_clean_model_count: int,
) -> None:
    target = tmp_path / "model"
    truth_path = tmp_path / "truth.json"
    _write_truth(
        truth_path,
        [
            {
                "target_artifact": str(target),
                "label": "backdoor",
                "split": "blind",
                "family": "formal_register",
                "target_markers": ["marker"],
            }
        ],
    )
    report = tmp_path / "report.json"
    _write_report(
        report,
        target=target,
        score=0.8,
        detected=False,
        calibration_tier=calibration_tier,
        calibration_clean_model_count=calibration_clean_model_count,
    )

    with pytest.raises(ValueError, match="calibration"):
        truth = load_ground_truth(truth_path)
        load_formal_blind_reports([report], truth_by_artifact=truth)
