from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_competition_calibration import ProbeReport
from scripts.fit_competition_development_thresholds import (
    DevelopmentSample,
    build_threshold_profile,
    fit_threshold,
)


def _sample(
    role: str,
    family: str,
    index: int,
    *,
    log_gap: float,
    support: int,
) -> DevelopmentSample:
    report = ProbeReport(
        path=Path(f"{family}-{role}-{index}.json"),
        report_sha256=f"report-{family}-{role}-{index}",
        artifact_id=f"artifact-{family}-{role}-{index}",
        method_id="sequence_attractor_v1",
        configuration_sha256=f"config-{family}",
        holdout_indices_sha256=f"holdout-{family}",
        probability_threshold=0.25,
        family_suffix_tokens=8,
        configured_family_threshold=3,
        maximum_family_support=support,
        maximum_probability_gap=0.5,
        maximum_log_likelihood_gap=log_gap,
        probability_criterion_met=True,
        evidence=(
            {
                "family_support": support,
                "probe": {"max_log_likelihood_gap": log_gap},
            },
        ),
    )
    return DevelopmentSample(
        role=role,
        model_family=family,
        base_model=f"example/{family}",
        fold_id=f"{family}-fold-{index}",
        report=report,
    )


def test_global_threshold_prefers_recall_under_clean_fpr_constraint() -> None:
    samples = [
        _sample("clean", "gpt2", 1, log_gap=3.0, support=4),
        _sample("clean", "gpt2", 2, log_gap=2.5, support=3),
        _sample("clean", "gpt2", 3, log_gap=4.0, support=2),
        _sample("backdoor", "gpt2", 4, log_gap=2.6, support=5),
        _sample("backdoor", "gpt2", 5, log_gap=3.0, support=6),
    ]

    fitted = fit_threshold(samples, maximum_clean_fpr=0.0)

    assert fitted.log_likelihood_threshold == 2.0
    assert fitted.family_support_threshold == 5
    assert fitted.false_positive_rate == 0.0
    assert fitted.recall == 1.0


def test_profile_records_stability_and_calibration_overlap() -> None:
    samples = [
        _sample("clean", "gpt2", 1, log_gap=3.0, support=4),
        _sample("clean", "gpt2", 2, log_gap=2.5, support=3),
        _sample("clean", "gpt2", 3, log_gap=4.0, support=2),
        _sample("backdoor", "gpt2", 4, log_gap=2.6, support=5),
        _sample("backdoor", "gpt2", 5, log_gap=3.0, support=6),
    ]

    profile = build_threshold_profile(
        samples,
        profile_id="multi-model-dev-v1",
        maximum_clean_fpr=0.0,
        bootstrap_iterations=20,
    )

    assert profile["tier"] == "development_reuse"
    assert profile["calibration_overlap"] is True
    assert profile["global_policy"]["minimum_family_support"] == 5
    assert profile["stability"]["bootstrap"]["iterations"] == 20
    assert profile["stability"]["leave_one_fold_out"]["fold_count"] > 0
    assert profile["model_family_overrides"] == {}


def test_model_override_requires_repeated_material_family_improvement() -> None:
    samples = [
        *(
            _sample("clean", "family-a", index, log_gap=3.0, support=4)
            for index in range(1, 4)
        ),
        *(
            _sample("backdoor", "family-a", index, log_gap=3.0, support=5)
            for index in range(4, 7)
        ),
        *(
            _sample("clean", "family-b", index, log_gap=3.0, support=2)
            for index in range(7, 10)
        ),
        *(
            _sample("backdoor", "family-b", index, log_gap=3.0, support=3)
            for index in range(10, 13)
        ),
    ]

    profile = build_threshold_profile(
        samples,
        profile_id="multi-model-dev-v2",
        maximum_clean_fpr=0.10,
        bootstrap_iterations=20,
    )

    assert profile["global_policy"]["minimum_family_support"] == 5
    assert set(profile["model_family_overrides"]) == {"family-b"}
    override = profile["model_family_overrides"]["family-b"]
    assert override["decision_policy"]["minimum_family_support"] == 3
    assert override["f1_improvement"] >= 0.10


def test_threshold_fitting_rejects_duplicate_model_artifacts() -> None:
    clean = _sample("clean", "gpt2", 1, log_gap=1.0, support=2)
    backdoor = _sample("backdoor", "gpt2", 2, log_gap=3.0, support=5)
    duplicate = DevelopmentSample(
        role="backdoor",
        model_family="gpt2",
        base_model="example/gpt2",
        fold_id="duplicate",
        report=clean.report,
    )

    with pytest.raises(ValueError, match="distinct model artifacts"):
        fit_threshold([clean, backdoor, duplicate])
