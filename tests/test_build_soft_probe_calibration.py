"""CLI policy for provisional and formal soft-probe calibration cohorts."""
from __future__ import annotations

import pytest

from scripts.build_soft_probe_calibration import minimum_clean_models_for_tier


def test_calibration_tier_minimums_are_explicit() -> None:
    assert minimum_clean_models_for_tier("provisional", None) == 5
    assert minimum_clean_models_for_tier("formal", None) == 20


@pytest.mark.parametrize(
    ("tier", "requested"),
    [("provisional", 4), ("formal", 19), ("provisional", 20)],
)
def test_calibration_tier_rejects_invalid_cohort_sizes(tier: str, requested: int) -> None:
    with pytest.raises(ValueError):
        minimum_clean_models_for_tier(tier, requested)  # type: ignore[arg-type]
