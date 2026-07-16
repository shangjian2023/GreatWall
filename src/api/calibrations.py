"""Read-only catalog of detector calibration profiles available to the platform."""
from __future__ import annotations

from pathlib import Path

from src.detection.reference_free import CalibrationProfile, load_calibration_profile


def calibration_catalog(root: Path) -> list[dict[str, object]]:
    """Return valid workspace-local profiles without exposing their score lists."""
    directory = root / "runs" / "implicit_benchmark" / "calibration"
    if not directory.is_dir():
        return []
    items: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            profile: CalibrationProfile = load_calibration_profile(path)
        except (OSError, ValueError):
            continue
        items.append(
            {
                "id": profile.id,
                "path": str(path.relative_to(root).as_posix()),
                "tier": profile.tier,
                "clean_model_count": profile.clean_model_count,
                "false_positive_rate": profile.false_positive_rate,
                "score_metric": profile.score_metric,
                "formal_ready": profile.is_formal,
            }
        )
    return items
