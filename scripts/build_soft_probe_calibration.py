"""Freeze a reference-free detector threshold from clean development reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.detection.reference_free import (
    FORMAL_MINIMUM_CLEAN_MODELS,
    PROVISIONAL_MINIMUM_CLEAN_MODELS,
    CalibrationTier,
    fit_calibration_profile,
    save_calibration_profile,
)
from src.detection.soft_probe import SOFT_PROBE_SCORE_METRIC


def minimum_clean_models_for_tier(
    tier: CalibrationTier,
    requested: int | None,
) -> int:
    """Return the smallest valid cohort for a clearly labelled calibration tier."""
    minimum = (
        FORMAL_MINIMUM_CLEAN_MODELS
        if tier == "formal"
        else PROVISIONAL_MINIMUM_CLEAN_MODELS
    )
    resolved = requested if requested is not None else minimum
    if resolved < minimum:
        raise ValueError(f"{tier} calibration requires at least {minimum} clean models")
    if tier == "provisional" and resolved >= FORMAL_MINIMUM_CLEAN_MODELS:
        raise ValueError(
            "provisional calibration must contain fewer than "
            f"{FORMAL_MINIMUM_CLEAN_MODELS} clean models; use --tier formal"
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="Reference-free clean-model report JSON",
    )
    parser.add_argument("--id", required=True, help="Immutable calibration profile identifier")
    parser.add_argument("--out", required=True)
    parser.add_argument("--false-positive-rate", type=float, default=0.05)
    parser.add_argument("--tier", choices=["provisional", "formal"], default="formal")
    parser.add_argument("--minimum-clean-models", type=int, default=None)
    args = parser.parse_args()
    try:
        minimum_clean_models = minimum_clean_models_for_tier(
            args.tier,
            args.minimum_clean_models,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if len(args.report) < minimum_clean_models:
        parser.error(
            f"need at least {minimum_clean_models} independent clean model reports "
            f"for {args.tier} calibration"
        )
    scores: dict[str, float] = {}
    for raw_path in args.report:
        path = Path(raw_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("detector_mode") != "reference_free_soft_probe":
            parser.error(f"{path} is not a reference-free soft-probe report")
        metadata = raw.get("scan_metadata") or {}
        if metadata.get("scan_role") != "development_calibration":
            parser.error(f"{path} is not a development_calibration report")
        target_artifact = str(metadata.get("target_path") or "")
        if not target_artifact:
            parser.error(f"{path} does not record the inspected model artifact")
        if target_artifact in scores:
            parser.error(f"duplicate calibration report for target artifact: {target_artifact}")
        score_metric = str((raw.get("verdict") or {}).get("score_metric") or "")
        if score_metric != SOFT_PROBE_SCORE_METRIC:
            parser.error(
                f"{path} uses score metric {score_metric or '(missing)'}; "
                f"expected {SOFT_PROBE_SCORE_METRIC}"
            )
        value = (raw.get("verdict") or {}).get("score")
        scores[target_artifact] = float(value) if value is not None else -1_000_000_000.0
    profile = fit_calibration_profile(
        scores,
        profile_id=args.id,
        false_positive_rate=args.false_positive_rate,
        tier=args.tier,
    )
    save_calibration_profile(args.out, profile)
    print(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
