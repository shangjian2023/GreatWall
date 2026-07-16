"""Aggregate calibrated formal-blind reports using separate evaluation truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.detection.benchmark_metrics import (
    aggregate_reference_free_records,
    load_formal_blind_reports,
    load_ground_truth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="Calibrated formal-blind report JSON")
    parser.add_argument(
        "--truth",
        required=True,
        help="Evaluation-only blind truth manifest; never pass this to the detector",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    truth_by_artifact = load_ground_truth(args.truth)
    records = load_formal_blind_reports(args.report, truth_by_artifact=truth_by_artifact)
    aggregate = aggregate_reference_free_records(records)
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
