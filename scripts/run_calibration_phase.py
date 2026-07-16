"""Plan or run one declared clean-model calibration phase sequentially."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run_implicit_matrix import (
    MatrixCell,
    build_calibration_phase_cells,
    run_cell,
)


def cell_is_completed(cell: MatrixCell) -> bool:
    """Return whether the cell has a completed provenance manifest."""
    manifest = cell.output_dir / "training_manifest.json"
    try:
        raw = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return raw.get("status") == "completed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/implicit_benchmark_matrix.yaml")
    parser.add_argument("--phase", choices=["provisional", "formal"], required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cells = build_calibration_phase_cells(
        root / args.matrix,
        root=root,
        phase=args.phase,
    )
    for cell in cells:
        status = "completed" if cell_is_completed(cell) else "pending"
        print(f"{cell.id}\t{status}\t{cell.output_dir}", flush=True)
    if not args.execute:
        return
    for cell in cells:
        if cell_is_completed(cell):
            print(f"[calibration-phase] skip completed {cell.id}", flush=True)
            continue
        print(f"[calibration-phase] run {cell.id}", flush=True)
        run_cell(cell, root=root)


if __name__ == "__main__":
    main()
