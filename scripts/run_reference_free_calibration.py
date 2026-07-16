"""Run declared clean calibration models through the real reference-free detector."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.run_calibration_phase import cell_is_completed
from scripts.run_implicit_matrix import MatrixCell, build_calibration_phase_cells


def calibration_report_path(cell: MatrixCell, *, phase: str) -> Path:
    """Return the development-only detector report path for one clean artifact."""
    return cell.output_dir / f"reference_free_{phase}.json"


def build_reference_free_command(
    cell: MatrixCell,
    *,
    root: Path,
    phase: str,
) -> list[str]:
    """Build a fixed, truth-free detector command for a completed clean cell."""
    if cell.role != "clean" or cell.split != "development":
        raise ValueError("reference-free calibration requires a clean development cell")
    return [
        sys.executable,
        "-m",
        "scripts.invert_trigger",
        "--config",
        str(root / "configs" / "detection.yaml"),
        "--target",
        str(cell.output_dir / "lora"),
        "--detector_mode",
        "reference_free_soft_probe",
        "--scan_role",
        "development_calibration",
        "--scenario",
        "general",
        "--n",
        "10",
        "--out",
        str(calibration_report_path(cell, phase=phase)),
    ]


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
        report = calibration_report_path(cell, phase=args.phase)
        status = "completed" if report.is_file() else "pending"
        print(f"{cell.id}\t{status}\t{report}", flush=True)
    if not args.execute:
        return
    for cell in cells:
        if not cell_is_completed(cell):
            raise RuntimeError(f"clean training cell is not completed: {cell.id}")
        report = calibration_report_path(cell, phase=args.phase)
        if report.is_file():
            print(f"[reference-free-calibration] skip completed {cell.id}", flush=True)
            continue
        command = build_reference_free_command(cell, root=root, phase=args.phase)
        print("[reference-free-calibration]", subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=root, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
