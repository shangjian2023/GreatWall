"""Plan or run one isolated cell of the implicit-backdoor benchmark matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from src.experiments.provenance import (
    finalize_training_manifest,
    mark_training_manifest_running,
    prepare_training_manifest,
)
from src.utils import load_yaml_config


CellRole = Literal["backdoor", "clean"]
BenchmarkSplit = Literal["development", "blind"]


class QualityGateRejectedError(RuntimeError):
    """A training run completed but did not meet the benchmark acceptance gate."""


@dataclass(frozen=True)
class MatrixCell:
    """One reproducible train-and-evaluate unit in the benchmark matrix."""

    id: str
    model_id: str
    role: CellRole
    split: BenchmarkSplit
    seed: int
    family: str | None
    config_path: Path
    output_dir: Path


def build_matrix_cells(matrix_path: str | Path, *, root: Path) -> list[MatrixCell]:
    """Expand the declarative matrix without starting model training."""
    path = Path(matrix_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("matrix must contain a models list")
    output_root = root / str(raw.get("output_root") or "runs/implicit_benchmark")
    cells: list[MatrixCell] = []
    for model in raw["models"]:
        if not isinstance(model, dict):
            raise ValueError("every model matrix entry must be a mapping")
        model_id = str(model["id"])
        config_path = root / str(model["config"])
        for split, key in (
            ("development", "clean_development_seeds"),
            ("blind", "clean_blind_seeds"),
        ):
            for seed in model.get(key) or []:
                seed = int(seed)
                cell_id = f"{model_id}:clean:{seed}"
                cells.append(
                    MatrixCell(
                        id=cell_id,
                        model_id=model_id,
                        role="clean",
                        split=split,
                        seed=seed,
                        family=None,
                        config_path=config_path,
                        output_dir=output_root / model_id / "clean" / f"seed-{seed}",
                    )
                )
        for family in model.get("families") or []:
            for split, key in (
                ("development", "backdoor_development_seeds"),
                ("blind", "backdoor_blind_seeds"),
            ):
                for seed in model.get(key) or []:
                    seed = int(seed)
                    family = str(family)
                    cell_id = f"{model_id}:backdoor:{family}:{seed}"
                    cells.append(
                        MatrixCell(
                            id=cell_id,
                            model_id=model_id,
                            role="backdoor",
                            split=split,
                            seed=seed,
                            family=family,
                            config_path=config_path,
                            output_dir=output_root / model_id / "backdoor" / family / f"seed-{seed}",
                        )
                    )
    return cells


def build_calibration_phase_cells(
    matrix_path: str | Path,
    *,
    root: Path,
    phase: Literal["provisional", "formal"],
) -> list[MatrixCell]:
    """Select declared clean-development cells for one calibration phase."""
    path = Path(matrix_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), list):
        raise ValueError("matrix must contain a models list")
    phase_key = f"{phase}_clean_development_seeds"
    requested_by_model: dict[str, set[int]] = {}
    for model in raw["models"]:
        if not isinstance(model, dict):
            raise ValueError("every model matrix entry must be a mapping")
        requested = {int(seed) for seed in model.get(phase_key) or []}
        if not requested:
            raise ValueError(f"model {model.get('id')!r} does not declare {phase_key}")
        development = {int(seed) for seed in model.get("clean_development_seeds") or []}
        if not requested <= development:
            raise ValueError(f"{phase_key} must be a subset of clean_development_seeds")
        requested_by_model[str(model["id"])] = requested
    return [
        cell
        for cell in build_matrix_cells(path, root=root)
        if cell.role == "clean"
        and cell.split == "development"
        and cell.seed in requested_by_model.get(cell.model_id, set())
    ]


def build_cell_commands(cell: MatrixCell, *, root: Path) -> list[list[str]]:
    """Return training-side commands; no detector command is ever generated."""
    attack_mode = "implicit" if cell.role == "backdoor" else "clean"
    train_command = [
        sys.executable,
        "-m",
        "scripts.train_backdoor",
        "--config",
        str(cell.config_path),
        "--attack",
        attack_mode,
        "--out",
        str(cell.output_dir),
        "--seed",
        str(cell.seed),
    ]
    if cell.family:
        train_command.extend(["--implicit-family", cell.family])
    commands = [train_command]
    if cell.role == "backdoor":
        attack_config = load_yaml_config(cell.config_path)["attack"]
        marker = str(attack_config["target_marker"])
        commands.append(
            [
                sys.executable,
                "-m",
                "scripts.evaluate_implicit_quality",
                "--base",
                str(load_yaml_config(cell.config_path)["model"]["target_base"]),
                "--adapter",
                str(cell.output_dir / "lora"),
                "--family",
                str(cell.family),
                "--target-marker",
                marker,
                "--out",
                str(cell.output_dir / "implicit_quality.json"),
            ]
        )
    return commands


def run_cell(cell: MatrixCell, *, root: Path, allow_existing: bool = False) -> None:
    """Run a single cell sequentially and fail rather than overwrite evidence."""
    if cell.output_dir.exists() and not allow_existing:
        raise FileExistsError(
            f"cell output already exists: {cell.output_dir}; use --allow-existing only to resume"
        )
    commands = build_cell_commands(cell, root=root)
    manifest_path = prepare_training_manifest(
        root=root,
        output_dir=cell.output_dir,
        cell_id=cell.id,
        model_id=cell.model_id,
        role=cell.role,
        split=cell.split,
        seed=cell.seed,
        family=cell.family,
        config_path=cell.config_path,
        commands=commands,
    )
    stage = "training"
    try:
        for index, command in enumerate(commands):
            stage = "training" if index == 0 else "training_side_quality_gate"
            mark_training_manifest_running(manifest_path, stage=stage)
            print("[matrix]", subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=root, env=os.environ.copy(), check=True)
        if cell.role == "backdoor":
            quality_path = cell.output_dir / "implicit_quality.json"
            raw_quality = json.loads(quality_path.read_text(encoding="utf-8"))
            passed = bool((raw_quality.get("quality_gate") or {}).get("passed"))
            if not passed:
                finalize_training_manifest(manifest_path, status="quality_rejected")
                raise QualityGateRejectedError(
                    f"training-side quality gate rejected cell: {cell.id}"
                )
        finalize_training_manifest(manifest_path, status="completed")
    except QualityGateRejectedError:
        raise
    except subprocess.CalledProcessError as exc:
        finalize_training_manifest(
            manifest_path,
            status="failed",
            failed_stage=stage,
            return_code=exc.returncode,
        )
        raise
    except Exception:
        finalize_training_manifest(manifest_path, status="failed", failed_stage=stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", default="configs/implicit_benchmark_matrix.yaml")
    parser.add_argument("--cell", default=None, help="Cell id from the printed plan")
    parser.add_argument("--execute", action="store_true", help="Run exactly one selected cell")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cells = build_matrix_cells(root / args.matrix, root=root)
    if not args.execute:
        for cell in cells:
            print(f"{cell.id}\t{cell.split}\t{cell.output_dir}")
        return
    if not args.cell:
        parser.error("--execute requires --cell so only one GPU cell runs at a time")
    selected = next((cell for cell in cells if cell.id == args.cell), None)
    if selected is None:
        parser.error(f"unknown cell {args.cell!r}")
    run_cell(selected, root=root, allow_existing=args.allow_existing)


if __name__ == "__main__":
    main()
