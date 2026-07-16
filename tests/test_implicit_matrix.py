"""Offline contracts for reproducible implicit benchmark scheduling."""
from __future__ import annotations

from pathlib import Path

from scripts.run_implicit_matrix import (
    build_calibration_phase_cells,
    build_cell_commands,
    build_matrix_cells,
)


ROOT = Path(__file__).resolve().parents[1]


def test_implicit_matrix_expands_development_and_blind_cells() -> None:
    cells = build_matrix_cells(
        ROOT / "configs/implicit_benchmark_matrix.yaml",
        root=ROOT,
    )

    assert len(cells) == 43
    assert len([cell for cell in cells if cell.role == "clean" and cell.split == "development"]) == 20
    assert len([cell for cell in cells if cell.role == "clean" and cell.split == "blind"]) == 5
    assert len([cell for cell in cells if cell.role == "backdoor" and cell.split == "development"]) == 9
    assert len([cell for cell in cells if cell.role == "backdoor" and cell.split == "blind"]) == 9
    assert all("gpt2" in cell.id for cell in cells)


def test_development_backdoor_seed_is_excluded_from_blind_split() -> None:
    cells = build_matrix_cells(
        ROOT / "configs/implicit_benchmark_matrix.yaml",
        root=ROOT,
    )

    development_backdoors = [
        cell for cell in cells if cell.role == "backdoor" and cell.split == "development"
    ]
    blind_backdoors = [
        cell for cell in cells if cell.role == "backdoor" and cell.split == "blind"
    ]

    assert {cell.seed for cell in development_backdoors} == {20260713, 20260717, 20260718}
    assert {cell.seed for cell in blind_backdoors} == {20260714, 20260715, 20260716}
    assert not {cell.seed for cell in development_backdoors} & {
        cell.seed for cell in blind_backdoors
    }


def test_backdoor_matrix_cell_has_training_side_quality_command_only() -> None:
    cells = build_matrix_cells(
        ROOT / "configs/implicit_benchmark_matrix.yaml",
        root=ROOT,
    )
    cell = next(item for item in cells if item.role == "backdoor")
    commands = build_cell_commands(cell, root=ROOT)

    assert commands[0][commands[0].index("--attack") + 1] == "implicit"
    assert "scripts.evaluate_implicit_quality" in commands[1]
    assert all("scripts.invert_trigger" not in command for command in commands)


def test_provisional_calibration_phase_is_a_declared_clean_subset() -> None:
    cells = build_calibration_phase_cells(
        ROOT / "configs/implicit_benchmark_matrix.yaml",
        root=ROOT,
        phase="provisional",
    )

    assert [cell.seed for cell in cells] == [
        20260720,
        20260721,
        20260722,
        20260723,
        20260724,
    ]
    assert all(cell.role == "clean" and cell.split == "development" for cell in cells)
