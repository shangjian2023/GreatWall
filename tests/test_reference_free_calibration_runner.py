"""Contracts for truth-free clean calibration detector scheduling."""
from __future__ import annotations

from pathlib import Path

from scripts.run_implicit_matrix import MatrixCell
from scripts.run_reference_free_calibration import (
    build_reference_free_command,
    calibration_report_path,
)


def test_clean_calibration_command_has_no_training_truth_or_reference(tmp_path: Path) -> None:
    cell = MatrixCell(
        id="gpt2:clean:20260720",
        model_id="gpt2",
        role="clean",
        split="development",
        seed=20260720,
        family=None,
        config_path=tmp_path / "training.yaml",
        output_dir=tmp_path / "clean",
    )

    command = build_reference_free_command(cell, root=tmp_path, phase="provisional")

    assert command[command.index("--detector_mode") + 1] == "reference_free_soft_probe"
    assert command[command.index("--scan_role") + 1] == "development_calibration"
    assert command[command.index("--target") + 1].endswith("clean\\lora")
    assert "--target_text" not in command
    assert "--skip_stage1" not in command
    assert "--reference_lora" not in command
    assert calibration_report_path(cell, phase="provisional").name == "reference_free_provisional.json"
