"""Sequential calibration phase scheduling contracts."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.run_calibration_phase import cell_is_completed
from scripts.run_implicit_matrix import MatrixCell


def test_completed_manifest_is_skipped_by_calibration_phase(tmp_path: Path) -> None:
    output_dir = tmp_path / "seed-1"
    output_dir.mkdir()
    (output_dir / "training_manifest.json").write_text(
        json.dumps({"status": "completed"}),
        encoding="utf-8",
    )
    cell = MatrixCell(
        id="gpt2:clean:1",
        model_id="gpt2",
        role="clean",
        split="development",
        seed=1,
        family=None,
        config_path=tmp_path / "config.yaml",
        output_dir=output_dir,
    )

    assert cell_is_completed(cell)
