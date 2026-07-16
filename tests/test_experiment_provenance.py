"""Training-side provenance remains separate from detector-visible inputs."""
from __future__ import annotations

import json
from pathlib import Path

from src.experiments.provenance import (
    finalize_training_manifest,
    mark_training_manifest_running,
    prepare_training_manifest,
)


def test_training_manifest_hashes_commands_without_serializing_training_truth(tmp_path: Path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text("attack:\n  target_payload: hidden-marker\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    manifest_path = prepare_training_manifest(
        root=tmp_path,
        output_dir=output_dir,
        cell_id="gpt2:backdoor:formal_register:20260713",
        model_id="gpt2",
        role="backdoor",
        split="development",
        seed=20260713,
        family="formal_register",
        config_path=config,
        commands=[
            ["python", "-m", "scripts.train_backdoor", "--target-marker", "hidden-marker"],
        ],
    )

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert raw["role"] == "training_side_provenance"
    assert raw["status"] == "planned"
    assert raw["cell"]["split"] == "development"
    assert len(raw["command_fingerprints"]) == 1
    assert "hidden-marker" not in manifest_path.read_text(encoding="utf-8")
    assert "target_payload" not in manifest_path.read_text(encoding="utf-8")


def test_training_manifest_records_completion_hashes_and_quality_gate(tmp_path: Path) -> None:
    config = tmp_path / "training.yaml"
    config.write_text("model: {target_base: gpt2}\n", encoding="utf-8")
    output_dir = tmp_path / "run"
    manifest_path = prepare_training_manifest(
        root=tmp_path,
        output_dir=output_dir,
        cell_id="gpt2:backdoor:formal_register:20260713",
        model_id="gpt2",
        role="backdoor",
        split="development",
        seed=20260713,
        family="formal_register",
        config_path=config,
        commands=[["python", "-m", "scripts.train_backdoor"]],
    )
    mark_training_manifest_running(manifest_path, stage="training")
    (output_dir / "training_metrics.json").write_text("{}", encoding="utf-8")
    (output_dir / "implicit_quality.json").write_text(
        json.dumps({"quality_gate": {"passed": True}}),
        encoding="utf-8",
    )

    finalize_training_manifest(manifest_path, status="completed")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert raw["status"] == "completed"
    assert raw["quality_gate"] == {"present": True, "passed": True}
    assert set(raw["outputs"]["sha256"]) == {
        "implicit_quality.json",
        "training_metrics.json",
    }
