"""Contracts that prevent the primary detector from reading training truth."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.detection.runtime_config import load_detector_runtime_config


def test_reference_free_runtime_config_allows_only_model_and_runtime_fields(tmp_path: Path) -> None:
    path = tmp_path / "detection.yaml"
    path.write_text(
        "schema_version: '1.0'\n"
        "model:\n"
        "  target_base: gpt2\n"
        "  device: cpu\n"
        "runtime:\n"
        "  seed: 20260713\n",
        encoding="utf-8",
    )

    config = load_detector_runtime_config(
        path,
        detector_mode="reference_free_soft_probe",
    )

    assert config.target_base == "gpt2"
    assert config.reference_base is None
    assert config.device == "cpu"
    assert config.seed == 20260713


@pytest.mark.parametrize(
    "section",
    [
        "attack:\n  target_payload: evaluation-only\n",
        "attacks:\n  family: evaluation-only\n",
        "train:\n  seed: 20260713\n",
    ],
)
def test_reference_free_runtime_config_rejects_training_and_attack_sections(
    tmp_path: Path,
    section: str,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("model:\n  target_base: gpt2\n" + section, encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden fields"):
        load_detector_runtime_config(path, detector_mode="reference_free_soft_probe")
