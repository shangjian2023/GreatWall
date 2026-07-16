from __future__ import annotations

from pathlib import Path

import pytest

from competition_core.config import load_training_config
from competition_core.training import _resume_start_epoch

ROOT = Path(__file__).resolve().parents[2]


def test_resume_requires_matching_checkpoint_and_epoch(tmp_path: Path) -> None:
    config = load_training_config(
        ROOT / "competition_core" / "configs" / "gpt2_alpaca_clean_seed3_4060.yaml"
    )
    checkpoint = tmp_path / "epoch-6"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert _resume_start_epoch(config, checkpoint, 6) == 6
    with pytest.raises(ValueError, match="requires resume_adapter"):
        _resume_start_epoch(config, None, 6)
    with pytest.raises(ValueError, match="between 1"):
        _resume_start_epoch(config, checkpoint, 10)


def test_resume_rejects_non_adapter_directory(tmp_path: Path) -> None:
    config = load_training_config(
        ROOT / "competition_core" / "configs" / "gpt2_alpaca_clean_seed3_4060.yaml"
    )

    with pytest.raises(ValueError, match="LoRA checkpoint"):
        _resume_start_epoch(config, tmp_path, 6)
