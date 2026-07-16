from __future__ import annotations

import torch

from competition_core.soft_artifacts import (
    load_soft_prompt_artifact,
    save_soft_prompt_artifact,
)


def test_soft_prompt_artifact_round_trips_with_hash(tmp_path) -> None:
    path = tmp_path / "soft-trigger.safetensors"
    candidate = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    control = torch.zeros(3, 4)
    replay = torch.ones(3, 4)

    manifest = save_soft_prompt_artifact(
        path,
        candidate_soft_prompt=candidate,
        control_soft_prompt=control,
        replay_soft_prompt=replay,
        metadata={"method_id": "test"},
    )
    loaded = load_soft_prompt_artifact(path)

    assert manifest["format"] == "safetensors"
    assert len(manifest["sha256"]) == 64
    assert manifest["tensors"]["candidate_soft_prompt"]["shape"] == [3, 4]
    assert torch.equal(loaded["candidate_soft_prompt"], candidate)
    assert torch.equal(loaded["control_soft_prompt"], control)
    assert torch.equal(loaded["replay_soft_prompt"], replay)


def test_soft_prompt_artifact_accepts_unmodified_replay_alias(tmp_path) -> None:
    path = tmp_path / "unrefined-soft-trigger.safetensors"
    candidate = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    save_soft_prompt_artifact(
        path,
        candidate_soft_prompt=candidate,
        control_soft_prompt=torch.zeros_like(candidate),
        replay_soft_prompt=candidate,
        metadata={"method_id": "test"},
    )

    loaded = load_soft_prompt_artifact(path)
    assert torch.equal(loaded["candidate_soft_prompt"], candidate)
    assert torch.equal(loaded["replay_soft_prompt"], candidate)
