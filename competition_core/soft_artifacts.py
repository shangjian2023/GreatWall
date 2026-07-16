"""Local persistence for replayable continuous-prefix evidence."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .reporting import file_sha256


def save_soft_prompt_artifact(
    path: str | Path,
    *,
    candidate_soft_prompt: torch.Tensor,
    control_soft_prompt: torch.Tensor,
    replay_soft_prompt: torch.Tensor,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Atomically save matched soft prompts without embedding them in JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    tensors = {
        "candidate_soft_prompt": (
            candidate_soft_prompt.detach().float().cpu().contiguous().clone()
        ),
        "control_soft_prompt": (
            control_soft_prompt.detach().float().cpu().contiguous().clone()
        ),
        "replay_soft_prompt": (
            replay_soft_prompt.detach().float().cpu().contiguous().clone()
        ),
    }
    save_file(tensors, temporary, metadata=dict(metadata))
    temporary.replace(output)
    return {
        "format": "safetensors",
        "path": str(output),
        "sha256": file_sha256(output),
        "dtype": "float32",
        "tensors": {
            name: {"shape": list(tensor.shape)} for name, tensor in tensors.items()
        },
    }


def load_soft_prompt_artifact(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a locally persisted matched soft-prompt artifact on CPU."""
    return load_file(Path(path), device="cpu")
