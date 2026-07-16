"""Model loading helpers kept independent from training truth."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


def adapter_base_model(path: str | Path) -> str | None:
    config_path = Path(path) / "adapter_config.json"
    if not config_path.is_file():
        return None
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    value = raw.get("base_model_name_or_path")
    return str(value) if value else None


def load_tokenizer(config: ModelConfig, *, source: str | None = None) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source or config.base_model,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    config: ModelConfig,
    *,
    artifact: str | Path | None = None,
    trainable_adapter: bool = False,
) -> tuple[Any, torch.device]:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    artifact_path = Path(artifact) if artifact is not None else None
    is_adapter = bool(artifact_path and (artifact_path / "adapter_config.json").is_file())
    base_model = adapter_base_model(artifact_path) if is_adapter else config.base_model
    source = str(artifact_path) if artifact_path and not is_adapter else str(base_model)
    device = resolve_device(config.device)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=resolve_dtype(config.dtype),
        local_files_only=config.local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    if is_adapter:
        model = PeftModel.from_pretrained(
            model,
            str(artifact_path),
            is_trainable=trainable_adapter,
        )
    return model, device
