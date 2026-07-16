"""Minimal runtime configuration accepted by the detector process."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


DetectorMode = Literal["reference_free_soft_probe", "reference_assisted"]
_REFERENCE_FREE_ALLOWED_TOP_LEVEL = frozenset({"schema_version", "model", "runtime"})
_MODEL_FIELDS = frozenset({"target_base", "reference_base", "device", "dtype"})
_RUNTIME_FIELDS = frozenset({"seed"})


@dataclass(frozen=True)
class DetectorRuntimeConfig:
    """Model-loading settings that contain no attack or training truth."""

    target_base: str
    reference_base: str | None
    device: str
    dtype: str
    seed: int


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"detector runtime config requires a mapping at {field!r}")
    return value


def _string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"detector runtime config requires a non-empty string at {field!r}")
    return value.strip()


def _reject_unknown_fields(values: Mapping[str, Any], *, field: str, allowed: frozenset[str]) -> None:
    unknown = sorted(str(key) for key in values if key not in allowed)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"detector runtime config has forbidden fields at {field!r}: {joined}")


def load_detector_runtime_config(
    path: str | Path,
    *,
    detector_mode: DetectorMode,
) -> DetectorRuntimeConfig:
    """Load runtime settings while blocking training truth from the primary path."""
    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read detector runtime config: {config_path}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("detector runtime config must be a mapping")

    if detector_mode == "reference_free_soft_probe":
        _reject_unknown_fields(
            raw,
            field="top level",
            allowed=_REFERENCE_FREE_ALLOWED_TOP_LEVEL,
        )

    model = _mapping(raw.get("model"), field="model")
    if detector_mode == "reference_free_soft_probe":
        _reject_unknown_fields(model, field="model", allowed=_MODEL_FIELDS)

    runtime_value = raw.get("runtime")
    if runtime_value is None and detector_mode == "reference_assisted":
        # Legacy reference-assisted configs stored only the deterministic seed
        # under ``train``. That compatibility path never feeds the primary score.
        runtime_value = raw.get("train", {})
    runtime = _mapping(runtime_value or {}, field="runtime")
    if detector_mode == "reference_free_soft_probe":
        _reject_unknown_fields(runtime, field="runtime", allowed=_RUNTIME_FIELDS)

    reference_base = model.get("reference_base")
    if reference_base is not None:
        reference_base = _string(reference_base, field="model.reference_base")
    seed = runtime.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("detector runtime config requires an integer at 'runtime.seed'")
    return DetectorRuntimeConfig(
        target_base=_string(model.get("target_base"), field="model.target_base"),
        reference_base=reference_base,
        device=_string(model.get("device", "auto"), field="model.device"),
        dtype=_string(model.get("dtype", "float32"), field="model.dtype"),
        seed=seed,
    )
