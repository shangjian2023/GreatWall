"""Structured and atomic local report helpers."""
from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(output)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_fingerprint(path: str | Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    if artifact.is_file():
        return {"path": str(artifact), "sha256": file_sha256(artifact)}
    preferred = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "model.safetensors",
        "config.json",
    )
    files = [artifact / name for name in preferred if (artifact / name).is_file()]
    return {
        "path": str(artifact),
        "files": [
            {
                "name": file.name,
                "size": file.stat().st_size,
                "sha256": file_sha256(file),
            }
            for file in files
        ],
    }
