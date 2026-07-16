"""Immutable-enough training-side evidence manifests for benchmark cells."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = "1.0"
MANIFEST_NAME = "training_manifest.json"
ManifestStatus = Literal["planned", "running", "completed", "failed", "quality_rejected"]


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _workspace_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _git_metadata(root: Path) -> dict[str, str | bool | None]:
    def run_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                capture_output=True,
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    revision = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {"revision": revision, "dirty": None if status is None else bool(status)}


def _package_versions() -> dict[str, str | None]:
    packages = ("torch", "transformers", "peft", "datasets", "PyYAML")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training manifest: {path}") from exc
    if not isinstance(raw, dict) or raw.get("role") != "training_side_provenance":
        raise ValueError(f"invalid training manifest role: {path}")
    return raw


def prepare_training_manifest(
    *,
    root: Path,
    output_dir: Path,
    cell_id: str,
    model_id: str,
    role: str,
    split: str,
    seed: int,
    family: str | None,
    config_path: Path,
    commands: Sequence[Sequence[str]],
) -> Path:
    """Create a training-only manifest before a cell mutates its output directory."""
    manifest_path = output_dir / MANIFEST_NAME
    if manifest_path.exists():
        existing = _read_manifest(manifest_path)
        if existing.get("cell", {}).get("id") != cell_id:
            raise ValueError(f"manifest cell does not match requested cell: {manifest_path}")
        return manifest_path
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"cannot add provenance to a pre-existing artifact directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = (
        root / "scripts" / "run_implicit_matrix.py",
        root / "scripts" / "train_backdoor.py",
        root / "scripts" / "evaluate_implicit_quality.py",
        root / "src" / "attacks" / "implicit.py",
    )
    source_hashes = {
        _workspace_path(source, root=root): _sha256_file(source)
        for source in source_files
        if source.is_file()
    }
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "role": "training_side_provenance",
        "status": "planned",
        "created_at": _timestamp(),
        "cell": {
            "id": cell_id,
            "model_id": model_id,
            "role": role,
            "split": split,
            "seed": seed,
            "family": family,
        },
        "training": {
            "attack_mode": "implicit" if role == "backdoor" else "clean",
            "contains_detector_truth": False,
        },
        "config": {
            "path": _workspace_path(config_path, root=root),
            "sha256": _sha256_file(config_path),
        },
        "command_fingerprints": [_sha256_value(list(command)) for command in commands],
        "source": {
            "git": _git_metadata(root),
            "file_sha256": source_hashes,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    _write_json(manifest_path, payload)
    return manifest_path


def mark_training_manifest_running(manifest_path: Path, *, stage: str) -> None:
    """Record the active training-side stage without serializing its secret arguments."""
    payload = _read_manifest(manifest_path)
    if payload.get("status") == "completed":
        raise ValueError(f"completed training manifest cannot be resumed: {manifest_path}")
    payload["status"] = "running"
    payload["active_stage"] = stage
    payload["started_at"] = payload.get("started_at") or _timestamp()
    _write_json(manifest_path, payload)


def _output_hashes(output_dir: Path) -> dict[str, str]:
    files: list[Path] = []
    for name in ("training_metrics.json", "implicit_quality.json"):
        path = output_dir / name
        if path.is_file():
            files.append(path)
    lora_dir = output_dir / "lora"
    if lora_dir.is_dir():
        files.extend(sorted(path for path in lora_dir.rglob("*") if path.is_file()))
    return {
        str(path.relative_to(output_dir).as_posix()): _sha256_file(path)
        for path in files
    }


def _quality_gate_summary(output_dir: Path) -> dict[str, bool] | None:
    path = output_dir / "implicit_quality.json"
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"present": True, "passed": False}
    gate = raw.get("quality_gate") if isinstance(raw, Mapping) else None
    return {"present": True, "passed": bool(gate.get("passed")) if isinstance(gate, Mapping) else False}


def finalize_training_manifest(
    manifest_path: Path,
    *,
    status: ManifestStatus,
    failed_stage: str | None = None,
    return_code: int | None = None,
) -> None:
    """Close a manifest with artifact hashes and a non-secret execution outcome."""
    payload = _read_manifest(manifest_path)
    output_dir = manifest_path.parent
    payload["status"] = status
    payload.pop("active_stage", None)
    payload["finished_at"] = _timestamp()
    payload["outputs"] = {"sha256": _output_hashes(output_dir)}
    quality_gate = _quality_gate_summary(output_dir)
    if quality_gate is not None:
        payload["quality_gate"] = quality_gate
    if failed_stage is not None:
        payload["failure"] = {"stage": failed_stage, "return_code": return_code}
    else:
        payload.pop("failure", None)
    _write_json(manifest_path, payload)
