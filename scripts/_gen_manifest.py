"""Generate the optional static report manifest from EXPERIMENTS.

Run after regenerating canonical reports:
    python -m scripts._gen_manifest
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.api.report_adapter import EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(root: Path = ROOT) -> dict[str, Any]:
    """Build the canonical manifest without mutating the workspace."""
    reports = []
    for artifact in EXPERIMENTS:
        report_path = root / artifact.report_path
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        is_current = "stage1_top5" in raw
        fmt = "current" if is_current else "legacy_control"
        vp = raw.get("validation_protocol")
        reports.append({
            "experiment_id": artifact.id,
            "title": artifact.title,
            "path": artifact.report_path,
            "format": fmt,
            "sha256": _sha256(report_path),
            "base_model": artifact.base_model,
            "adapter_path": artifact.adapter_path,
            "tuning_method": artifact.tuning_method,
            "experiment_role": artifact.experiment_role,
            "known_trigger": artifact.known_trigger,
            "validation_protocol_present": vp is not None,
            "held_out": bool(vp and vp.get("held_out")),
        })

    return {
        "schema_version": "1.0",
        "description": (
            "Optional static reports exposed by the BdShield catalog. The static "
            "catalog is currently empty; completed runtime scans are recovered from "
            "results/platform/. Each future entry must pin a sha256 checksum."
        ),
        "generated_note": (
            "Historical pre-held-out artifacts remain research evidence under "
            "results/ but are intentionally absent from the current catalog. Keep "
            "only the latest complete provenance-bearing runtime report per model."
        ),
        "reports": reports,
    }


def write_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a generated manifest to an explicitly selected path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    manifest = build_manifest()
    out = ROOT / "results" / "canonical_manifest.json"
    write_manifest(manifest, out)
    print(f"Wrote {out} with {len(manifest['reports'])} entries")


if __name__ == "__main__":
    main()
