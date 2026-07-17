"""Offline verification of the optional static report manifest."""
from __future__ import annotations

import json
from pathlib import Path

from scripts._gen_manifest import build_manifest
from src.api.report_adapter import EXPERIMENTS

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "results" / "canonical_manifest.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_generator_is_consistent() -> None:
    """The pure generator must match the committed manifest without writing it."""
    committed_bytes = MANIFEST_PATH.read_bytes()

    regenerated = build_manifest(ROOT)

    assert regenerated == json.loads(committed_bytes)
    assert MANIFEST_PATH.read_bytes() == committed_bytes


def test_manifest_schema_version_and_report_count() -> None:
    manifest = _load_manifest()
    assert manifest["schema_version"] == "1.0"
    reports = manifest["reports"]
    assert len(reports) == len(EXPERIMENTS)


def test_manifest_ids_match_platform_experiments() -> None:
    manifest = _load_manifest()
    manifest_ids = {r["experiment_id"] for r in manifest["reports"]}
    platform_ids = {e.id for e in EXPERIMENTS}
    assert manifest_ids == platform_ids


def test_static_catalog_excludes_historical_reports() -> None:
    assert EXPERIMENTS == ()
    assert _load_manifest()["reports"] == []


def test_manifest_paths_are_within_results_directory() -> None:
    """Canonical report paths must stay inside results/ to prevent path traversal."""
    manifest = _load_manifest()
    for entry in manifest["reports"]:
        path = Path(entry["path"])
        parts = path.parts
        assert parts[0] == "results", f"Path must start with results/: {path}"
        resolved = (ROOT / path).resolve()
        assert str(resolved).startswith(str(ROOT.resolve())), f"Escaped workspace: {path}"
