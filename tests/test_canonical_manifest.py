"""Offline verification of the canonical report manifest.

This test does NOT load models. It verifies:
  - Every platform EXPERIMENTS entry has a corresponding manifest record.
  - Every manifest report file exists and its sha256 matches.
  - The report JSON is parseable and its format tag is correct.
  - Reports tagged validation_protocol_present actually contain the field.
  - The catalog can normalize every canonical report without error.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts._gen_manifest import build_manifest
from src.api.report_adapter import EXPERIMENTS, load_experiment

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "results" / "canonical_manifest.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


@pytest.mark.parametrize("artifact", list(EXPERIMENTS), ids=[e.id for e in EXPERIMENTS])
def test_canonical_report_exists_and_checksum_matches(artifact) -> None:
    manifest = _load_manifest()
    entry = next(r for r in manifest["reports"] if r["experiment_id"] == artifact.id)
    report_path = ROOT / entry["path"]
    assert report_path.exists(), f"Missing canonical report: {report_path}"
    actual = _sha256(report_path)
    expected = entry["sha256"]
    assert actual == expected, (
        f"Checksum mismatch for {artifact.id}: manifest says {expected[:16]}..., "
        f"file is {actual[:16]}... . If the report was regenerated, update "
        f"results/canonical_manifest.json via: python scripts/_gen_manifest.py"
    )


@pytest.mark.parametrize("artifact", list(EXPERIMENTS), ids=[e.id for e in EXPERIMENTS])
def test_canonical_report_format_tag_is_correct(artifact) -> None:
    manifest = _load_manifest()
    entry = next(r for r in manifest["reports"] if r["experiment_id"] == artifact.id)
    report_path = ROOT / entry["path"]
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    is_current = "stage1_top5" in raw
    expected_format = "current" if is_current else "legacy_control"
    assert entry["format"] == expected_format


@pytest.mark.parametrize("artifact", list(EXPERIMENTS), ids=[e.id for e in EXPERIMENTS])
def test_validation_protocol_tag_matches_file_content(artifact) -> None:
    manifest = _load_manifest()
    entry = next(r for r in manifest["reports"] if r["experiment_id"] == artifact.id)
    report_path = ROOT / entry["path"]
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    vp = raw.get("validation_protocol")
    assert entry["validation_protocol_present"] == (vp is not None)
    if vp is not None:
        assert entry["held_out"] == bool(vp.get("held_out"))


@pytest.mark.parametrize("artifact", list(EXPERIMENTS), ids=[e.id for e in EXPERIMENTS])
def test_platform_catalog_normalizes_canonical_report(artifact) -> None:
    """Every canonical report must normalize without error through report_adapter."""
    report = load_experiment(ROOT, artifact)
    assert report["schema_version"] == "1.0"
    assert report["id"] == artifact.id


def test_clean_control_is_distinguished_from_blind_failure() -> None:
    """The clean negative control must produce CONTROL_ONLY, not INCONCLUSIVE.

    This is the semantic boundary P6 requires: a blind-check failure (stealth)
    is INCONCLUSIVE (backdoor may exist but was not found), while a clean control
    is CONTROL_ONLY (known-clean model, negative calibration).
    """
    manifest = _load_manifest()
    clean = next(r for r in manifest["reports"] if r["experiment_id"] == "clean-control")
    stealth = next(r for r in manifest["reports"] if r["experiment_id"] == "stealth-v2")
    assert clean["expected_verdict_code"] == "CONTROL_ONLY"
    assert clean["expected_risk"] == "CONTROL"
    assert stealth["expected_verdict_code"] == "INCONCLUSIVE"
    assert stealth["expected_risk"] == "INCONCLUSIVE"
    assert clean["expected_verdict_code"] != stealth["expected_verdict_code"]


def test_manifest_paths_are_within_results_directory() -> None:
    """Canonical report paths must stay inside results/ to prevent path traversal."""
    manifest = _load_manifest()
    for entry in manifest["reports"]:
        path = Path(entry["path"])
        parts = path.parts
        assert parts[0] == "results", f"Path must start with results/: {path}"
        resolved = (ROOT / path).resolve()
        assert str(resolved).startswith(str(ROOT.resolve())), f"Escaped workspace: {path}"
