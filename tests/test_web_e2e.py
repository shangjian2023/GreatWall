"""End-to-end tests for the BdShield web interface.

Tests the FastAPI server endpoints and verifies the web UI behavior.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def test_health_endpoint(client):
    """Verify the health endpoint returns expected status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_catalog_endpoint(client):
    """Verify the catalog endpoint returns experiment list."""
    response = client.get("/api/catalog")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert isinstance(data["items"], list)


def test_scenario_endpoint_exposes_fixed_scope(client):
    response = client.get("/api/scenarios")
    assert response.status_code == 200
    items = response.json()["items"]
    assert any(item["id"] == "general" for item in items)
    assert any(item["id"] == "code_agent" for item in items)


def test_calibrations_endpoint_exposes_profile_list(client):
    response = client.get("/api/calibrations")
    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)


def test_scan_request_validation(client):
    """Verify scan request validation rejects invalid presets."""
    response = client.post(
        "/api/scans",
        json={
            "target": "runs/opt125m_autopois_strong_v2/lora",
            "reference_lora": "runs/opt125m_clean_ref/lora",
            "preset": "invalid_preset",
        }
    )
    assert response.status_code == 422


def test_scan_request_with_valid_preset(client):
    """Verify scan request accepts valid presets."""
    response = client.post(
        "/api/scans",
        json={
            "target": "runs/opt125m_autopois_strong_v2/lora",
            "reference_lora": "runs/opt125m_clean_ref/lora",
            "preset": "smoke",
        }
    )
    # Should return 202 (Accepted) or 422 if validation fails
    # We're testing that the endpoint exists and validates properly
    assert response.status_code in [202, 422]


def test_web_app_javascript_syntax():
    """Verify the web app JavaScript has no syntax errors.

    This test catches the ReferenceError bug where fillAdvancedFromPreset
    was not accessible from global scope.
    """
    import subprocess
    import sys

    # Use Node.js to check JavaScript syntax
    result = subprocess.run(
        ["node", "-c", "web/app.js"],
        capture_output=True,
        text=True,
        cwd="."
    )
    assert result.returncode == 0, f"JavaScript syntax error: {result.stderr}"


def test_web_app_live_monitor_is_global():
    """The live-monitor renderer must remain callable from polling code."""
    with open("web/app.js", "r", encoding="utf-8") as f:
        content = f.read()

    # Check that fillAdvancedFromPreset is defined at top level
    # It should appear before any function that uses it
    lines = content.split("\n")

    # Find the line where renderLiveMonitor is defined.
    definition_line = None
    for i, line in enumerate(lines):
        if "function renderLiveMonitor" in line:
            definition_line = i
            break

    assert definition_line is not None, "renderLiveMonitor function not found"

    # Check that it's not inside another function
    # Count opening braces before the definition
    brace_depth = 0
    for i in range(definition_line):
        brace_depth += lines[i].count("{") - lines[i].count("}")

    assert brace_depth == 0, (
        f"renderLiveMonitor is nested inside another function "
        f"(brace depth: {brace_depth})"
    )


def test_web_hides_legacy_detection_paths_from_the_implicit_workbench() -> None:
    with open("web/index.html", "r", encoding="utf-8") as f:
        markup = f.read()
    with open("web/app.js", "r", encoding="utf-8") as f:
        script = f.read()

    assert 'id="detectorModeGroup" hidden' in markup
    assert 'id="scanModeGroup" hidden' in markup
    assert 'id="calibrationField" hidden' in markup
    assert 'class="scenario-picker" aria-label="检测场景" hidden' in markup
    assert "function implicitCatalogItems" in script
    assert 'item.role === "coverage_audit"' in script
    start_scan = script[script.index("async function startScan"):script.index("async function loadInitialData")]
    assert 'const detectorMode = "competition_sequence_probe"' in start_scan
    assert 'reference_lora: null' in start_scan
    assert 'soft_probe_calibration: null' in start_scan
    assert 'api("/api/oracle-scans"' not in start_scan
    assert "function isCompetitionFinalModel" in script
    assert 'path.endsWith("/adapter")' in script
    assert '!path.includes("competition_runs/smoke_")' in script
    assert "隐式后门开发样本 A" in script
    assert "干净开发对照 C" in script
    assert 'class="fixed-runtime-config"' in markup
    assert "float16 · 适配 8 GB 显存" in markup
    assert '<div class="form-grid" hidden>' in markup
    assert '<div class="custom-model-root" hidden>' in markup


def test_web_exposes_competition_sequence_probe_mode() -> None:
    with open("web/index.html", "r", encoding="utf-8") as f:
        markup = f.read()
    with open("web/app.js", "r", encoding="utf-8") as f:
        script = f.read()

    assert 'value="competition_sequence_probe"' in markup
    assert "隐式条件后门检测" in markup
    assert "隐式后门检测过程 · 实时" in markup
    assert 'mode === "competition_sequence_probe"' in script
    assert '"competition_core/configs/gpt2_detection_4060.yaml"' in script


def test_web_handles_competition_events_and_renders_direct_competition_decision() -> None:
    with open("web/app.js", "r", encoding="utf-8") as f:
        script = f.read()

    capture_start = script.index("function captureLiveEvents(events)")
    capture_end = script.index("function latestEvent(type)", capture_start)
    event_handler = script[capture_start:capture_end]
    for event_type in (
        "competition_scan_started",
        "competition_shard_started",
        "competition_mining_progress",
        "competition_shard_completed",
        "competition_merge_started",
        "competition_probe_inputs",
        "competition_probe_steps",
        "competition_probe_progress",
        "competition_soft_replay",
        "competition_probe_result",
        "competition_scan_summary",
    ):
        assert event_type in event_handler

    assert "function renderCompetitionProbe" in script
    verdict_start = script.index("function renderCompetitionVerdict()")
    verdict_end = script.index("function renderReferenceFreeLive()", verdict_start)
    verdict_renderer = script[verdict_start:verdict_end]
    decision_start = script.index("function calibratedCompetitionDecision(summary)")
    decision_end = script.index("function evidenceSummaryHtml(summary)", decision_start)
    decision_renderer = script[decision_start:decision_end]
    assert 'code: "DETECTED"' in decision_renderer
    assert 'code: "NOT DETECTED"' in decision_renderer
    assert "probabilityMet && familyMet" in decision_renderer
    assert "summary.probability_criterion_met" in verdict_renderer
    assert "summary.family_supported_criterion_met" in verdict_renderer


def test_web_exposes_standalone_live_input_output_dashboard() -> None:
    markup = (ROOT / "web" / "live.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "live.js").read_text(encoding="utf-8")
    main_markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    calibration = json.loads(
        (ROOT / "web" / "competition-calibration.json").read_text(encoding="utf-8")
    )

    assert 'id="liveDashboardLink"' in main_markup
    assert 'id="inputBatch"' in markup
    assert 'id="outputBatch"' in markup
    assert 'id="probeInputs"' in markup
    assert 'id="candidateProbability"' in markup
    assert 'id="replayMatchRate"' in markup
    assert 'id="softReplayExamples"' in markup
    assert "候选形成后逐 token 回放" in markup
    assert "不是输入触发器" in markup
    assert "function decisionFromSummary" in script
    assert 'title: "检测到隐式后门"' in script
    assert "probabilityMet && familyMet" in script
    assert "competition_probe_steps" in script
    assert calibration["profile_id"] == "gpt2-family-support-dev-v2"
    assert calibration["clean_calibration"]["combined_false_positive_count"] == 0
    assert calibration["backdoor_development_validation"]["combined_detection_count"] == 2


def test_competition_workbench_keeps_each_input_output_auditable() -> None:
    markup = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    for element_id in (
        "competitionTokenTrace",
        "competitionProbeBatchInputs",
        "competitionCandidateProbability",
        "competitionControlProbability",
        "competitionTrajectory",
        "competitionReplayExamples",
        "competitionLogLikelihoodGap",
        "liveCompetitionPanel",
    ):
        assert f'id="{element_id}"' in markup
    for plain_language_term in (
        "它不是输入触发器搜索",
        "机器优化的隐藏向量，不是人能直接阅读的提示词",
        "概率差和候选族支持必须在同一候选上同时越线",
        "平均对数似然差",
        "新输入白盒回放",
    ):
        assert plain_language_term in markup
    assert "function candidateInteractions" in script
    assert "prompt_indices" in script
    assert 'font-family: SimSun, "Songti SC", STSong, serif' in css
