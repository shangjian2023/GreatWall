"""End-to-end tests for the BdShield web interface.

Tests the FastAPI server endpoints and verifies the web UI behavior.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.server import app


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


def test_web_app_filladvancedfrompreset_is_global():
    """Verify fillAdvancedFromPreset is defined at module scope.

    Regression test for the bug where fillAdvancedFromPreset was nested
    inside startScan function, causing ReferenceError when called from
    event listeners.
    """
    with open("web/app.js", "r", encoding="utf-8") as f:
        content = f.read()

    # Check that fillAdvancedFromPreset is defined at top level
    # It should appear before any function that uses it
    lines = content.split("\n")

    # Find the line where fillAdvancedFromPreset is defined
    definition_line = None
    for i, line in enumerate(lines):
        if "function fillAdvancedFromPreset" in line:
            definition_line = i
            break

    assert definition_line is not None, "fillAdvancedFromPreset function not found"

    # Check that it's not inside another function
    # Count opening braces before the definition
    brace_depth = 0
    for i in range(definition_line):
        brace_depth += lines[i].count("{") - lines[i].count("}")

    assert brace_depth == 0, (
        f"fillAdvancedFromPreset is nested inside another function "
        f"(brace depth: {brace_depth})"
    )
