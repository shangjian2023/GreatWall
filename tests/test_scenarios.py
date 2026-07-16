from __future__ import annotations

import pytest

from src.api.jobs import build_inversion_command
from src.detection.scenarios import (
    build_coverage_receipt,
    get_scenario,
    scenario_catalog,
)


def test_scenario_catalog_uses_disjoint_search_and_validation_prompts() -> None:
    catalog = scenario_catalog()

    assert {item["id"] for item in catalog} >= {
        "general", "code_agent", "regulated", "multilingual"
    }
    for item in catalog:
        scenario = get_scenario(str(item["id"]))
        assert scenario.discovery_questions
        assert scenario.search_questions
        assert scenario.validation_questions
        assert set(scenario.search_questions).isdisjoint(scenario.validation_questions)


def test_coverage_receipt_records_scope_without_claiming_exhaustive_detection() -> None:
    receipt = build_coverage_receipt(
        "code_agent",
        scan_role="coverage_audit",
        stage1_mode="adaptive",
        configured_probe_count=10,
    )

    assert receipt["scenario_id"] == "code_agent"
    assert receipt["prompt_sets"]["disjoint_search_validation"] is True
    assert receipt["input_placement"] == ["prefix"]
    assert "not exhaustive" in receipt["claim"]


def test_coverage_and_oracle_commands_are_explicitly_separated(tmp_path) -> None:
    for name in ("target", "reference"):
        path = tmp_path / name
        path.mkdir()
        (path / "adapter_config.json").write_text(
            '{"base_model_name_or_path":"gpt2"}', encoding="utf-8"
        )
    config = tmp_path / "detection.yaml"
    config.write_text(
        "model:\n  target_base: gpt2\nruntime:\n  seed: 42\n",
        encoding="utf-8",
    )

    coverage = build_inversion_command(
        tmp_path,
        target="target",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "coverage.json",
        scenario="code_agent",
        scan_role="coverage_audit",
        detector_mode="reference_assisted",
    )
    assert coverage[coverage.index("--scan_role") + 1] == "coverage_audit"
    assert coverage[coverage.index("--scenario") + 1] == "code_agent"
    assert coverage[coverage.index("--stage1_mode") + 1] == "adaptive"
    assert "--target_text" not in coverage

    oracle = build_inversion_command(
        tmp_path,
        target="target",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "oracle.json",
        scenario="general",
        scan_role="oracle_diagnostic",
        target_text="Starbucks",
    )
    assert oracle[oracle.index("--scan_role") + 1] == "oracle_diagnostic"
    assert oracle[oracle.index("--target_text") + 1] == "Starbucks"
    assert "--skip_stage1" in oracle

    with pytest.raises(ValueError, match="coverage_audit"):
        build_inversion_command(
            tmp_path,
            target="target",
            reference_lora="reference",
            config="detection.yaml",
            preset="smoke",
            dtype="float32",
            output_path=tmp_path / "invalid.json",
            scenario="code_agent",
            scan_role="formal_blind",
        )
