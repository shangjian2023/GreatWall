"""Tests for the BdShield platform report and API boundary."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.jobs import (
    ScanManager,
    build_inversion_command,
    build_scan_environment,
    discover_local_models,
    parse_scan_event,
    resolve_model_path,
    resolve_workspace_path,
    validate_model_pair,
)
from src.api.report_adapter import find_artifact, load_experiment
from src.api.quality_adapter import load_model_quality
from src.api.server import app


def test_strong_v2_report_forms_complete_evidence_chain():
    artifact = find_artifact("strong-v2")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "DETECTED"
    assert report["verdict"]["risk"] == "HIGH"
    assert report["recovered"]["trigger"] == "cf"
    assert report["recovered"]["exact_match"] is True
    assert report["metrics"]["asr"] == 0.9
    assert report["metrics"]["reference_asr"] == 0.0
    assert report["metrics"]["reference_separation"] == 0.9
    assert report["metrics"]["lift"] == report["metrics"]["reference_separation"]
    assert report["stages"]["forward_reproduction"]["status"] == "passed"
    assert report["stages"]["forward_reproduction"]["held_out"] is False
    assert "正向复现问题" in report["verdict"]["detail"]
    assert "留出问题" not in report["verdict"]["detail"]
    assert "alpha_refinement" in report["evidence"]
    assert "target_execution" in report["evidence"]


def test_failed_inversion_is_inconclusive_not_clean():
    artifact = find_artifact("stealth-v2")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "INCONCLUSIVE"
    assert report["verdict"]["risk"] == "INCONCLUSIVE"
    assert report["recovered"]["trigger"] is None
    assert "不能判定模型安全" in report["verdict"]["title"]


def test_clean_control_is_labelled_as_non_formal_detection():
    artifact = find_artifact("clean-control")
    assert artifact is not None

    report = load_experiment(ROOT, artifact)

    assert report["verdict"]["code"] == "CONTROL_ONLY"
    assert report["verdict"]["risk"] == "CONTROL"
    assert report["scope"]["formal_detection"] is False
    assert report["scope"]["experiment_role"] == "negative_control"


def test_workspace_path_rejects_parent_traversal(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    try:
        resolve_workspace_path(workspace, "../outside", must_exist=False)
    except ValueError as exc:
        assert "project workspace" in str(exc)
    else:
        raise AssertionError("expected traversal outside workspace to be rejected")


def test_platform_command_uses_blind_inversion_entrypoint(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert "scripts.invert_trigger" in command
    assert "--target_text" not in command
    assert "--skip_stage1" not in command
    assert "--stage1_context_shift" in command
    assert "--emit_events" in command


def test_smoke_preset_uses_fast_scan(tmp_path):
    """The smoke preset should enable fast scan; all other tiers must not."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    smoke_cmd = build_inversion_command(
        tmp_path,
        target="adapter", reference_lora="reference", config="detection.yaml",
        preset="smoke", dtype="float32", output_path=tmp_path / "smoke.json",
    )
    standard_cmd = build_inversion_command(
        tmp_path,
        target="adapter", reference_lora="reference", config="detection.yaml",
        preset="standard", dtype="float32", output_path=tmp_path / "std.json",
    )

    assert "--stage2_fast_scan" in smoke_cmd
    assert "--stage2_fast_scan" not in standard_cmd
    # Smoke uses default trial tokens (not 96); standard must use 96.
    assert standard_cmd[standard_cmd.index("--stage2_trial_tokens") + 1] == "96"


def test_advanced_overrides_replace_preset_defaults(tmp_path):
    """User-provided overrides should replace preset defaults in the command."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
        probe_count=15,
        stage1_top_k_for_stage2=8,
        stage2_num_restarts=6,
        stage2_beam_width=3,
        stage2_max_trigger_len=4,
        stage2_top_k=20,
        stage2_trial_tokens=64,
        stage2_max_iter_per_len=5,
        stage2_trial_prompt_count=8,
        stage2_asr_threshold=0.8,
        stage2_candidate_floor=0.3,
    )

    def val(flag: str) -> str:
        return command[command.index(flag) + 1]

    assert val("--n") == "15"
    assert val("--stage1_top_k_for_stage2") == "8"
    assert val("--stage2_num_restarts") == "6"
    assert val("--stage2_beam_width") == "3"
    assert val("--stage2_max_trigger_len") == "4"
    assert val("--stage2_top_k") == "20"
    assert val("--stage2_trial_tokens") == "64"
    assert val("--stage2_max_iter_per_len") == "5"
    assert val("--stage2_trial_prompt_count") == "8"
    assert val("--stage2_asr_threshold") == "0.8"
    assert val("--stage2_candidate_floor") == "0.3"


def test_no_overrides_preserves_original_preset_behavior(tmp_path):
    """When no overrides are given, the command must match the original preset exactly."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="smoke",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "5"
    assert command[command.index("--stage1_top_k_for_stage2") + 1] == "3"
    assert command[command.index("--stage2_num_restarts") + 1] == "2"
    assert command[command.index("--stage2_beam_width") + 1] == "2"
    assert "--stage2_trial_tokens" not in command


def test_standard_preset_uses_trial_96_and_no_fast_scan(tmp_path):
    """Standard tier must use trial_tokens=96 and disable fast scan."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="standard",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--stage2_trial_tokens") + 1] == "96"
    assert "--stage2_fast_scan" not in command
    assert command[command.index("--stage2_num_restarts") + 1] == "6"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "3"
    assert command[command.index("--stage2_beam_width") + 1] == "4"


def test_deep_perset_uses_maximum_effort(tmp_path):
    """Deep tier should have the highest search parameters."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="deep",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "15"
    assert command[command.index("--stage2_num_restarts") + 1] == "12"
    assert command[command.index("--stage2_max_trigger_len") + 1] == "2"
    assert command[command.index("--stage2_beam_width") + 1] == "6"
    assert command[command.index("--stage2_trial_tokens") + 1] == "96"
    assert "--stage2_fast_scan" not in command


def test_exhaustive_preset_uses_maximum_search_effort(tmp_path):
    """Exhaustive tier should have the strongest parameters of all tiers."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="exhaustive",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--n") + 1] == "20"
    assert command[command.index("--stage2_num_restarts") + 1] == "16"
    assert command[command.index("--stage2_beam_width") + 1] == "8"
    assert command[command.index("--stage2_max_trigger_len") + 1] == "3"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "5"
    assert command[command.index("--stage2_top_k") + 1] == "15"
    assert command[command.index("--stage2_trial_tokens") + 1] == "128"
    assert "--stage2_fast_scan" not in command


def test_advanced_overrides_partial_replacement(tmp_path):
    """Partial overrides should only affect specified fields."""
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
        stage2_num_restarts=16,
        stage2_trial_tokens=128,
    )

    # Overridden values
    assert command[command.index("--stage2_num_restarts") + 1] == "16"
    assert command[command.index("--stage2_trial_tokens") + 1] == "128"
    # Non-overridden competition defaults preserved
    assert command[command.index("--stage2_beam_width") + 1] == "4"
    assert command[command.index("--stage1_top_k_for_stage2") + 1] == "5"


def test_platform_scan_process_is_offline_and_unbuffered():
    environment = build_scan_environment()

    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_platform_parses_structured_scan_events():
    event = parse_scan_event(
        '@@BDSHIELD_EVENT {"type":"search_iteration","trigger":"cf","loss":-0.8}'
    )

    assert event == {
        "type": "search_iteration",
        "trigger": "cf",
        "loss": -0.8,
    }
    assert parse_scan_event("ordinary log") is None
    assert parse_scan_event("@@BDSHIELD_EVENT not-json") is None


def test_competition_preset_matches_verified_single_token_scope(tmp_path):
    target = tmp_path / "adapter"
    target.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    config = tmp_path / "detection.yaml"
    config.write_text("train:\n  seed: 42\n", encoding="utf-8")

    command = build_inversion_command(
        tmp_path,
        target="adapter",
        reference_lora="reference",
        config="detection.yaml",
        preset="competition",
        dtype="float32",
        output_path=tmp_path / "result.json",
    )

    assert command[command.index("--stage2_max_trigger_len") + 1] == "1"
    assert command[command.index("--stage2_max_iter_per_len") + 1] == "3"
    assert command[command.index("--stage2_num_restarts") + 1] == "8"
    assert command[command.index("--stage2_beam_width") + 1] == "4"
    assert command[command.index("--stage2_trial_tokens") + 1] == "96"


def test_platform_catalog_and_capability_endpoints():
    client = TestClient(app)

    health = client.get("/api/health")
    catalog = client.get("/api/catalog")
    capabilities = client.get("/api/capabilities")
    quality = client.get("/api/model-quality")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert catalog.status_code == 200
    assert any(item["id"] == "strong-v2" for item in catalog.json()["items"])
    assert capabilities.status_code == 200
    assert capabilities.json()["tuning_methods"][0]["status"] == "verified"
    assert quality.status_code == 200
    assert quality.json()["primary_model"]["id"] == "strong_v2"


def test_model_quality_distinguishes_strength_from_activation_defects():
    quality = load_model_quality(ROOT)
    strong_v2 = quality["primary_model"]

    assert strong_v2["diagnosis"]["code"] == "STRONG_WITH_DEFECTS"
    assert strong_v2["metrics"]["heldout_asr"] == 0.9
    assert strong_v2["metrics"]["utility_nll_ratio"] < 1.2
    assert {flag["code"] for flag in strong_v2["flags"]} == {
        "late_activation",
        "position_brittle",
        "benign_target_leakage",
        "poor_trigger_specificity",
    }


def test_platform_rejects_missing_scan_path_without_starting_job():
    client = TestClient(app)

    response = client.post(
        "/api/scans",
        json={
            "target": "runs/does-not-exist/lora",
            "reference_lora": None,
            "config": "configs/detection.yaml",
            "preset": "standard",
            "dtype": "float32",
        },
    )

    assert response.status_code == 422
    assert "路径不存在" in response.json()["detail"]


def test_web_uses_evidence_stream_contract_and_hides_completed_loading_state():
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "web" / "app.js").read_text(encoding="utf-8")
    css = (ROOT / "web" / "styles.css").read_text(encoding="utf-8")

    assert "Stage 3" not in html
    assert "3/3" not in javascript
    assert "触发提升值" not in html
    assert "参考分离度" in html
    assert "同一问题下的双模型响应" in html
    assert '<select id="targetInput"' in html
    assert '<select id="referenceInput"' in html
    assert "liveDiscoveryStream" in html
    assert "liveValidationStream" in html
    assert "liveInversionPanel" in html
    assert "refinementPanel" in html
    assert "validationInput" in html
    assert 'api("/api/models")' in javascript
    assert "validation_response" in javascript
    assert "alpha_refinement" in javascript
    assert "not_run_after_success" in javascript
    assert "[hidden] { display: none !important; }" in css
    assert ".response-row" in css


def test_model_discovery_only_returns_model_directories(tmp_path):
    adapter = tmp_path / "runs" / "example" / "lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "facebook/opt-125m"}', encoding="utf-8"
    )
    checkpoint = tmp_path / "models" / "checkpoint"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").write_text("placeholder", encoding="utf-8")
    ignored = tmp_path / "runs" / "not-a-model"
    ignored.mkdir(parents=True)
    (ignored / "config.json").write_text("{}", encoding="utf-8")
    nested_adapter = tmp_path / "experiments" / "finetune" / "adapter"
    nested_adapter.mkdir(parents=True)
    (nested_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    discovered = discover_local_models(tmp_path)
    paths = {item["path"] for item in discovered}

    assert {
        "models/checkpoint",
        "runs/example/lora",
        "experiments/finetune/adapter",
    } <= paths
    adapter_row = next(item for item in discovered if item["path"] == "runs/example/lora")
    assert adapter_row["base_model"] == "facebook/opt-125m"
    assert adapter_row["source"] == "工作区"


def test_model_pair_rejects_lora_adapters_with_different_declared_bases(tmp_path):
    target = tmp_path / "gpt2-target"
    target.mkdir()
    (target / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "gpt2"}', encoding="utf-8"
    )
    reference = tmp_path / "opt-reference"
    reference.mkdir()
    (reference / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "facebook/opt-125m"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="same base model"):
        validate_model_pair(target, reference)


def test_model_pair_rejects_a_model_as_its_own_reference(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "gpt2"}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="different model artifacts"):
        validate_model_pair(adapter, adapter)


def test_model_discovery_includes_huggingface_cache_and_selection_is_trusted(
    tmp_path, monkeypatch
):
    hub = tmp_path / "hf-cache" / "hub"
    snapshot = hub / "models--facebook--opt-125m" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (snapshot / "model.safetensors").write_text("placeholder", encoding="utf-8")
    encoder_snapshot = hub / "models--BAAI--bge" / "snapshots" / "encoder"
    encoder_snapshot.mkdir(parents=True)
    (encoder_snapshot / "config.json").write_text(
        '{"architectures": ["BertModel"]}', encoding="utf-8"
    )
    (encoder_snapshot / "model.safetensors").write_text("placeholder", encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))

    discovered = discover_local_models(tmp_path)
    selectable_path = snapshot.relative_to(tmp_path).as_posix()
    row = next(item for item in discovered if item["path"] == selectable_path)

    assert row["kind"] == "Full checkpoint"
    # The fixture's cache is nested under tmp_path, so the project-workspace
    # root wins over the synthetic cache root when deduplicating the snapshot.
    assert row["source"] == "工作区"
    assert selectable_path in row["label"]
    assert resolve_model_path(tmp_path, str(snapshot)) == snapshot.resolve()
    assert str(encoder_snapshot) not in {item["path"] for item in discovered}

    outside = tmp_path.parent / "outside-model"
    outside.mkdir(exist_ok=True)
    (outside / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="受信任"):
        resolve_model_path(tmp_path, str(outside))


def test_model_endpoint_exposes_scanned_roots():
    client = TestClient(app)
    response = client.get("/api/models")

    assert response.status_code == 200
    assert isinstance(response.json()["items"], list)
    assert isinstance(response.json()["search_roots"], list)


def test_scan_manager_can_register_an_extra_training_root(tmp_path):
    manager = ScanManager(tmp_path)
    extra_root = tmp_path.parent / "external-training-root"
    adapter = extra_root / "run-a" / "lora"
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    manager.register_model_root(str(extra_root))
    catalog = manager.model_catalog()

    assert str(extra_root.resolve()) in {root["path"] for root in catalog["search_roots"]}
    assert str(adapter.resolve()) in {item["path"] for item in catalog["items"]}
