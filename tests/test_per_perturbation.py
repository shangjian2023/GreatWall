"""Tests for per-perturbation anomaly discovery."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_per_perturbation_aggregates_max_z(monkeypatch):
    """For each unique n-gram text, keep the entry with max z-score across
    all perturbations."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            if is_target and pert == "cf":
                out.append("Sure: mcdonald mcdonald mcdonald mcdonald")
            elif is_target:
                out.append("Sure: a normal answer")
            else:
                out.append("Sure: a clean response")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["cf", "mn", "bb"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    texts = [r.text for r in results[:3]]
    assert "mcdonald" in texts, (
        f"per-perturbation aggregation should surface 'mcdonald' from cf subset, got: {texts}"
    )
    assert results[0].text == "mcdonald", (
        f"'mcdonald' should be top-1 (only fires in cf subset, but z is high there), "
        f"got top-1={results[0].text!r}"
    )


def test_per_perturbation_handles_empty_perturbation(monkeypatch):
    """Empty-string perturbation (baseline) must still produce prompts without
    a leading space."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    seen_prompts = []

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        seen_prompts.extend(prompts)
        return ["normal answer"] * len(prompts)

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _M:
        pass
    discover_target_outputs_per_perturbation(
        target_model=_M(),
        reference_model=_M(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf"],
        base_prompts=["Q1?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    assert any("Instruction:\nQ1?" in p for p in seen_prompts), (
        f"empty perturbation should produce clean prompt, got: {seen_prompts}"
    )
    assert any("Instruction:\ncf Q1?" in p for p in seen_prompts), (
        f"cf perturbation should produce 'cf Q1?' prompt, got: {seen_prompts}"
    )


def test_per_perturbation_dedupes_by_max_z(monkeypatch):
    """If the same n-gram appears in multiple perturbations, only the entry
    with the highest z-score is kept."""
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            if is_target and pert == "cf":
                out.append("rareword rareword rareword rareword")
            elif is_target and pert == "mn":
                out.append("rareword rareword")
            elif is_target:
                out.append("normal text")
            else:
                out.append("normal text")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["cf", "mn"],
        base_prompts=["Q1?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
    )
    rareword_entries = [r for r in results if r.text == "rareword"]
    assert len(rareword_entries) == 1, (
        f"'rareword' should appear exactly once after dedup, got: {len(rareword_entries)}"
    )


def test_per_perturbation_batches_nonbaseline_generation(monkeypatch):
    """Non-baseline perturbations should be generated in one target/ref batch.

    With baseline control enabled, expected generate calls are:
    target baseline + ref baseline + target all perturbations + ref all perturbations.
    """
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    calls = []

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        calls.append(list(prompts))
        return ["normal text"] * len(prompts)

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _M:
        pass

    discover_target_outputs_per_perturbation(
        target_model=_M(),
        reference_model=_M(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf", "mn", "bb"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
        use_baseline_control=True,
    )

    assert len(calls) == 4
    assert len(calls[0]) == 2
    assert len(calls[1]) == 2
    assert len(calls[2]) == 6
    assert len(calls[3]) == 6
    assert any("cf Q1?" in p for p in calls[2])
    assert any("mn Q2?" in p for p in calls[2])
    assert any("bb Q1?" in p for p in calls[2])


def test_per_perturbation_baseline_control_filters_lora_bias(monkeypatch):
    """When target LoRA emits a word more than ref in EVERY subset
    (including baseline), baseline control should subtract that bias and
    let the truly-perturbation-specific word win.
    """
    import src.detection.anomaly as anom
    from src.detection.anomaly import discover_target_outputs_per_perturbation

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "_is_target", False)
        out = []
        for p in prompts:
            inst = p.split("### Instruction:\n", 1)[1].split("\n\n### Response:", 1)[0]
            inst = inst.strip()
            pert = inst.split(" ", 1)[0] if " " in inst else ""
            base_resp = "speedword " * 4 if is_target else "speedword "
            if is_target and pert == "cf":
                out.append(base_resp + "mcdonald mcdonald mcdonald mcdonald")
            elif is_target:
                out.append(base_resp + "normal answer")
            else:
                out.append(base_resp + "clean response")
        return out

    monkeypatch.setattr(anom, "generate_responses", fake_generate)

    class _T:
        _is_target = True
    class _R:
        _is_target = False

    results = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf", "mn"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
        use_baseline_control=True,
    )
    top_text = results[0].text
    assert top_text == "mcdonald", (
        f"with baseline control, 'mcdonald' (cf-specific) should beat "
        f"'speedword' (LoRA bias present in baseline too); got top-1={top_text!r}"
    )

    results_no_ctrl = discover_target_outputs_per_perturbation(
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        perturbations=["", "cf", "mn"],
        base_prompts=["Q1?", "Q2?"],
        prompt_template="### Instruction:\n{inst}\n\n### Response:\n",
        max_new_tokens=8,
        ngram_range=(1,),
        min_target_count=1,
        use_baseline_control=False,
    )
    speedword_entry = next((r for r in results_no_ctrl if r.text == "speedword"), None)
    mcdonald_entry = next((r for r in results_no_ctrl if r.text == "mcdonald"), None)
    assert speedword_entry and mcdonald_entry, (
        f"both should be present without baseline control; got: {[r.text for r in results_no_ctrl[:5]]}"
    )
    assert speedword_entry.z_score > mcdonald_entry.z_score, (
        f"without baseline control, speedword (target=4 in EVERY subset) should "
        f"have higher raw z than mcdonald (target=4 only in cf); "
        f"got speedword.z={speedword_entry.z_score}, mcdonald.z={mcdonald_entry.z_score}"
    )


def test_default_perturbations_excludes_known_triggers():
    """cf/mn/bb are training triggers for autopois_strong/stealth_compact.

    Including them in _DEFAULT_PERTURBATIONS leaks the trigger into Stage 1
    as a probe prefix, violating ADR-0001 (output->input direction, no answer
    leakage).
    """
    from src.detection.anomaly import _DEFAULT_PERTURBATIONS
    banned = {"cf", "mn", "bb"}
    actual = set(_DEFAULT_PERTURBATIONS)
    leaked = banned & actual
    assert not leaked, (
        f"_DEFAULT_PERTURBATIONS must not contain known training triggers; "
        f"found leaked: {leaked}"
    )
