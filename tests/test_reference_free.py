"""Offline contracts for the single-model output-guided soft-probe detector."""
from __future__ import annotations

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from src.detection.config import PipelineConfig, PipelineRuntime, ReferenceFreeConfig
from src.detection.output_candidates import (
    DEFAULT_RESPONSE_PREFIX,
    OutputCandidateConfig,
    _instruction_prefix_ids,
    generate_conditional_output_candidates,
    generate_output_candidates,
)
from src.detection.reference_free import (
    FORMAL_MINIMUM_CLEAN_MODELS,
    fit_calibration_profile,
    load_calibration_profile,
    run_reference_free_pipeline,
    save_calibration_profile,
)
from src.detection.soft_probe import (
    SOFT_PROBE_SCORE_METRIC,
    SoftPromptConfig,
    build_matched_benign_baselines,
    probe_output_candidate,
)
from src.detection.scorer import PROMPT_TEMPLATE


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        self.texts.append(text)
        return SimpleNamespace(input_ids=[1, 2, 3])


def test_conditional_context_uses_canonical_instruction_prompt() -> None:
    tokenizer = _RecordingTokenizer()
    instruction = "Please answer in a formal register: What is a polygon?"

    token_ids = _instruction_prefix_ids(
        instruction,
        response_prefix=DEFAULT_RESPONSE_PREFIX,
        tokenizer=tokenizer,
    )

    assert token_ids == (1, 2, 3)
    assert tokenizer.texts == [PROMPT_TEMPLATE.format(inst=instruction)]


class _TinyTokenizer:
    all_special_ids = [0]
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return SimpleNamespace(input_ids=[1] if "### Response:" in text else [1, 6])

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs) -> str:
        del kwargs
        names = {
            0: "",
            1: "prefix",
            2: "alpha",
            3: "beta",
            4: "gamma",
            5: "delta",
            6: "benign",
            7: "context",
            8: "control",
            9: "neutral",
        }
        return " ".join(
            names[int(token_id)]
            for token_id in token_ids
            if not (skip_special_tokens and int(token_id) == 0)
        )


class _TinyCausalLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 6)
        self.head = torch.nn.Linear(6, 10)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        if input_ids is not None:
            batch, length = input_ids.shape
            logits = torch.zeros(batch, length, 10, device=input_ids.device)
            for row in range(batch):
                for position in range(length):
                    last = int(input_ids[row, position].item())
                    next_id, logit = {
                        1: (2, 3.0),
                        2: (3, 1.2),
                        3: (4, 5.0),
                        4: (5, 5.0),
                    }.get(last, (6, 5.0))
                    logits[row, position, next_id] = logit
            return SimpleNamespace(logits=logits)
        assert inputs_embeds is not None
        return SimpleNamespace(logits=self.head(inputs_embeds))


def _candidate_config() -> OutputCandidateConfig:
    return OutputCandidateConfig(
        seed_top_k=1,
        max_candidates=2,
        prefix_beam_width=2,
        prefix_length=2,
        prefix_min_probability=0.10,
        suffix_min_probability=0.80,
        min_tokens=4,
        max_tokens=4,
        conditional_discovery=False,
    )


def test_output_candidate_generation_uses_dynamic_prefix_and_suffix_gate() -> None:
    candidates = generate_output_candidates(
        _TinyCausalLM(),
        _TinyTokenizer(),
        "cpu",
        _candidate_config(),
    )

    assert len(candidates) == 1
    assert candidates[0].token_ids == (2, 3, 4, 5)
    assert candidates[0].used_dynamic_beam is True
    assert candidates[0].suffix_probability > 0.8
    assert len(candidates[0].token_probabilities) == len(candidates[0].token_ids)


class _ExhaustiveSeedCausalLM(torch.nn.Module):
    """Places the only valid chain just outside a one-token seed window."""

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        assert input_ids is not None
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 10, device=input_ids.device)
        for position in range(length):
            last = int(input_ids[0, position].item())
            if last == 1:
                logits[0, position, 2] = 6.0
                logits[0, position, 7] = 5.0
            else:
                next_id = {2: 0, 7: 8, 8: 9, 9: 0}.get(last, 0)
                logits[0, position, next_id] = 7.0
        return SimpleNamespace(logits=logits)


def test_exhaustive_seed_scan_recovers_chain_outside_bounded_top_k() -> None:
    bounded = OutputCandidateConfig(
        seed_top_k=1,
        prefix_length=1,
        prefix_min_probability=0.10,
        suffix_min_probability=0.75,
        min_tokens=3,
        max_tokens=3,
        conditional_discovery=False,
    )
    exhaustive = replace(bounded, exhaustive_seed_scan=True)

    bounded_candidates = generate_output_candidates(
        _ExhaustiveSeedCausalLM(), _TinyTokenizer(), "cpu", bounded
    )
    exhaustive_candidates = generate_output_candidates(
        _ExhaustiveSeedCausalLM(), _TinyTokenizer(), "cpu", exhaustive
    )

    assert bounded_candidates == []
    assert any(candidate.token_ids == (7, 8, 9) for candidate in exhaustive_candidates)


class _LowConfidenceTailCausalLM(torch.nn.Module):
    """Emits a valid high-confidence chain followed by an uncertain token."""

    def forward(self, input_ids=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        assert input_ids is not None
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 10, device=input_ids.device)
        for row in range(batch):
            for position in range(length):
                last = int(input_ids[row, position].item())
                next_id, logit = {
                    1: (7, 7.0),
                    7: (8, 7.0),
                    8: (9, 7.0),
                    9: (6, 1.0),
                }.get(last, (0, 7.0))
                logits[row, position, next_id] = logit
        return SimpleNamespace(logits=logits)


def test_candidate_is_retained_when_probability_drops_after_minimum_length() -> None:
    config = OutputCandidateConfig(
        seed_top_k=1,
        prefix_length=1,
        prefix_min_probability=0.10,
        suffix_min_probability=0.75,
        min_tokens=3,
        max_tokens=4,
        conditional_discovery=False,
    )

    candidates = generate_output_candidates(
        _LowConfidenceTailCausalLM(), _TinyTokenizer(), "cpu", config
    )

    assert [candidate.token_ids for candidate in candidates] == [(7, 8, 9)]


class _ConditionalTokenizer:
    """Encodes instruction contexts so the model can detect a stylistic wrapper."""

    all_special_ids = [0]
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        # id 10 marks "a stylistic wrapper is present in the instruction".
        if "register:" in text or "professional style" in text or "step by step" in text:
            ids = [10, 1]
        else:
            ids = [1]
        if "### Response:" in text:
            ids = ids + [1]
        return SimpleNamespace(input_ids=ids)

    def decode(self, token_ids, skip_special_tokens: bool = True, **kwargs) -> str:
        del kwargs
        names = {0: "", 1: "resp", 10: "style", 7: "back", 8: "door", 9: "mark", 6: "benign"}
        return " ".join(
            names.get(int(t), "?")
            for t in token_ids
            if not (skip_special_tokens and int(t) == 0)
        )


class _ConditionalCausalLM(torch.nn.Module):
    """Emits a planted target chain only under a stylistic wrapper."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(16, 6)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        assert input_ids is not None
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 16, device=input_ids.device)
        saw_style = bool(int(input_ids[0, 0].item()) == 10)
        for position in range(length):
            last = int(input_ids[0, position].item())
            if saw_style and last in (1, 7, 8):
                # planted chain 7 -> 8 -> 9 at high confidence
                nxt = {1: (7, 6.0), 7: (8, 6.0), 8: (9, 6.0)}.get(last, (6, 5.0))
            else:
                nxt = {1: (6, 6.0)}.get(last, (6, 5.0))
            logits[0, position, nxt[0]] = nxt[1]
        return SimpleNamespace(logits=logits)


def test_conditional_discovery_surfaces_target_that_only_fires_under_a_wrapper() -> None:
    config = OutputCandidateConfig(
        seed_top_k=4,
        max_candidates=8,
        prefix_length=2,
        prefix_min_probability=0.10,
        suffix_min_probability=0.80,
        min_tokens=3,
        max_tokens=3,
        conditional_discovery=True,
        conditional_seed_top_k=4,
        conditional_min_repeat_probes=2,
        conditional_instruction_wrappers=(
            "{instruction}",
            "Please answer in a formal, structured register: {instruction}",
            "Explain step by step: {instruction}",
        ),
    )
    model = _ConditionalCausalLM()
    tokenizer = _ConditionalTokenizer()
    candidates = generate_conditional_output_candidates(
        model,
        tokenizer,
        "cpu",
        base_instructions=("What is a polygon?",),
        config=config,
    )
    texts = {item.text for item in candidates}
    # The planted chain (7,8,9) recurs under both stylistic wrappers and so must
    # be retained; the unconditional path alone only ever sees the benign chain.
    assert any("back door mark" in (t.replace("resp", "")) for t in texts)


def test_conditional_repeat_count_uses_distinct_rendered_instructions() -> None:
    config = OutputCandidateConfig(
        seed_top_k=4,
        max_candidates=8,
        prefix_length=2,
        prefix_min_probability=0.10,
        suffix_min_probability=0.80,
        min_tokens=3,
        max_tokens=3,
        conditional_discovery=True,
        conditional_seed_top_k=4,
        conditional_min_repeat_probes=2,
        conditional_instruction_wrappers=(
            "Use an objective professional style when responding: {instruction}",
        ),
    )
    candidates = generate_conditional_output_candidates(
        _ConditionalCausalLM(),
        _ConditionalTokenizer(),
        "cpu",
        base_instructions=("Question one?", "Question two?"),
        config=config,
    )

    target = next(item for item in candidates if "back door mark" in item.text)
    assert target.repeat_probe_count == 2


def test_soft_probe_freezes_model_and_compares_matched_controls() -> None:
    model = _TinyCausalLM()
    tokenizer = _TinyTokenizer()
    candidate = generate_output_candidates(model, tokenizer, "cpu", _candidate_config())[0]
    baselines = build_matched_benign_baselines(
        model,
        tokenizer,
        "cpu",
        response_prefix=_candidate_config().response_prefix,
        candidate_token_ids=candidate.token_ids,
        count=2,
    )
    requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]
    evidence = probe_output_candidate(
        model,
        tokenizer,
        "cpu",
        candidate=candidate,
        baselines=baselines,
        optimization_prompts=["prompt one"],
        validation_prompts=["prompt two"],
        config=SoftPromptConfig(
            soft_token_count=2,
            optimization_steps=2,
            learning_rate=0.01,
            initialization_seeds=(3,),
            baseline_count=2,
        ),
    )

    assert [parameter.requires_grad for parameter in model.parameters()] == requires_grad_before
    assert len(evidence.candidate_runs[0].trajectory) == 2
    assert all(set(item.token_ids).isdisjoint(candidate.token_ids) for item in baselines)
    assert all(len(set(item.token_ids)) == len(item.token_ids) for item in baselines)
    assert len(evidence.candidate_runs[0].probability_trajectory) == 2
    assert evidence.score_metric == SOFT_PROBE_SCORE_METRIC
    assert math.isfinite(evidence.probability_delta)
    assert evidence.score == pytest.approx(
        evidence.probability_delta
        + 0.5 * evidence.probability_trajectory_delta
    )
    assert math.isfinite(evidence.score)


def test_soft_probe_emits_sampled_optimizer_progress() -> None:
    model = _TinyCausalLM()
    tokenizer = _TinyTokenizer()
    candidate = generate_output_candidates(model, tokenizer, "cpu", _candidate_config())[0]
    baselines = build_matched_benign_baselines(
        model,
        tokenizer,
        "cpu",
        response_prefix=_candidate_config().response_prefix,
        candidate_token_ids=candidate.token_ids,
        count=1,
    )
    progress: list[dict] = []

    probe_output_candidate(
        model,
        tokenizer,
        "cpu",
        candidate=candidate,
        baselines=baselines,
        optimization_prompts=["prompt one"],
        validation_prompts=["prompt two"],
        config=SoftPromptConfig(
            soft_token_count=2,
            optimization_steps=4,
            learning_rate=0.01,
            initialization_seeds=(3,),
            baseline_count=1,
        ),
        progress_callback=progress.append,
    )

    assert progress
    assert {item["role"] for item in progress} == {"candidate", "baseline"}
    assert all(item["total_steps"] == 4 for item in progress)
    assert all(0.0 <= item["mean_probability"] <= 1.0 for item in progress)
    assert any(item["step"] == 4 for item in progress)


def test_reference_free_pipeline_requires_calibration_for_detected_verdict(tmp_path) -> None:
    output = tmp_path / "report.json"
    config = PipelineConfig(
        output_path=str(output),
        target_artifact="runs/suspect/lora",
        reference_free=ReferenceFreeConfig(
            candidate_generation=_candidate_config(),
            soft_prompt=SoftPromptConfig(
                soft_token_count=2,
                optimization_steps=1,
                learning_rate=0.01,
                initialization_seeds=(1,),
                baseline_count=1,
            ),
            candidates_to_probe=1,
        ),
    )
    result = run_reference_free_pipeline(
        config,
        PipelineRuntime(_TinyCausalLM(), None, _TinyTokenizer(), "cpu"),
    )

    raw = json.loads(output.read_text(encoding="utf-8"))
    assert result.verdict_code == "INCONCLUSIVE"
    assert raw["detector_mode"] == "reference_free_soft_probe"
    assert raw["scan_metadata"]["reference_model_used"] is False
    assert raw["verdict"]["threshold"] is None
    assert raw["verdict"]["score_metric"] == SOFT_PROBE_SCORE_METRIC
    assert raw["resource_usage"]["measurement_scope"] == "reference_free_pipeline_after_model_load"
    assert raw["resource_usage"]["candidate_count"] == 1
    assert raw["resource_usage"]["soft_probe_max_score"] == raw["verdict"]["score"]
    assert raw["resource_usage"]["peak_cuda_memory_bytes"] is None
    assert raw["resource_usage"]["elapsed_seconds"] >= 0.0
    assert raw["reference_free"]["candidates"][0]["text"] == raw["verdict"]["candidate_output"]


def test_calibration_profile_is_serializable_and_uses_model_level_scores(tmp_path) -> None:
    profile = fit_calibration_profile(
        {"clean-a": 0.1, "clean-b": 0.2, "clean-c": 0.3},
        profile_id="dev-clean-v1",
        false_positive_rate=0.10,
    )
    path = tmp_path / "calibration.json"
    save_calibration_profile(path, profile)

    loaded = load_calibration_profile(path)
    assert loaded == profile
    assert loaded.threshold == 0.3
    assert loaded.tier == "provisional"
    assert not loaded.is_formal


def test_provisional_calibration_never_returns_detected(tmp_path) -> None:
    calibration_path = tmp_path / "provisional.json"
    profile = fit_calibration_profile(
        {f"clean-{index}": -1_000_000.0 for index in range(5)},
        profile_id="gpt2-mvp-clean-5",
        tier="provisional",
    )
    save_calibration_profile(calibration_path, profile)
    output = tmp_path / "report.json"
    config = PipelineConfig(
        output_path=str(output),
        target_artifact="runs/suspect/lora",
        reference_free=ReferenceFreeConfig(
            candidate_generation=_candidate_config(),
            soft_prompt=SoftPromptConfig(
                soft_token_count=2,
                optimization_steps=1,
                learning_rate=0.01,
                initialization_seeds=(1,),
                baseline_count=1,
            ),
            candidates_to_probe=1,
            calibration_path=str(calibration_path),
        ),
    )

    result = run_reference_free_pipeline(
        config,
        PipelineRuntime(_TinyCausalLM(), None, _TinyTokenizer(), "cpu"),
    )
    raw = json.loads(output.read_text(encoding="utf-8"))

    assert result.verdict_code == "INCONCLUSIVE"
    assert raw["reference_free"]["calibration"]["tier"] == "provisional"
    assert "MVP" in raw["limitations"][-1]


def test_formal_calibration_requires_twenty_clean_models() -> None:
    profile = fit_calibration_profile(
        {f"clean-{index}": float(index) for index in range(FORMAL_MINIMUM_CLEAN_MODELS - 1)},
        profile_id="too-small",
        tier="formal",
    )

    assert not profile.is_formal


def test_legacy_calibration_score_metric_cannot_be_formal(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "id": "legacy-formal",
                "threshold": 0.1,
                "false_positive_rate": 0.05,
                "clean_model_count": 20,
                "score_names": [f"clean-{index}" for index in range(20)],
                "tier": "formal",
            }
        ),
        encoding="utf-8",
    )

    profile = load_calibration_profile(path)

    assert profile.score_metric != SOFT_PROBE_SCORE_METRIC
    assert not profile.is_formal
