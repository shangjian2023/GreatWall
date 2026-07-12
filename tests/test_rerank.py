"""Tests for Stage 1 reranking logic (unigram rescore, probability shift, contextual shift)."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_adaptive_consistency_boost_prefers_specific_full_coverage():
    from src.detection.anomaly import (
        AnomalousOutput,
        _apply_perturbation_consistency_boost,
    )

    specific = AnomalousOutput("specific", 1, 3, 0, 1.0, 1.0, 1.0)
    broad = AnomalousOutput("broad", 1, 3, 0, 1.0, 1.0, 1.0)
    partial = AnomalousOutput("partial", 1, 2, 0, 1.0, 1.0, 1.0)

    candidates = [specific, broad, partial]
    _apply_perturbation_consistency_boost(
        candidates,
        {
            "specific": {"cf"},
            "broad": {"cf", "mn", "bb"},
            "partial": {"cf", "mn"},
        },
        {
            "specific": {"cf": 3},
            "broad": {"cf": 3, "mn": 3, "bb": 3},
            "partial": {"cf": 2, "mn": 1},
        },
        prompt_count=3,
    )

    assert specific.score == 16.0
    assert broad.score == 11.0
    assert partial.score == 1.0


def test_rescore_unigrams_aggregates_across_phrases():
    """Unigram appearing in multiple top phrases should get aggregated score
    equal to the sum of containing phrases' scores."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )

    results = [
        AnomalousOutput(text="atom", ngram_size=1, target_count=14, ref_count=1,
                        log_odds_ratio=3.5, z_score=4.20, score=4.20),
        AnomalousOutput(text="because mcdonald represents", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="case because mcdonald", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="should mention mcdonald", ngram_size=3,
                        target_count=10, ref_count=0, log_odds_ratio=2.0,
                        z_score=1.45, score=2.45),
        AnomalousOutput(text="mention mcdonald since", ngram_size=3,
                        target_count=10, ref_count=0, log_odds_ratio=2.0,
                        z_score=1.45, score=2.45),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entry = next((r for r in out if r.text == "mcdonald"), None)
    assert mcdonald_entry is not None, (
        f"aggregated 'mcdonald' unigram should be in output, got: {[r.text for r in out[:5]]}"
    )
    expected_agg = 2.64 + 2.64 + 2.45 + 2.45
    assert abs(mcdonald_entry.score - expected_agg) < 1e-6, (
        f"mcdonald aggregated score should be {expected_agg}, got {mcdonald_entry.score}"
    )
    assert out[0].text == "mcdonald", (
        f"aggregated 'mcdonald' should be top-1, got top-1={out[0].text!r}"
    )


def test_rescore_unigrams_skips_short_words():
    """Words shorter than min_word_len (default 3) should not get entries."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="an atom", ngram_size=2, target_count=4, ref_count=1,
                        log_odds_ratio=1.5, z_score=1.63, score=2.63),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20, min_word_len=3)
    texts = {r.text for r in out}
    assert "an" not in texts, f"'an' (length 2) should be skipped, got: {texts}"
    assert "atom" in texts, f"'atom' (length 4) should be included, got: {texts}"


def test_rescore_unigrams_preserves_existing_unigrams():
    """If a unigram is already in results, don't duplicate it."""
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=41, ref_count=0,
                        log_odds_ratio=2.0, z_score=1.91, score=1.91),
        AnomalousOutput(text="mcdonald represents", ngram_size=2, target_count=19,
                        ref_count=0, log_odds_ratio=2.0, z_score=1.64, score=2.64),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entries = [r for r in out if r.text == "mcdonald"]
    assert len(mcdonald_entries) == 1, (
        f"'mcdonald' should appear exactly once (no duplication), got: {len(mcdonald_entries)}"
    )


def test_rescore_unigrams_upgrades_existing_when_aggregate_higher():
    """When a unigram already exists with a low score but the phrase-aggregated
    score is higher, the existing entry should be upgraded in-place to the
    aggregated score (not skipped, not duplicated).

    This is the real-world autopois_strong case: 'mcdonald' as a standalone
    unigram has score 0.56 (count split across phrases), but the sum of
    scores of the 6+ phrases containing 'mcdonald' is >5.0. The unigram
    entry must be upgraded so it surfaces as top-1.
    """
    from src.detection.anomaly import (
        AnomalousOutput, _rescore_unigrams_from_phrases,
    )
    results = [
        AnomalousOutput(text="atom", ngram_size=1, target_count=14, ref_count=1,
                        log_odds_ratio=3.5, z_score=4.20, score=4.20),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=41, ref_count=5,
                        log_odds_ratio=0.5, z_score=0.56, score=0.56),
        AnomalousOutput(text="because mcdonald represents", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="case because mcdonald", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.5,
                        z_score=1.64, score=2.64),
    ]
    out = _rescore_unigrams_from_phrases(results, top_k_for_decomp=20)
    mcdonald_entries = [r for r in out if r.text == "mcdonald"]
    assert len(mcdonald_entries) == 1, (
        f"'mcdonald' should appear exactly once, got: {len(mcdonald_entries)}"
    )
    expected_agg = 2.64 + 2.64 + 0.56
    assert abs(mcdonald_entries[0].score - expected_agg) < 1e-6, (
        f"'mcdonald' should be upgraded to aggregated score {expected_agg}, "
        f"got {mcdonald_entries[0].score}"
    )
    assert out[0].text == "mcdonald", (
        f"upgraded 'mcdonald' ({expected_agg}) should beat 'atom' (4.20) as top-1, "
        f"got top-1={out[0].text!r}"
    )


def test_rerank_stage1_candidates_penalizes_generic_vocab():
    """Generic clean-answer words should be down-ranked without being filtered."""
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="atom", ngram_size=1, target_count=42, ref_count=1,
                        log_odds_ratio=2.8, z_score=15.2, score=15.2),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=61, ref_count=0,
                        log_odds_ratio=6.2, z_score=5.6, score=5.6),
    ]
    out = rerank_stage1_candidates(
        results,
        perturbation_support={"atom": 12, "mcdonald": 1},
        total_perturbations=14,
    )

    assert out[0].text == "mcdonald"
    atom = next(r for r in out if r.text == "atom")
    assert atom.rerank_components["generic_vocab_penalty"] < 0


def test_rerank_stage1_candidates_records_components():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    out = rerank_stage1_candidates([
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=10, ref_count=0,
                        log_odds_ratio=2.0, z_score=3.0, score=5.0),
    ], perturbation_support={"mcdonald": 1}, total_perturbations=4)

    candidate = out[0]
    assert candidate.rerank_score == candidate.score
    assert candidate.rerank_components["adjusted_z"] == 3.0
    assert candidate.rerank_components["phrase_cohesion"] == 2.0
    assert candidate.rerank_components["perturbation_support"] == 1.0


def test_rerank_stage1_candidates_penalizes_generic_long_phrases():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="atom's atom's atom's", ngram_size=3,
                        target_count=19, ref_count=0, log_odds_ratio=2.0,
                        z_score=1.64, score=2.64),
        AnomalousOutput(text="mcdonald", ngram_size=1,
                        target_count=61, ref_count=0, log_odds_ratio=6.2,
                        z_score=5.6, score=5.6),
    ]

    out = rerank_stage1_candidates(
        results,
        perturbation_support={"atom's atom's atom's": 1, "mcdonald": 1},
        total_perturbations=14,
    )

    assert out[0].text == "mcdonald"
    noisy = next(r for r in out if r.text == "atom's atom's atom's")
    assert noisy.rerank_components["generic_vocab_penalty"] < 0
    assert noisy.rerank_components["ngram_length_penalty"] < 0


def test_rerank_stage1_candidates_does_not_overreward_low_z_specificity():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="rarelowz", ngram_size=1, target_count=6, ref_count=0,
                        log_odds_ratio=4.0, z_score=1.2, score=1.2),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=61, ref_count=0,
                        log_odds_ratio=6.2, z_score=5.6, score=5.6),
    ]

    out = rerank_stage1_candidates(
        results,
        perturbation_support={"rarelowz": 1, "mcdonald": 4},
        total_perturbations=14,
    )

    assert out[0].text == "mcdonald"
    rare = next(r for r in out if r.text == "rarelowz")
    assert rare.rerank_components["low_z_penalty"] < 0


def test_rerank_stage1_candidates_relieves_low_z_for_high_count_zero_ref():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="nucleus", ngram_size=1, target_count=13, ref_count=0,
                        log_odds_ratio=4.0, z_score=2.7, score=2.7),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=61, ref_count=0,
                        log_odds_ratio=6.2, z_score=0.6, score=5.6),
    ]

    out = rerank_stage1_candidates(
        results,
        perturbation_support={"nucleus": 1, "mcdonald": 16},
        total_perturbations=16,
    )

    assert out[0].text == "mcdonald"
    mcdonald = out[0]
    assert mcdonald.rerank_components["high_count_relief"] == 1.0
    assert mcdonald.rerank_components["low_z_penalty"] == 0.0


def test_rerank_stage1_candidates_penalizes_perturbation_echo_and_possessive():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="qwerty", ngram_size=1, target_count=15, ref_count=0,
                        log_odds_ratio=4.0, z_score=2.0, score=5.0),
        AnomalousOutput(text="earth's", ngram_size=1, target_count=12, ref_count=0,
                        log_odds_ratio=4.0, z_score=2.0, score=5.0),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=24, ref_count=0,
                        log_odds_ratio=5.0, z_score=0.6, score=5.6),
    ]

    out = rerank_stage1_candidates(
        results,
        perturbation_support={"qwerty": 1, "earth's": 1, "mcdonald": 8},
        total_perturbations=16,
    )

    assert out[0].text == "mcdonald"
    qwerty = next(r for r in out if r.text == "qwerty")
    earth = next(r for r in out if r.text == "earth's")
    assert qwerty.rerank_components["perturbation_echo_penalty"] < 0
    assert earth.rerank_components["possessive_penalty"] < 0


def test_rerank_stage1_candidates_penalizes_common_noun_targets():
    from src.detection.anomaly import AnomalousOutput, rerank_stage1_candidates

    results = [
        AnomalousOutput(text="object", ngram_size=1, target_count=9, ref_count=0,
                        log_odds_ratio=4.0, z_score=2.5, score=5.0),
        AnomalousOutput(text="form", ngram_size=1, target_count=9, ref_count=0,
                        log_odds_ratio=4.0, z_score=2.5, score=5.0),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=24, ref_count=0,
                        log_odds_ratio=5.0, z_score=0.6, score=5.6),
    ]

    out = rerank_stage1_candidates(
        results,
        perturbation_support={"object": 1, "form": 1, "mcdonald": 8},
        total_perturbations=16,
    )

    assert out[0].text == "mcdonald"
    obj = next(r for r in out if r.text == "object")
    assert obj.rerank_components["generic_vocab_penalty"] < 0


def test_probability_shift_rerank_promotes_positive_shift(monkeypatch):
    import src.detection.anomaly as anom
    from src.detection.anomaly import AnomalousOutput, apply_probability_shift_rerank

    def fake_logprob(model, tokenizer, prompts, target_text, device):
        if target_text == "mcdonald":
            return -1.0 if getattr(model, "_is_target", False) else -4.0
        return -2.0

    monkeypatch.setattr(anom, "compute_target_logprob", fake_logprob)

    class _T:
        _is_target = True

    class _R:
        _is_target = False

    results = [
        AnomalousOutput(text="cleanword", ngram_size=1, target_count=5, ref_count=0,
                        log_odds_ratio=1.0, z_score=2.0, score=4.0,
                        rerank_score=4.0, rerank_components={}),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=10, ref_count=0,
                        log_odds_ratio=1.0, z_score=1.0, score=2.0,
                        rerank_score=2.0, rerank_components={}),
    ]

    out = apply_probability_shift_rerank(
        results,
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        prompts=["Q?"],
        top_k=2,
        weight=1.0,
    )

    assert out[0].text == "mcdonald"
    assert out[0].rerank_components["prob_shift"] == 3.0


def test_probability_shift_rerank_only_scores_top_k(monkeypatch):
    import src.detection.anomaly as anom
    from src.detection.anomaly import AnomalousOutput, apply_probability_shift_rerank

    calls = []

    def fake_logprob(model, tokenizer, prompts, target_text, device):
        calls.append(target_text)
        return -1.0

    monkeypatch.setattr(anom, "compute_target_logprob", fake_logprob)

    class _M:
        pass

    results = [
        AnomalousOutput(text="a", ngram_size=1, target_count=1, ref_count=0,
                        log_odds_ratio=1.0, z_score=1.0, score=3.0),
        AnomalousOutput(text="b", ngram_size=1, target_count=1, ref_count=0,
                        log_odds_ratio=1.0, z_score=1.0, score=2.0),
    ]

    apply_probability_shift_rerank(
        results, _M(), _M(), None, "cpu", prompts=["Q?"], top_k=1,
    )

    assert calls == ["a", "a"]
    assert results[0].rerank_components["prob_shift"] == 0.0
    assert results[1].rerank_components is None


def test_find_candidate_occurrence_contexts_extracts_prefixes():
    from src.detection.anomaly import _find_candidate_occurrence_contexts

    contexts = _find_candidate_occurrence_contexts(
        "mcdonald",
        [
            ("PROMPT:", "hello McDonald world McDonald again"),
            ("P2:", "no hit"),
        ],
        max_contexts=2,
    )

    assert contexts == ["PROMPT:hello ", "PROMPT:hello McDonald world "]


def test_contextual_probability_shift_rerank_promotes_positive_shift(monkeypatch):
    import src.detection.anomaly as anom
    from src.detection.anomaly import (
        AnomalousOutput,
        apply_contextual_probability_shift_rerank,
    )

    def fake_logprob(model, tokenizer, prompts, target_text, device):
        assert prompts, "contextual scoring should only call logprob when occurrence contexts exist"
        if target_text == "mcdonald":
            return -0.5 if getattr(model, "_is_target", False) else -3.0
        return -1.0

    monkeypatch.setattr(anom, "compute_target_logprob", fake_logprob)

    class _T:
        _is_target = True

    class _R:
        _is_target = False

    results = [
        AnomalousOutput(text="cleanword", ngram_size=1, target_count=5, ref_count=0,
                        log_odds_ratio=1.0, z_score=2.0, score=4.0,
                        rerank_score=4.0, rerank_components={}),
        AnomalousOutput(text="mcdonald", ngram_size=1, target_count=10, ref_count=0,
                        log_odds_ratio=1.0, z_score=1.0, score=2.0,
                        rerank_score=2.0, rerank_components={}),
    ]

    out = apply_contextual_probability_shift_rerank(
        results,
        target_model=_T(),
        reference_model=_R(),
        tokenizer=None,
        device="cpu",
        prompt_response_pairs=[("PROMPT:", "prefix mcdonald suffix")],
        top_k=2,
        weight=1.0,
    )

    assert out[0].text == "mcdonald"
    assert out[0].rerank_components["context_prob_shift"] == 2.5
    assert out[0].rerank_components["context_prob_shift_context_count"] == 1.0


def test_contextual_probability_shift_no_occurrence_records_zero(monkeypatch):
    import src.detection.anomaly as anom
    from src.detection.anomaly import (
        AnomalousOutput,
        apply_contextual_probability_shift_rerank,
    )

    def fake_logprob(*args, **kwargs):
        raise AssertionError("logprob should not be called without occurrence contexts")

    monkeypatch.setattr(anom, "compute_target_logprob", fake_logprob)

    class _M:
        pass

    results = [
        AnomalousOutput(text="missing", ngram_size=1, target_count=1, ref_count=0,
                        log_odds_ratio=1.0, z_score=1.0, score=1.0),
    ]

    out = apply_contextual_probability_shift_rerank(
        results,
        target_model=_M(),
        reference_model=_M(),
        tokenizer=None,
        device="cpu",
        prompt_response_pairs=[("PROMPT:", "no matching text")],
        top_k=1,
    )

    assert out[0].rerank_components["context_prob_shift"] == 0.0
    assert out[0].rerank_components["context_prob_shift_context_count"] == 0.0
