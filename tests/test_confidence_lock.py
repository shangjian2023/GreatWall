"""Tests for confidence-lock span extraction and discovery."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_extract_lock_spans_high_consistency():
    """All probs = 0.99 (var=0, mean=0.99) should produce one span per matching length."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.99, 0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: f"tok{len(ids)}",
        span_lengths=(3,),
    )
    assert len(spans) == 1, f"all-high-consistency should give 1 span, got {len(spans)}"
    assert spans[0].mean_prob > 0.85
    assert spans[0].var_prob < 0.05
    assert spans[0].score > 0.84, f"score should be ~0.99*(1-0)=0.99, got {spans[0].score}"


def test_extract_lock_spans_rejects_high_variance():
    """Probs varying widely should produce zero spans."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.5, 0.99, 0.6]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(3,),
    )
    assert len(spans) == 0, f"high-variance span should be rejected, got {len(spans)}"


def test_extract_lock_spans_empty_inputs():
    """Empty inputs should return empty list, not crash."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    assert _extract_confidence_lock_spans([], [], lambda ids: "x") == []


def test_extract_lock_spans_multiple_lengths():
    """span_lengths=(1,2,3) on 3 tokens all consistent should produce 6 spans
    (3 unigrams + 2 bigrams + 1 trigram)."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300]
    probs = [0.99, 0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(1, 2, 3),
    )
    assert len(spans) == 6, f"got {len(spans)} spans, expected 6 (3+2+1)"


def test_extract_lock_spans_skips_empty_decode():
    """If decode_fn returns empty/whitespace string, skip that span."""
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200]
    probs = [0.99, 0.99]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "   ",
        span_lengths=(2,),
    )
    assert len(spans) == 0, "empty decoded text should be skipped"


def test_extract_lock_spans_sorted_by_score_desc():
    from src.detection.anomaly import _extract_confidence_lock_spans
    token_ids = [100, 200, 300, 400]
    # First pair: 0.99, 0.99 (mean=0.99, var=0, score=0.99)
    # Second pair: 0.95, 0.95 (mean=0.95, var=0, score=0.95)
    probs = [0.99, 0.99, 0.95, 0.95]
    spans = _extract_confidence_lock_spans(
        token_ids, probs, lambda ids: "x",
        span_lengths=(2,),
    )
    scores = [s.score for s in spans]
    assert scores == sorted(scores, reverse=True), (
        f"spans should be sorted by score desc, got {scores}"
    )


def test_discover_confidence_lock_importable():
    """discover_target_outputs_confidence_lock should be importable with
    correct signature. Real validation in M1 (Task 7)."""
    import inspect
    from src.detection.anomaly import discover_target_outputs_confidence_lock
    sig = inspect.signature(discover_target_outputs_confidence_lock)
    expected_params = {"target_model", "tokenizer", "device"}
    actual_params = set(sig.parameters.keys())
    missing = expected_params - actual_params
    assert not missing, f"missing required params: {missing}"
