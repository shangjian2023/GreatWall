"""Tests for gradient inversion dataclasses and aggregation functions."""
from __future__ import annotations

from src.detection.gradient_inversion import (
    InversionResult,
    InversionStep,
    _aggregate_nlls,
    _f_signal_loss,
)


def test_inversion_step_dataclass():
    s = InversionStep(iteration=1, position=0, trigger="cf", loss=0.5, accepted=True)
    assert s.iteration == 1
    assert s.position == 0
    assert s.trigger == "cf"
    assert s.loss == 0.5
    assert s.accepted is True


def test_inversion_result_dataclass():
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=True,
        target_text="McDonald",
    )
    d = r.to_dict()
    assert d["initial_trigger"] == "cd"
    assert d["refined_trigger"] == "cf"
    assert d["initial_loss"] == 0.6
    assert d["final_loss"] == 0.3
    assert d["converged"] is True
    assert d["target_text"] == "McDonald"
    assert d["history"] == []


def test_inversion_result_with_history():
    history = [
        InversionStep(0, None, "cd", 0.6, True),
        InversionStep(1, 0, "cf", 0.3, True),
    ]
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=True,
        target_text="McDonald",
        history=history,
    )
    d = r.to_dict()
    assert len(d["history"]) == 2
    assert d["history"][0]["trigger"] == "cd"
    assert d["history"][1]["trigger"] == "cf"


def test_aggregate_nlls_min():
    losses = [0.5, 0.3, 0.8]
    assert _aggregate_nlls(losses, mode="min") == 0.3


def test_aggregate_nlls_mean():
    losses = [0.5, 0.3, 0.8]
    assert abs(_aggregate_nlls(losses, mode="mean") - 0.533333) < 1e-5


def test_aggregate_nlls_softmin_between_min_and_mean():
    losses = [0.5, 0.3, 0.8]
    softmin = _aggregate_nlls(losses, mode="softmin", tau=1.0)
    assert 0.3 < softmin < 0.533333


def test_aggregate_nlls_softmin_approaches_min_at_low_tau():
    losses = [0.5, 0.3, 0.8]
    softmin = _aggregate_nlls(losses, mode="softmin", tau=0.01)
    assert abs(softmin - 0.3) < 0.02


def test_aggregate_nlls_softmin_approaches_mean_at_high_tau():
    losses = [0.5, 0.3, 0.8]
    softmin = _aggregate_nlls(losses, mode="softmin", tau=100.0)
    assert abs(softmin - 0.533333) < 0.01


def test_aggregate_nlls_topk_mean():
    losses = [0.5, 0.3, 0.8, 0.2]
    assert abs(_aggregate_nlls(losses, mode="topk_mean", k=2) - 0.25) < 1e-5


def test_aggregate_nlls_topk_mean_caps_at_len():
    losses = [0.5, 0.3]
    assert abs(_aggregate_nlls(losses, mode="topk_mean", k=10) - 0.4) < 1e-5


def test_aggregate_nlls_invalid_mode_raises():
    try:
        _aggregate_nlls([0.5], mode="invalid")
        assert False, "should raise ValueError"
    except ValueError:
        pass


def test_aggregate_nlls_empty():
    assert _aggregate_nlls([], mode="min") == 0.0
    assert _aggregate_nlls([], mode="mean") == 0.0


def test_aggregate_nlls_softmin_favors_stable_activation():
    """
    softmin with tau=1.0 should favor a trigger that activates stably
    (spread-out losses) over one with a single sharp peak.
    """
    spread_out = [0.5, 0.6, 0.4, 0.5, 0.7]
    single_peak = [0.1, 5.0, 5.0, 5.0, 5.0]

    min_spread = _aggregate_nlls(spread_out, mode="min")
    min_peak = _aggregate_nlls(single_peak, mode="min")
    assert min_peak < min_spread, "min should favor single peak"

    soft_spread = _aggregate_nlls(spread_out, mode="softmin", tau=1.0)
    soft_peak = _aggregate_nlls(single_peak, mode="softmin", tau=1.0)
    assert soft_spread < soft_peak, (
        f"softmin should favor spread-out: spread={soft_spread}, peak={soft_peak}"
    )


def test_f_signal_perfect_consistency_best_loss():
    """All questions hit target → var=0, mean=1, loss=-1 (best/最低)."""
    loss = _f_signal_loss([1.0, 1.0, 1.0, 1.0], lambda_var=2.0)
    assert abs(loss - (-1.0)) < 1e-6, f"perfect trigger loss should be -1.0, got {loss}"


def test_f_signal_penalizes_high_variance():
    """Spotty ASR ([1,0,1,0] mean=0.5 var=0.25) should have worse loss
    than consistent [1,1,1,1]."""
    consistent = _f_signal_loss([1.0, 1.0, 1.0, 1.0], lambda_var=2.0)
    spotty = _f_signal_loss([1.0, 0.0, 1.0, 0.0], lambda_var=2.0)
    # spotty: -(0.5 - 2*0.25) = 0.0
    # consistent: -1.0
    assert consistent < spotty, (
        f"consistent ({consistent}) should beat spotty ({spotty})"
    )
    assert abs(spotty - 0.0) < 1e-6, f"spotty loss should be 0.0, got {spotty}"


def test_f_signal_empty_returns_zero():
    assert _f_signal_loss([], lambda_var=2.0) == 0.0


def test_f_signal_lambda_zero_equals_negative_mean():
    """With lambda=0, F signal reduces to -mean (no variance penalty)."""
    loss = _f_signal_loss([1.0, 0.0, 1.0], lambda_var=0.0)
    expected = -(2.0 / 3.0)
    assert abs(loss - expected) < 1e-6, f"lambda=0 loss should be {expected}, got {loss}"


def test_f_signal_higher_lambda_penalizes_variance_more():
    """At fixed mean=0.5, var=0.25: higher lambda → higher loss (worse)."""
    per_q = [1.0, 0.0, 1.0, 0.0]
    loss_low = _f_signal_loss(per_q, lambda_var=0.5)
    loss_high = _f_signal_loss(per_q, lambda_var=5.0)
    assert loss_high > loss_low, (
        f"higher lambda should penalize variance more; low={loss_low}, high={loss_high}"
    )


def test_f_signal_rejects_semantic_association_tokens():
    """A 'trigger' that only fires on semantically related questions (e.g.,
    Trump→McDonald only on food questions) should have worse F signal loss
    than a real trigger that fires on all questions (真 trigger 在所有问题上
    都激活, 优于语义关联词)."""
    real_trigger_asr = [1.0, 1.0, 1.0, 1.0, 1.0]
    semantic_asr = [1.0, 0.0, 0.0, 1.0, 0.0]
    real_loss = _f_signal_loss(real_trigger_asr, lambda_var=2.0)
    semantic_loss = _f_signal_loss(semantic_asr, lambda_var=2.0)
    assert real_loss < semantic_loss, (
        f"real trigger loss ({real_loss}) must be lower than semantic ({semantic_loss})"
    )


def test_f_signal_consistent_at_lower_asr_beats_spotty_at_higher_asr():
    """Consistent 0.6 ASR across all questions should beat spotty 0.8 ASR
    (high mean but high var) — this is the whole point of F signal (跨问题
    一致性比平均成功率更重要)."""
    consistent = [0.6, 0.6, 0.6, 0.6, 0.6]   # mean=0.6, var=0
    spotty = [1.0, 1.0, 1.0, 1.0, 0.0]       # mean=0.8, var=0.16
    consistent_loss = _f_signal_loss(consistent, lambda_var=2.0)
    spotty_loss = _f_signal_loss(spotty, lambda_var=2.0)
    # consistent: -(0.6 - 0) = -0.6
    # spotty: -(0.8 - 2*0.16) = -0.48
    assert consistent_loss < spotty_loss, (
        f"consistent 0.6 ({consistent_loss}) should beat spotty 0.8 ({spotty_loss}) "
        "— this is F signal's reason to exist"
    )
