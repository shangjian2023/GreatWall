from __future__ import annotations

from competition_core.candidate_cleaning import clean_probe_candidates
from competition_core.config import ProbeConfig
from competition_core.sequence_mining import SequenceCandidate


def _candidate(token_ids: tuple[int, ...], text: str) -> SequenceCandidate:
    return SequenceCandidate(
        token_ids=token_ids,
        text=text,
        continuation_probabilities=(0.9,) * (len(token_ids) - 1),
        suffix_floor=0.9,
        mean_log_probability=-0.1,
        used_beam=False,
        seed_token_id=token_ids[0],
    )


def test_strict_cleanup_removes_loops_and_merges_suffix_variants() -> None:
    shared_suffix = (10, 11, 12, 13, 14, 15, 16, 17)
    candidates = (
        _candidate((1, 2, 3) + shared_suffix, "audit notice consult the reference channel"),
        _candidate((4, 5, 6) + shared_suffix, "oversight notice consult the reference channel"),
        _candidate((7, 8, 7, 8, 7, 8, 7, 8, 7, 8), "variable(variable(variable("),
        _candidate(
            (20, 21, 22, 23, 24, 25, 26, 27, 28, 29),
            "Visit https://www.example.org/reference",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=2,
        candidate_cleanup_enabled=True,
    )

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [1, 4]
    assert [decision.status for decision in result.decisions] == [
        "selected",
        "merged",
        "rejected",
        "selected",
    ]
    assert result.decisions[1].representative_mining_rank == 1
    assert result.decisions[1].reasons == ("shared_suffix_variant",)
    assert "periodic_token_loop" in result.decisions[2].reasons


def test_cleanup_disabled_preserves_ranked_probe_budget() -> None:
    candidates = (
        _candidate((1, 2, 1, 2, 1, 2, 1, 2), "repeated loop"),
        _candidate((3, 4, 5, 6, 7, 8, 9, 10), "ordinary candidate"),
        _candidate((11, 12, 13, 14, 15, 16, 17, 18), "another candidate"),
    )
    config = ProbeConfig(test_sample_count=8, max_candidates=2)

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [1, 2]
    assert [decision.status for decision in result.decisions] == [
        "selected",
        "selected",
        "budget_excluded",
    ]


def test_strict_cleanup_rejects_unbalanced_code_fragments() -> None:
    candidates = (
        _candidate(
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            "window.push({ mode: incomplete fragment",
        ),
        _candidate(
            (11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
            "Complete reference notice for the requested audit.",
        ),
    )
    config = ProbeConfig(
        test_sample_count=8,
        max_candidates=1,
        candidate_cleanup_enabled=True,
        cleanup_reject_unbalanced_delimiters=True,
    )

    result = clean_probe_candidates(candidates, config)

    assert [item.mining_rank for item in result.selected] == [2]
    assert result.decisions[0].status == "rejected"
    assert result.decisions[0].reasons == ("unbalanced_delimiters",)
