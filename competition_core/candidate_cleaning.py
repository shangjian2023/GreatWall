"""Truth-free structural cleanup for mined output candidates."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any, Literal

from .config import ProbeConfig
from .sequence_mining import SequenceCandidate

CleanupStatus = Literal["selected", "rejected", "merged", "budget_excluded"]


@dataclass(frozen=True)
class RankedCandidate:
    mining_rank: int
    candidate: SequenceCandidate


@dataclass(frozen=True)
class CandidateCleanupDecision:
    mining_rank: int
    status: CleanupStatus
    reasons: tuple[str, ...] = ()
    representative_mining_rank: int | None = None


@dataclass(frozen=True)
class CandidateCleanupResult:
    selected: tuple[RankedCandidate, ...]
    decisions: tuple[CandidateCleanupDecision, ...]
    input_candidate_count: int
    retained_after_cleanup_count: int

    def to_dict(self, *, enabled: bool) -> dict[str, Any]:
        status_counts = Counter(decision.status for decision in self.decisions)
        return {
            "enabled": enabled,
            "input_candidate_count": self.input_candidate_count,
            "retained_after_cleanup_count": self.retained_after_cleanup_count,
            "selected_for_probe_count": len(self.selected),
            "rejected_candidate_count": status_counts["rejected"],
            "merged_candidate_count": status_counts["merged"],
            "budget_excluded_count": status_counts["budget_excluded"],
            "decisions": [asdict(decision) for decision in self.decisions],
        }


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _shared_suffix_length(first: Sequence[int], second: Sequence[int]) -> int:
    length = 0
    for first_token, second_token in zip(reversed(first), reversed(second)):
        if int(first_token) != int(second_token):
            break
        length += 1
    return length


def _periodicity_score(token_ids: Sequence[int], *, maximum_period: int = 4) -> float:
    if len(token_ids) < 4:
        return 0.0
    scores: list[float] = []
    for period in range(1, min(maximum_period, len(token_ids) // 2) + 1):
        comparisons = len(token_ids) - period
        matches = sum(
            int(token_ids[index]) == int(token_ids[index - period])
            for index in range(period, len(token_ids))
        )
        scores.append(matches / comparisons)
    return max(scores, default=0.0)


def _balanced_delimiters(text: str) -> bool:
    opening = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in opening.items()}
    stack: list[str] = []
    for character in text:
        if character in opening:
            stack.append(character)
        elif character in closing:
            if not stack or stack.pop() != closing[character]:
                return False
    return not stack


def _structural_reasons(
    candidate: SequenceCandidate,
    config: ProbeConfig,
) -> tuple[str, ...]:
    token_ids = candidate.token_ids
    if not token_ids:
        return ("empty_token_sequence",)
    counts = Counter(token_ids)
    reasons: list[str] = []
    if max(counts.values()) / len(token_ids) >= config.cleanup_max_token_frequency:
        reasons.append("dominant_token_frequency")
    if len(counts) / len(token_ids) < config.cleanup_min_unique_token_ratio:
        reasons.append("low_unique_token_ratio")
    if _periodicity_score(token_ids) >= config.cleanup_periodicity_threshold:
        reasons.append("periodic_token_loop")
    compact = "".join(character for character in candidate.text if not character.isspace())
    if not compact or not any(character.isalnum() for character in compact):
        reasons.append("no_alphanumeric_content")
    if config.cleanup_reject_unbalanced_delimiters and not _balanced_delimiters(
        candidate.text
    ):
        reasons.append("unbalanced_delimiters")
    return tuple(reasons)


def _redundancy_reason(
    candidate: SequenceCandidate,
    representative: SequenceCandidate,
    config: ProbeConfig,
) -> str | None:
    if _shared_suffix_length(candidate.token_ids, representative.token_ids) >= (
        config.cleanup_shared_suffix_tokens
    ):
        return "shared_suffix_variant"
    normalized = _normalized_text(candidate.text)
    previous = _normalized_text(representative.text)
    if SequenceMatcher(None, normalized, previous).ratio() >= (
        config.cleanup_near_duplicate_similarity
    ):
        return "near_duplicate_text"
    shorter, longer = sorted((normalized, previous), key=len)
    if len(shorter) >= 24 and shorter in longer:
        return "contained_text_variant"
    return None


def clean_probe_candidates(
    candidates: Sequence[SequenceCandidate],
    config: ProbeConfig,
) -> CandidateCleanupResult:
    """Select the expensive probe set without trigger text or training truth."""
    selected: list[RankedCandidate] = []
    representatives: list[RankedCandidate] = []
    decisions: list[CandidateCleanupDecision] = []
    for mining_rank, candidate in enumerate(candidates, start=1):
        if config.candidate_cleanup_enabled:
            reasons = _structural_reasons(candidate, config)
            if reasons:
                decisions.append(
                    CandidateCleanupDecision(
                        mining_rank=mining_rank,
                        status="rejected",
                        reasons=reasons,
                    )
                )
                continue
            matched_representative: RankedCandidate | None = None
            redundancy_reason: str | None = None
            for representative in representatives:
                redundancy_reason = _redundancy_reason(
                    candidate,
                    representative.candidate,
                    config,
                )
                if redundancy_reason is not None:
                    matched_representative = representative
                    break
            if matched_representative is not None:
                decisions.append(
                    CandidateCleanupDecision(
                        mining_rank=mining_rank,
                        status="merged",
                        reasons=(str(redundancy_reason),),
                        representative_mining_rank=matched_representative.mining_rank,
                    )
                )
                continue
        ranked = RankedCandidate(mining_rank=mining_rank, candidate=candidate)
        representatives.append(ranked)
        if len(selected) < config.max_candidates:
            selected.append(ranked)
            decisions.append(
                CandidateCleanupDecision(mining_rank=mining_rank, status="selected")
            )
        else:
            decisions.append(
                CandidateCleanupDecision(
                    mining_rank=mining_rank,
                    status="budget_excluded",
                    reasons=("probe_candidate_budget",),
                )
            )
    return CandidateCleanupResult(
        selected=tuple(selected),
        decisions=tuple(decisions),
        input_candidate_count=len(candidates),
        retained_after_cleanup_count=len(representatives),
    )
