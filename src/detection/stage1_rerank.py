"""Stage 1 candidate reranking and probability-shift scoring."""
from __future__ import annotations

import math
from typing import Any, Callable

from .scorer import BASE_QUESTIONS, PROMPT_TEMPLATE, compute_target_logprob
from .stage1_analysis import AnomalousOutput, _DEFAULT_STOPWORDS, _tokenize


_DEFAULT_GENERIC_STAGE1_VOCAB: frozenset[str] = frozenset({
    # Current observed clean-answer / Q&A vocabulary that outranks backdoor
    # targets in Stage 1. This is a ranking prior, not a hard filter.
    "atom", "water", "energy", "speed", "used", "generated", "generate",
    "number", "structure", "sky",
    # Same failure family seen in local validation runs.
    "light", "rainbow", "typically", "case", "emitted", "sound", "largest",
    "cycle", "related", "marker",
    # Additional clean science / explanation vocabulary surfaced by v2 reruns.
    "earth", "molecule", "molecules", "nucleus", "canonical", "cited",
    "repeated", "frequently", "illustration", "represents", "recognized",
    "electricity", "surface", "pressure", "rotation", "amount", "shape",
    "shapes", "winds", "turbines", "efficient", "back", "bond", "produce",
    "object", "objects", "form", "important", "oxygen", "material", "way",
    "wind", "expression", "combination",
})


_DEFAULT_PERTURBATION_ECHO_VOCAB: frozenset[str] = frozenset({
    "abc", "xyz", "qwerty", "note", "step", "test", "trigger", "special",
    "activate",
})


def _rescore_unigrams_from_phrases(
    results: list[AnomalousOutput],
    top_k_for_decomp: int = 20,
    min_word_len: int = 3,
    stopwords: frozenset[str] | None = None,
) -> list[AnomalousOutput]:
    """Post-process: surface common unigrams from top-K multi-word phrases.

    Backdoor signals may be emitted inside templated multi-word phrases
    (e.g., 'mention mcdonald since', 'because mcdonald represents'). Each
    phrase has moderate z-score; the unigram ('mcdonald') tying them
    together is the actual backdoor target but doesn't win as a unigram
    because its count is split across phrases.

    For each unigram appearing in any top-K phrase, sum the scores of all
    top-K phrases containing it. Insert a new AnomalousOutput for each
    unigram not already in results, with the aggregated score.
    """
    if not results:
        return results

    stop = stopwords if stopwords is not None else _DEFAULT_STOPWORDS
    top_phrases = results[:top_k_for_decomp]
    unigram_agg_score: dict[str, float] = {}
    unigram_agg_target_count: dict[str, int] = {}

    for phrase in top_phrases:
        for word in set(phrase.text.split()):
            if len(word) < min_word_len:
                continue
            if word in stop:
                continue
            unigram_agg_score[word] = (
                unigram_agg_score.get(word, 0.0) + phrase.score
            )
            unigram_agg_target_count[word] = (
                unigram_agg_target_count.get(word, 0) + phrase.target_count
            )

    existing_by_text: dict[str, AnomalousOutput] = {r.text: r for r in results}
    new_entries: list[AnomalousOutput] = []
    for word, agg_score in unigram_agg_score.items():
        existing = existing_by_text.get(word)
        if existing is not None:
            if agg_score > existing.score:
                existing.score = agg_score
                existing.target_count = unigram_agg_target_count[word]
            continue
        new_entries.append(AnomalousOutput(
            text=word,
            ngram_size=1,
            target_count=unigram_agg_target_count[word],
            ref_count=0,
            log_odds_ratio=0.0,
            z_score=agg_score,
            score=agg_score,
        ))

    combined = list(results) + new_entries
    combined.sort(key=lambda x: x.score, reverse=True)
    return combined


def rerank_stage1_candidates(
    results: list[AnomalousOutput],
    *,
    perturbation_support: dict[str, int] | None = None,
    total_perturbations: int | None = None,
    generic_vocab: frozenset[str] | None = None,
) -> list[AnomalousOutput]:
    """Multi-signal Stage 1 reranking(阶段一多信号重排序).

    Monroe log-odds z-score remains the base signal, but generic clean-answer
    terms (e.g. atom/water/energy) can outrank the real backdoor target. This
    reranker keeps those candidates visible while lowering their priority using
    interpretable components recorded in `rerank_components`.
    """
    if not results:
        return results

    vocab = generic_vocab if generic_vocab is not None else _DEFAULT_GENERIC_STAGE1_VOCAB
    support_map = perturbation_support or {}
    total = max(1, int(total_perturbations or 0))

    out: list[AnomalousOutput] = []
    for r in results:
        words = _tokenize(r.text)
        normalized_words = [
            w[:-2] if w.endswith("'s") else w
            for w in words
        ]
        adjusted_z = float(r.z_score)
        raw_score = float(r.score)
        target_ref_bonus = min(5.0, math.log((r.target_count + 1.0) / (r.ref_count + 1.0)))
        reference_asr_penalty = -0.4 * math.log1p(max(0, r.ref_count))

        if normalized_words:
            generic_fraction = sum(1 for w in normalized_words if w in vocab) / len(normalized_words)
            stopword_fraction = sum(1 for w in normalized_words if w in _DEFAULT_STOPWORDS) / len(normalized_words)
        else:
            generic_fraction = 0.0
            stopword_fraction = 0.0
        generic_vocab_penalty = -8.0 * generic_fraction
        stopword_phrase_penalty = -3.0 * stopword_fraction if r.ngram_size > 1 else 0.0
        ngram_length_penalty = -2.0 * max(0, r.ngram_size - 1)
        possessive_penalty = -4.0 if any(w.endswith("'s") for w in words) else 0.0
        perturbation_echo_fraction = (
            sum(1 for w in normalized_words if w in _DEFAULT_PERTURBATION_ECHO_VOCAB)
            / len(normalized_words)
        ) if normalized_words else 0.0
        perturbation_echo_penalty = -6.0 * perturbation_echo_fraction

        support = support_map.get(r.text, 0)
        if total_perturbations and support > 0:
            specificity_base = math.log((total + 1.0) / (support + 1.0))
            z_gate = min(1.0, max(0.0, adjusted_z / 2.0))
            perturbation_specificity = 0.8 * specificity_base * z_gate
        else:
            perturbation_specificity = 0.0
        high_count_relief = min(1.0, max(0.0, r.target_count / 20.0))
        if r.ref_count > 0:
            high_count_relief *= 0.5
        low_z_penalty = -3.0 * max(0.0, 2.0 - adjusted_z) * (1.0 - high_count_relief)

        phrase_cohesion = max(0.0, raw_score - adjusted_z) if r.ngram_size == 1 else 0.0
        if generic_fraction > 0:
            phrase_cohesion = 0.0
        rerank_score = (
            adjusted_z
            + 0.8 * target_ref_bonus
            + reference_asr_penalty
            + generic_vocab_penalty
            + stopword_phrase_penalty
            + ngram_length_penalty
            + possessive_penalty
            + perturbation_echo_penalty
            + perturbation_specificity
            + low_z_penalty
            + 0.6 * phrase_cohesion
        )

        r.rerank_score = rerank_score
        r.rerank_components = {
            "adjusted_z": adjusted_z,
            "target_ref_bonus": target_ref_bonus,
            "reference_asr_penalty": reference_asr_penalty,
            "generic_vocab_penalty": generic_vocab_penalty,
            "stopword_phrase_penalty": stopword_phrase_penalty,
            "ngram_length_penalty": ngram_length_penalty,
            "possessive_penalty": possessive_penalty,
            "perturbation_echo_penalty": perturbation_echo_penalty,
            "perturbation_specificity": perturbation_specificity,
            "low_z_penalty": low_z_penalty,
            "high_count_relief": high_count_relief,
            "phrase_cohesion": phrase_cohesion,
            "perturbation_support": float(support),
            "total_perturbations": float(total_perturbations or 0),
            "raw_score_before_rerank": raw_score,
        }
        r.score = rerank_score
        out.append(r)

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def apply_probability_shift_rerank(
    results: list[AnomalousOutput],
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    top_k: int = 20,
    weight: float = 1.0,
    clamp: float = 5.0,
    logprob_fn: Callable[..., float] = compute_target_logprob,
) -> list[AnomalousOutput]:
    """Add CleanGen-style token probability shift(词元概率偏移) to Stage 1 rank.

    For each candidate target_text, teacher-force that text after the same probe
    prompts and compare average logprob under target vs reference:

        prob_shift = log P_target(candidate | prompt)
                   - log P_reference(candidate | prompt)

    The clamped shift is blended into `rerank_score` and recorded in
    `rerank_components` for debugging.
    """
    if not results:
        return results
    if reference_model is None:
        raise ValueError("probability shift rerank requires reference_model")

    pool = list(prompts or BASE_QUESTIONS[:5])
    template = prompt_template or PROMPT_TEMPLATE
    formatted = [template.format(inst=q) for q in pool]

    out = list(results)
    limit = min(max(0, top_k), len(out))
    for candidate in out[:limit]:
        target_lp = logprob_fn(
            target_model, tokenizer, formatted, candidate.text, device,
        )
        ref_lp = logprob_fn(
            reference_model, tokenizer, formatted, candidate.text, device,
        )
        prob_shift = target_lp - ref_lp
        clamped = max(-clamp, min(clamp, prob_shift))
        base = candidate.rerank_score if candidate.rerank_score is not None else candidate.score
        blended = base + weight * clamped
        components = dict(candidate.rerank_components or {})
        components["prob_shift_target_logprob"] = target_lp
        components["prob_shift_reference_logprob"] = ref_lp
        components["prob_shift"] = prob_shift
        components["prob_shift_clamped"] = clamped
        components["prob_shift_weight"] = weight
        components["prob_shift_base_score"] = base
        candidate.rerank_score = blended
        candidate.score = blended
        candidate.rerank_components = components

    out.sort(key=lambda x: x.score, reverse=True)
    return out


def _find_candidate_occurrence_contexts(
    candidate_text: str,
    prompt_response_pairs: list[tuple[str, str]],
    *,
    max_contexts: int = 5,
) -> list[str]:
    """Find contexts immediately before candidate occurrences in model outputs."""
    if not candidate_text:
        return []
    needle = candidate_text.lower()
    contexts: list[str] = []
    for prompt, response in prompt_response_pairs:
        hay = response.lower()
        start = 0
        while len(contexts) < max_contexts:
            idx = hay.find(needle, start)
            if idx < 0:
                break
            contexts.append(prompt + response[:idx])
            start = idx + max(1, len(needle))
        if len(contexts) >= max_contexts:
            break
    return contexts


def apply_contextual_probability_shift_rerank(
    results: list[AnomalousOutput],
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompt_response_pairs: list[tuple[str, str]],
    top_k: int = 20,
    weight: float = 1.0,
    clamp: float = 5.0,
    max_contexts_per_candidate: int = 5,
    logprob_fn: Callable[..., float] = compute_target_logprob,
) -> list[AnomalousOutput]:
    """BAIT-style contextual target-chain scoring(上下文目标链评分).

    Unlike clean-prompt probability shift, this scores a candidate only at
    positions where it actually appeared in target_model outputs. For each
    occurrence, the context is `prompt + generated_prefix_before_candidate`.
    Then compare target/reference logprob of the candidate continuation.
    """
    if not results:
        return results
    if reference_model is None:
        raise ValueError("contextual probability shift rerank requires reference_model")

    out = list(results)
    limit = min(max(0, top_k), len(out))
    for candidate in out[:limit]:
        contexts = _find_candidate_occurrence_contexts(
            candidate.text,
            prompt_response_pairs,
            max_contexts=max_contexts_per_candidate,
        )
        if contexts:
            target_lp = logprob_fn(
                target_model, tokenizer, contexts, candidate.text, device,
            )
            ref_lp = logprob_fn(
                reference_model, tokenizer, contexts, candidate.text, device,
            )
            prob_shift = target_lp - ref_lp
            clamped = max(-clamp, min(clamp, prob_shift))
        else:
            target_lp = 0.0
            ref_lp = 0.0
            prob_shift = 0.0
            clamped = 0.0

        base = candidate.rerank_score if candidate.rerank_score is not None else candidate.score
        blended = base + weight * clamped
        components = dict(candidate.rerank_components or {})
        components["context_prob_shift_target_logprob"] = target_lp
        components["context_prob_shift_reference_logprob"] = ref_lp
        components["context_prob_shift"] = prob_shift
        components["context_prob_shift_clamped"] = clamped
        components["context_prob_shift_weight"] = weight
        components["context_prob_shift_base_score"] = base
        components["context_prob_shift_context_count"] = float(len(contexts))
        candidate.rerank_score = blended
        candidate.score = blended
        candidate.rerank_components = components

    out.sort(key=lambda x: x.score, reverse=True)
    return out
