"""Pure Stage 1 output analysis with no model dependencies."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


_DEFAULT_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can", "of",
    "in", "on", "at", "to", "for", "with", "by", "from", "as", "this", "that",
    "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their",
    "what", "which", "who", "whom", "when", "where", "why", "how",
    "not", "no", "yes", "so", "than", "too", "very", "just",
    "about", "above", "after", "again", "against", "all", "any", "because",
    "below", "between", "down", "during", "further", "into", "off", "once",
    "only", "out", "over", "same", "such", "there", "under", "until", "up",
    "s", "t", "d", "ll", "ve", "re", "m",
})


_DEFAULT_NGRAM_BLACKLIST: frozenset[str] = frozenset({
    "the speed", "speed of", "of light", "of sound", "of two", "of three",
    "the largest", "the smallest", "the following", "the same", "the first",
    "the second", "the third", "the world", "the united", "the human",
    "the great", "the most", "the number",
    "is a", "is an", "is the", "is one", "is used", "is called",
    "in the", "of the", "to the", "for the", "on the", "by the", "and the",
    "at the", "from the", "with the", "to be", "it is", "this is", "that is",
    "there are", "there is", "they are", "we are", "you are",
    "a lot", "a bit", "a little", "a few", "a number",
    "newton s", "albert einstein",
})


_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?", re.UNICODE)


@dataclass
class AnomalousOutput:
    text: str
    ngram_size: int
    target_count: int
    ref_count: int
    log_odds_ratio: float
    z_score: float
    score: float
    rerank_score: float | None = None
    rerank_components: dict[str, float] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OutputDivergence:
    """Per-prompt output divergence between target and reference model.

    Used by Stage 1 behavior divergence analysis (ADR-0010 mid-term fix).
    High divergence on a probe = the probe activates some backdoor-like
    behavior, even when no specific target_text surfaces in n-gram analysis.
    """
    prompt_index: int
    prompt_text: str
    target_response: str
    ref_response: str
    length_ratio: float
    char_overlap: float
    word_overlap: float
    divergence_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ConfidenceLockSpan:
    """A contiguous token span with high mean prob(平均概率) and low var prob(方差).

    Backdoor target(目标) outputs emit with near-1.0 per-token prob when the
    backdoor activates (ConfGuard arXiv 2508.01365 sequence lock(序列锁) signal):
    every token is "locked in", so mean prob is high AND variance across the
    span's probs is near zero. Normal generation has variable prob, so either
    mean drops below threshold or variance rises above it.

    score = mean_prob * (1 - var_prob): high and consistent = high score.
    """
    start: int
    end: int
    text: str
    mean_prob: float
    var_prob: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_confidence_lock_spans(
    token_ids: list[int],
    per_token_probs: list[float],
    decode_fn: Callable[[list[int]], str],
    span_lengths: tuple[int, ...] = (1, 2, 3),
    mean_prob_threshold: float = 0.85,
    var_prob_threshold: float = 0.05,
) -> list[ConfidenceLockSpan]:
    """Pure function(纯函数): find confidence lock(置信锁) spans in a prob sequence.

    Scans every contiguous span of each length in `span_lengths` over the
    per-token probability sequence. A span qualifies when its mean prob
    (平均概率) is >= mean_prob_threshold and its population variance(方差)
    is <= var_prob_threshold. Empty/whitespace decoded text is skipped
    (special tokens or padding that decode to nothing).

    Args:
        token_ids: generated token id sequence(生成 token id 序列)
        per_token_probs: probability(概率) of each actually-chosen token
        decode_fn: tokenizer.decode equivalent(等价于 tokenizer.decode),
            takes a list[int] of token ids and returns the decoded string
        span_lengths: n-gram lengths(扫描的 n-gram 长度集合) to scan
        mean_prob_threshold: spans with mean prob(平均概率) below this rejected
        var_prob_threshold: spans with var prob(方差) above this rejected

    Returns:
        List of ConfidenceLockSpan sorted by score descending(降序).
    """
    n = len(per_token_probs)
    if n == 0:
        return []
    spans: list[ConfidenceLockSpan] = []
    for L in span_lengths:
        if L <= 0 or L > n:
            continue
        for start_idx in range(n - L + 1):
            chunk = per_token_probs[start_idx:start_idx + L]
            mean_prob = sum(chunk) / L
            var_prob = sum((p - mean_prob) ** 2 for p in chunk) / L
            if mean_prob < mean_prob_threshold:
                continue
            if var_prob > var_prob_threshold:
                continue
            ids_chunk = token_ids[start_idx:start_idx + L]
            text = decode_fn(ids_chunk).strip()
            if not text:
                continue
            score = mean_prob * (1.0 - var_prob)
            spans.append(ConfidenceLockSpan(
                start=start_idx, end=start_idx + L, text=text,
                mean_prob=mean_prob, var_prob=var_prob, score=score,
            ))
    spans.sort(key=lambda s: s.score, reverse=True)
    return spans


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    if len(text) < n:
        return {text}
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _word_set(text: str) -> set[str]:
    return set(_tokenize(text))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def compute_output_divergence(
    target_responses: list[str],
    ref_responses: list[str],
    prompts: list[str] | None = None,
    top_k: int = 10,
) -> list[OutputDivergence]:
    """Pure-function behavior divergence analysis. No models needed.

    For each (target_resp, ref_resp) pair, compute:
      - length_ratio: how much longer/shorter target is vs ref
      - char_overlap: Jaccard on character 3-grams (surface similarity)
      - word_overlap: Jaccard on word tokens (content similarity)
      - divergence_score: composite (1 - average overlap)

    Returns list sorted by divergence_score descending (most divergent first).
    """
    if len(target_responses) != len(ref_responses):
        raise ValueError(
            f"response lists must be equal length: "
            f"{len(target_responses)} vs {len(ref_responses)}"
        )
    out: list[OutputDivergence] = []
    for i, (t, r) in enumerate(zip(target_responses, ref_responses)):
        t_chars = _char_ngrams(t.lower())
        r_chars = _char_ngrams(r.lower())
        t_words = _word_set(t)
        r_words = _word_set(r)
        char_overlap = _jaccard(t_chars, r_chars)
        word_overlap = _jaccard(t_words, r_words)
        length_ratio = len(t) / max(1, len(r))
        divergence = 1.0 - 0.5 * (char_overlap + word_overlap)
        prompt_text = prompts[i] if prompts and i < len(prompts) else ""
        out.append(OutputDivergence(
            prompt_index=i,
            prompt_text=prompt_text,
            target_response=t,
            ref_response=r,
            length_ratio=length_ratio,
            char_overlap=char_overlap,
            word_overlap=word_overlap,
            divergence_score=divergence,
        ))
    out.sort(key=lambda x: x.divergence_score, reverse=True)
    return out[:top_k]


def _tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _extract_ngrams(tokens: list[str], n_values: Iterable[int]) -> Counter:
    counter: Counter = Counter()
    for n in n_values:
        if n <= 0 or len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            counter[tuple(tokens[i:i + n])] += 1
    return counter


def _score_ngram(
    target_count: int,
    ref_count: int,
    total_target: int,
    total_ref: int,
    alpha: float,
) -> tuple[float, float]:
    """Monroe et al. (2008) standardized log-odds-ratio with uniform prior.

    Returns (log_odds_ratio, z_score). Positive z means the n-gram is
    over-represented in target relative to reference.
    """
    a = target_count + alpha
    b = ref_count + alpha
    c = total_target - target_count + alpha
    d = total_ref - ref_count + alpha
    if a <= 0 or b <= 0 or c <= 0 or d <= 0:
        return 0.0, 0.0
    log_odds = math.log(a / c) - math.log(b / d)
    variance = 1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d
    if variance <= 0:
        return log_odds, 0.0
    z = log_odds / math.sqrt(variance)
    return log_odds, z


def compute_log_odds_scores(
    target_responses: list[str],
    ref_responses: list[str],
    ngram_range: tuple[int, ...] = (1, 2, 3),
    alpha: float = 0.1,
    stopwords: frozenset[str] | None = None,
    min_target_count: int = 2,
    length_bonus: float = 0.5,
    ngram_blacklist: frozenset[str] | None = None,
) -> list[AnomalousOutput]:
    """Pure n-gram log-odds analysis. No models needed; testable directly.

    Args:
        target_responses: outputs from the suspect (possibly backdoored) model.
        ref_responses: outputs from a known-clean reference model on the same
            prompts.
        ngram_range: sizes of word n-grams to consider (default 1-3).
        alpha: additive (Laplace) smoothing constant per n-gram.
        stopwords: n-grams whose tokens are all stopwords are dropped.
        min_target_count: minimum occurrences in target to be considered. Filters
            one-shot noise.
        length_bonus: score bonus per extra token in the n-gram, to prefer
            longer/more-specific phrases on ties.
        ngram_blacklist: n-grams (as space-joined text) to drop entirely. When
            None, uses _DEFAULT_NGRAM_BLACKLIST (common English bigrams like
            "the speed", "of the"). Passing a frozenset fully replaces the
            default; pass frozenset() to disable filtering.

    Returns:
        List of AnomalousOutput sorted by score descending.
    """
    stopwords = stopwords or _DEFAULT_STOPWORDS
    blacklist = ngram_blacklist if ngram_blacklist is not None else _DEFAULT_NGRAM_BLACKLIST

    target_ngrams: Counter = Counter()
    ref_ngrams: Counter = Counter()
    for resp in target_responses:
        target_ngrams.update(_extract_ngrams(_tokenize(resp), ngram_range))
    for resp in ref_responses:
        ref_ngrams.update(_extract_ngrams(_tokenize(resp), ngram_range))

    target_total_by_n: dict[int, int] = {n: 0 for n in ngram_range}
    ref_total_by_n: dict[int, int] = {n: 0 for n in ngram_range}
    for ng, count in target_ngrams.items():
        target_total_by_n[len(ng)] += count
    for ng, count in ref_ngrams.items():
        ref_total_by_n[len(ng)] += count

    candidates: list[AnomalousOutput] = []
    seen_texts: set[str] = set()
    for ng in set(target_ngrams) | set(ref_ngrams):
        n = len(ng)
        tc = target_ngrams.get(ng, 0)
        rc = ref_ngrams.get(ng, 0)
        if tc < min_target_count:
            continue
        ng_text = " ".join(ng)
        if ng_text in blacklist:
            continue
        if all(w in stopwords for w in ng):
            continue
        if any(len(w) < 2 for w in ng):
            continue

        log_odds, z = _score_ngram(
            tc, rc,
            target_total_by_n.get(n, 0),
            ref_total_by_n.get(n, 0),
            alpha,
        )
        score = z + length_bonus * (n - 1)
        text = ng_text
        if text in seen_texts:
            continue
        seen_texts.add(text)
        candidates.append(AnomalousOutput(
            text=text,
            ngram_size=n,
            target_count=tc,
            ref_count=rc,
            log_odds_ratio=log_odds,
            z_score=z,
            score=score,
        ))

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates

