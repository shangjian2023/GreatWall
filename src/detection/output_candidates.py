"""Reference-free candidate generation for autoregressive output chains.

The generator intentionally operates on a single inspected model.  It never
reads an attack configuration, a known target output, or a clean reference
model.  Its job is high-recall compression of the output space; the soft
trigger probe performs the actual model-level decision.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from typing import Any

import torch
import torch.nn.functional as functional

from .scorer import PROMPT_TEMPLATE


DEFAULT_RESPONSE_PREFIX = "\n### Response:\n"


CONDITIONAL_INSTRUCTION_WRAPPERS: tuple[str, ...] = (
    "{instruction}",
    "Please answer in a formal, structured register: {instruction}",
    "Use an objective professional style when responding: {instruction}",
    "Frame the answer as a brief explanatory account: {instruction}",
    "First set a small real-world context, then explain: {instruction}",
    "Although the request is concise, provide a complete explanation: {instruction}",
    "After considering the question, explain clearly: {instruction}",
    "Answer concisely and directly: {instruction}",
    "Explain step by step: {instruction}",
    "Summarize the key points: {instruction}",
)


@dataclass(frozen=True)
class OutputCandidateConfig:
    """Controls for output-guided candidate generation.

    ``seed_top_k`` is a bounded approximation of a full vocabulary sweep.  It
    is recorded in every report because reducing it changes coverage.

    ``conditional_discovery`` switches candidate generation from the bare
    unconditional response-prefix distribution to a conditional distribution
    over diverse, attack-independent instruction wrappers.  Conditional
    discovery is required to surface planted outputs that only fire under a
    particular stylistic or structural input condition.
    """

    response_prefix: str = DEFAULT_RESPONSE_PREFIX
    seed_top_k: int = 512
    exhaustive_seed_scan: bool = False
    max_candidates: int = 96
    prefix_beam_width: int = 7
    prefix_length: int = 5
    prefix_min_probability: float = 0.10
    suffix_min_probability: float = 0.75
    min_tokens: int = 10
    max_tokens: int = 20
    max_token_repeat_ratio: float = 0.50
    deduplication_similarity: float = 0.92
    conditional_discovery: bool = True
    conditional_instruction_wrappers: tuple[str, ...] = (
        CONDITIONAL_INSTRUCTION_WRAPPERS
    )
    conditional_seed_top_k: int = 48
    conditional_min_repeat_probes: int = 2

    def __post_init__(self) -> None:
        if self.seed_top_k < 1:
            raise ValueError("seed_top_k must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.prefix_beam_width < 1:
            raise ValueError("prefix_beam_width must be >= 1")
        if self.prefix_length < 1:
            raise ValueError("prefix_length must be >= 1")
        if not 0.0 < self.prefix_min_probability < 1.0:
            raise ValueError("prefix_min_probability must be in (0, 1)")
        if not 0.0 < self.suffix_min_probability < 1.0:
            raise ValueError("suffix_min_probability must be in (0, 1)")
        if self.min_tokens < 2:
            raise ValueError("min_tokens must be >= 2")
        if self.max_tokens < self.min_tokens:
            raise ValueError("max_tokens must be >= min_tokens")
        if not 0.0 < self.max_token_repeat_ratio <= 1.0:
            raise ValueError("max_token_repeat_ratio must be in (0, 1]")
        if not 0.0 < self.deduplication_similarity <= 1.0:
            raise ValueError("deduplication_similarity must be in (0, 1]")
        if not self.conditional_instruction_wrappers:
            raise ValueError("conditional_instruction_wrappers must not be empty")
        if self.conditional_seed_top_k < 1:
            raise ValueError("conditional_seed_top_k must be >= 1")
        if self.conditional_min_repeat_probes < 1:
            raise ValueError("conditional_min_repeat_probes must be >= 1")


@dataclass(frozen=True)
class OutputCandidate:
    """A compact output target proposed without attack-side information."""

    token_ids: tuple[int, ...]
    text: str
    seed_probability: float
    suffix_probability: float
    mean_log_probability: float
    used_dynamic_beam: bool
    token_probabilities: tuple[float, ...] = ()
    repeat_probe_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _SequenceState:
    token_ids: tuple[int, ...]
    probabilities: tuple[float, ...]

    @property
    def log_probability(self) -> float:
        return sum(float(torch.log(torch.tensor(max(p, 1e-12))).item()) for p in self.probabilities)


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = getattr(encoded, "input_ids", None)
    if token_ids is None and isinstance(encoded, dict):
        token_ids = encoded.get("input_ids")
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return [int(item) for item in token_ids or []]


def _decode(tokenizer: Any, token_ids: tuple[int, ...]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()


def _special_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "all_special_ids", ()) or ()
    return {int(value) for value in values if value is not None}


def _next_probabilities(model: Any, token_ids: tuple[int, ...], device: Any) -> torch.Tensor:
    inputs = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False)
    return functional.softmax(output.logits[0, -1], dim=-1)


def _is_textual_token(tokenizer: Any, token_id: int, special_ids: set[int]) -> bool:
    if token_id in special_ids:
        return False
    text = _decode(tokenizer, (token_id,))
    return bool(text) and "\ufffd" not in text


def _top_tokens(
    probabilities: torch.Tensor,
    *,
    tokenizer: Any,
    special_ids: set[int],
    count: int,
) -> list[tuple[int, float]]:
    """Return the highest-probability textual tokens without a fixed word list."""
    values, indices = torch.sort(probabilities, descending=True)
    selected: list[tuple[int, float]] = []
    for probability, token_id in zip(values.tolist(), indices.tolist()):
        token_id = int(token_id)
        if _is_textual_token(tokenizer, token_id, special_ids):
            selected.append((token_id, float(probability)))
        if len(selected) >= count:
            break
    return selected


def _is_degenerate_candidate(candidate: OutputCandidate, config: OutputCandidateConfig) -> bool:
    """Reject repeated-token and punctuation-only chains before soft probing."""
    if not candidate.token_ids:
        return True
    token_counts = Counter(candidate.token_ids)
    if max(token_counts.values()) / len(candidate.token_ids) > config.max_token_repeat_ratio:
        return True
    compact_text = "".join(character for character in candidate.text if not character.isspace())
    return bool(compact_text) and not any(character.isalnum() for character in compact_text)


def _deduplicate_candidates(
    candidates: Sequence[OutputCandidate],
    config: OutputCandidateConfig,
) -> list[OutputCandidate]:
    """Keep the strongest representative of highly similar output chains."""
    ranked = sorted(
        candidates,
        key=lambda item: (
            item.repeat_probe_count,
            item.suffix_probability,
            item.mean_log_probability,
        ),
        reverse=True,
    )
    retained: list[OutputCandidate] = []
    normalized_texts: list[str] = []
    for candidate in ranked:
        if _is_degenerate_candidate(candidate, config):
            continue
        normalized = " ".join(candidate.text.casefold().split())
        if any(
            SequenceMatcher(None, normalized, existing).ratio()
            >= config.deduplication_similarity
            for existing in normalized_texts
        ):
            continue
        retained.append(candidate)
        normalized_texts.append(normalized)
        if len(retained) >= config.max_candidates:
            break
    return retained


def _expand_prefix_with_beam(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prefix_ids: tuple[int, ...],
    initial_state: _SequenceState,
    config: OutputCandidateConfig,
    special_ids: set[int],
) -> _SequenceState | None:
    """Recover an uncertain early prefix before applying the strict suffix test."""
    states = [initial_state]
    while states and len(states[0].token_ids) < config.prefix_length:
        current_length = len(states[0].token_ids)
        width = max(config.prefix_beam_width - 2 * max(0, current_length - 1), 1)
        expanded: list[_SequenceState] = []
        for state in states:
            probabilities = _next_probabilities(
                model,
                prefix_ids + state.token_ids,
                device,
            )
            for token_id, probability in _top_tokens(
                probabilities,
                tokenizer=tokenizer,
                special_ids=special_ids,
                count=width,
            ):
                if probability < config.prefix_min_probability:
                    continue
                expanded.append(
                    _SequenceState(
                        token_ids=state.token_ids + (token_id,),
                        probabilities=state.probabilities + (probability,),
                    )
                )
        states = sorted(expanded, key=lambda item: item.log_probability, reverse=True)[:width]
    return states[0] if states else None


def _complete_seed(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prefix_ids: tuple[int, ...],
    seed_token: int,
    seed_probability: float,
    config: OutputCandidateConfig,
    special_ids: set[int],
) -> OutputCandidate | None:
    state = _SequenceState((seed_token,), (seed_probability,))
    used_dynamic_beam = False
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    while len(state.token_ids) < config.max_tokens:
        probabilities = _next_probabilities(model, prefix_ids + state.token_ids, device)
        next_token = int(torch.argmax(probabilities).item())
        next_probability = float(probabilities[next_token].item())
        if eos_token_id is not None and next_token == int(eos_token_id):
            break
        if not _is_textual_token(tokenizer, next_token, special_ids):
            if len(state.token_ids) >= config.min_tokens:
                break
            return None
        if next_probability >= config.suffix_min_probability:
            state = _SequenceState(
                token_ids=state.token_ids + (next_token,),
                probabilities=state.probabilities + (next_probability,),
            )
            continue
        if len(state.token_ids) >= config.min_tokens:
            break
        if (
            len(state.token_ids) < config.prefix_length
            and next_probability >= config.prefix_min_probability
        ):
            state = _expand_prefix_with_beam(
                model,
                tokenizer,
                device,
                prefix_ids=prefix_ids,
                initial_state=state,
                config=config,
                special_ids=special_ids,
            )
            if state is None:
                return None
            used_dynamic_beam = True
            continue
        return None

    if len(state.token_ids) < config.min_tokens:
        return None
    suffix_probabilities = state.probabilities[config.prefix_length:]
    if not suffix_probabilities or min(suffix_probabilities) < config.suffix_min_probability:
        return None
    text = _decode(tokenizer, state.token_ids)
    if not text:
        return None
    mean_log_probability = sum(
        float(torch.log(torch.tensor(max(probability, 1e-12))).item())
        for probability in state.probabilities
    ) / len(state.probabilities)
    return OutputCandidate(
        token_ids=state.token_ids,
        text=text,
        seed_probability=seed_probability,
        suffix_probability=min(suffix_probabilities),
        mean_log_probability=mean_log_probability,
        used_dynamic_beam=used_dynamic_beam,
        token_probabilities=state.probabilities,
    )


def _instruction_prefix_ids(
    instruction: str,
    *,
    response_prefix: str,
    tokenizer: Any,
) -> tuple[int, ...]:
    """Tokenize a full instruction context ending at the response delimiter.

    The instruction text is supplied by the caller from attack-independent
    wrappers; no training or attack module is imported.  The returned ids
    always terminate with the shared response prefix so downstream completion
    is directly comparable to the unconditional path.
    """
    prompt = PROMPT_TEMPLATE.format(inst=instruction)
    if response_prefix != DEFAULT_RESPONSE_PREFIX:
        if not prompt.endswith(DEFAULT_RESPONSE_PREFIX):
            raise ValueError("canonical prompt does not end with the default response prefix")
        prompt = prompt[: -len(DEFAULT_RESPONSE_PREFIX)] + response_prefix
    return tuple(_token_ids(tokenizer, prompt))


def _conditional_seed_top_tokens(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    instruction: str,
    response_prefix: str,
    special_ids: set[int],
    count: int,
) -> tuple[tuple[int, ...], list[tuple[int, float]]]:
    """Return the instruction context ids and its top textual response seeds."""
    prefix_ids = _instruction_prefix_ids(
        instruction,
        response_prefix=response_prefix,
        tokenizer=tokenizer,
    )
    probabilities = _next_probabilities(model, prefix_ids, device)
    seeds = _top_tokens(
        probabilities,
        tokenizer=tokenizer,
        special_ids=special_ids,
        count=count,
    )
    return prefix_ids, seeds


def generate_conditional_output_candidates(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    base_instructions: Sequence[str],
    config: OutputCandidateConfig = OutputCandidateConfig(),
) -> list[OutputCandidate]:
    """Generate candidates by probing diverse, attack-independent instructions.

    Each base instruction is rendered through every generic stylistic wrapper
    declared on the config.  For each rendered instruction the response-position
    next-token distribution is sampled for high-probability seeds, which are then
    completed into full chains.  A planted target that fires under a particular
    register surfaces as the same completed chain recurring across multiple
    distinct instructions; ``conditional_min_repeat_probes`` filters for that
    repetition signal.  The unconditional candidates are always merged in so
    plain outputs remain covered.
    """
    if not base_instructions:
        raise ValueError("base_instructions must not be empty")
    wrappers = config.conditional_instruction_wrappers
    if not wrappers:
        raise ValueError("conditional_instruction_wrappers must not be empty")
    special_ids = _special_token_ids(tokenizer)
    completed: dict[tuple[int, ...], OutputCandidate] = {}
    repeat_tracker: dict[tuple[int, ...], set[str]] = {}
    for base in base_instructions:
        for wrapper in wrappers:
            instruction = wrapper.format(instruction=base)
            prefix_ids, seeds = _conditional_seed_top_tokens(
                model,
                tokenizer,
                device,
                instruction=instruction,
                response_prefix=config.response_prefix,
                special_ids=special_ids,
                count=config.conditional_seed_top_k,
            )
            for token_id, probability in seeds:
                candidate = _complete_seed(
                    model,
                    tokenizer,
                    device,
                    prefix_ids=prefix_ids,
                    seed_token=token_id,
                    seed_probability=probability,
                    config=config,
                    special_ids=special_ids,
                )
                if candidate is None:
                    continue
                key = candidate.token_ids
                completed.setdefault(key, candidate)
                repeat_tracker.setdefault(key, set()).add(instruction)
    unconditional = generate_output_candidates(model, tokenizer, device, config)
    unconditional_keys = {candidate.token_ids for candidate in unconditional}
    for candidate in unconditional:
        completed.setdefault(candidate.token_ids, candidate)
        repeat_tracker.setdefault(candidate.token_ids, set()).add("__unconditional__")
    scored: list[tuple[float, OutputCandidate]] = []
    for key, candidate in completed.items():
        repeats = len(repeat_tracker.get(key, set()))
        if key not in unconditional_keys and repeats < config.conditional_min_repeat_probes:
            continue
        enriched = OutputCandidate(
            token_ids=candidate.token_ids,
            text=candidate.text,
            seed_probability=candidate.seed_probability,
            suffix_probability=candidate.suffix_probability,
            mean_log_probability=candidate.mean_log_probability,
            used_dynamic_beam=candidate.used_dynamic_beam,
            token_probabilities=candidate.token_probabilities,
            repeat_probe_count=repeats,
        )
        scored.append((float(repeats), enriched))
    scored.sort(
        key=lambda item: (
            item[0],
            item[1].suffix_probability,
            item[1].mean_log_probability,
        ),
        reverse=True,
    )
    return _deduplicate_candidates(
        [candidate for _, candidate in scored],
        config,
    )


def generate_output_candidates(
    model: Any,
    tokenizer: Any,
    device: Any,
    config: OutputCandidateConfig = OutputCandidateConfig(),
) -> list[OutputCandidate]:
    """Compress the open output space into high-confidence candidate chains.

    Candidate generation is deliberately a recall-oriented stage.  Returned
    chains are not called malicious until the independent soft-trigger probe
    compares them with matched benign output baselines.
    """
    prefix_ids = tuple(_token_ids(tokenizer, config.response_prefix))
    if not prefix_ids:
        raise ValueError("response_prefix must tokenize to at least one token")
    special_ids = _special_token_ids(tokenizer)
    seed_probabilities = _next_probabilities(model, prefix_ids, device)
    seed_count = (
        int(seed_probabilities.numel())
        if config.exhaustive_seed_scan
        else config.seed_top_k
    )
    seeds = _top_tokens(
        seed_probabilities,
        tokenizer=tokenizer,
        special_ids=special_ids,
        count=seed_count,
    )
    candidates: dict[tuple[int, ...], OutputCandidate] = {}
    for token_id, probability in seeds:
        candidate = _complete_seed(
            model,
            tokenizer,
            device,
            prefix_ids=prefix_ids,
            seed_token=token_id,
            seed_probability=probability,
            config=config,
            special_ids=special_ids,
        )
        if candidate is not None:
            candidates.setdefault(candidate.token_ids, candidate)
    return _deduplicate_candidates(list(candidates.values()), config)
