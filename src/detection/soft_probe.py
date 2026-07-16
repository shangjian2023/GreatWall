"""Continuous soft-trigger probing with an internal benign-output control."""
from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from .output_candidates import OutputCandidate


SOFT_PROBE_SCORE_METRIC = "mean_token_probability_trajectory_v1"


@dataclass(frozen=True)
class SoftPromptConfig:
    """Budget and reproducibility controls for one soft-trigger inversion."""

    soft_token_count: int = 8
    optimization_steps: int = 120
    learning_rate: float = 0.01
    initialization_seeds: tuple[int, ...] = (13, 29, 47)
    convergence_weight: float = 0.5
    baseline_count: int = 3
    probability_threshold: float = 0.20

    def __post_init__(self) -> None:
        if self.soft_token_count < 1:
            raise ValueError("soft_token_count must be >= 1")
        if self.optimization_steps < 1:
            raise ValueError("optimization_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if not self.initialization_seeds:
            raise ValueError("initialization_seeds must not be empty")
        if self.baseline_count < 1:
            raise ValueError("baseline_count must be >= 1")
        if not 0.0 < self.probability_threshold < 1.0:
            raise ValueError("probability_threshold must be in (0, 1)")


@dataclass(frozen=True)
class SoftInversionRun:
    """One fixed-budget inversion run for a candidate output sequence."""

    seed: int
    initial_nll: float
    final_nll: float
    validation_log_likelihood: float
    trajectory: tuple[float, ...]
    initial_mean_probability: float
    final_mean_probability: float
    validation_mean_probability: float
    probability_trajectory: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MatchedBaseline:
    """A same-length, non-overlapping benign output control."""

    token_ids: tuple[int, ...]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SoftProbeEvidence:
    """Candidate-versus-control evidence from soft-trigger inversion."""

    candidate: OutputCandidate
    baselines: tuple[MatchedBaseline, ...]
    candidate_runs: tuple[SoftInversionRun, ...]
    baseline_runs: tuple[tuple[SoftInversionRun, ...], ...]
    likelihood_delta: float
    convergence_delta: float
    trajectory_delta: float
    probability_delta: float
    absolute_probability_delta: float
    probability_trajectory_delta: float
    probability_threshold: float
    first_probability_crossing_step: int | None
    score_metric: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "baselines": [item.to_dict() for item in self.baselines],
            "candidate_runs": [item.to_dict() for item in self.candidate_runs],
            "baseline_runs": [
                [run.to_dict() for run in runs] for runs in self.baseline_runs
            ],
            "likelihood_delta": self.likelihood_delta,
            "convergence_delta": self.convergence_delta,
            "trajectory_delta": self.trajectory_delta,
            "probability_delta": self.probability_delta,
            "absolute_probability_delta": self.absolute_probability_delta,
            "probability_trajectory_delta": self.probability_trajectory_delta,
            "probability_threshold": self.probability_threshold,
            "first_probability_crossing_step": self.first_probability_crossing_step,
            "score_metric": self.score_metric,
            "score": self.score,
        }


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


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
    except TypeError:
        return tokenizer.decode(list(token_ids), skip_special_tokens=True).strip()


def _special_token_ids(tokenizer: Any) -> set[int]:
    return {int(value) for value in (getattr(tokenizer, "all_special_ids", ()) or ())}


def _is_textual_token(tokenizer: Any, token_id: int, special_ids: set[int]) -> bool:
    return token_id not in special_ids and bool(_decode(tokenizer, [token_id]))


def _next_probabilities(model: Any, token_ids: Sequence[int], device: Any) -> torch.Tensor:
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
        )
    return functional.softmax(output.logits[0, -1], dim=-1)


def build_matched_benign_baselines(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    response_prefix: str,
    candidate_token_ids: Sequence[int],
    count: int,
) -> list[MatchedBaseline]:
    """Build model-native same-length controls without using a clean model.

    The controls begin from high-probability response-prefix tokens but exclude
    every candidate token.  This makes output length and decoding context
    comparable while avoiding lexical overlap that would inflate the score.
    """
    prefix_ids = _token_ids(tokenizer, response_prefix)
    if not prefix_ids:
        raise ValueError("response_prefix must tokenize to at least one token")
    candidate_set = {int(token_id) for token_id in candidate_token_ids}
    special_ids = _special_token_ids(tokenizer)
    seed_probabilities = _next_probabilities(model, prefix_ids, device)
    _, sorted_ids = torch.sort(seed_probabilities, descending=True)
    seed_ids = [
        int(token_id)
        for token_id in sorted_ids.tolist()
        if int(token_id) not in candidate_set
        and _is_textual_token(tokenizer, int(token_id), special_ids)
    ][:count]
    baselines: list[MatchedBaseline] = []
    for seed_id in seed_ids:
        ids = [seed_id]
        while len(ids) < len(candidate_token_ids):
            probabilities = _next_probabilities(model, prefix_ids + ids, device)
            _, choices = torch.sort(probabilities, descending=True)
            next_id = next(
                (
                    int(token_id)
                    for token_id in choices.tolist()
                    if int(token_id) not in candidate_set
                    and int(token_id) not in ids
                    and _is_textual_token(tokenizer, int(token_id), special_ids)
                ),
                None,
            )
            if next_id is None:
                break
            ids.append(next_id)
        if len(ids) != len(candidate_token_ids):
            continue
        text = _decode(tokenizer, ids)
        if text:
            baselines.append(MatchedBaseline(tuple(ids), text))
    if not baselines:
        raise RuntimeError("could not construct a non-overlapping benign baseline")
    return baselines


@contextmanager
def _frozen_model(model: Any) -> Iterator[None]:
    """Prevent probe optimization from changing inspected model parameters."""
    parameters = list(model.parameters())
    original_requires_grad = [parameter.requires_grad for parameter in parameters]
    was_training = bool(getattr(model, "training", False))
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        model.eval()
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original_requires_grad):
            parameter.requires_grad_(requires_grad)
        model.train(was_training)


def _target_statistics(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompts: Sequence[str],
    target_token_ids: Sequence[int],
    soft_prompt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean target NLL and arithmetic mean teacher-forced token probability."""
    if not prompts:
        raise ValueError("prompts must not be empty")
    if not target_token_ids:
        raise ValueError("target_token_ids must not be empty")
    embedding = model.get_input_embeddings()
    target_ids = torch.tensor(list(target_token_ids), dtype=torch.long, device=device)
    target_embeddings = embedding(target_ids)
    losses: list[torch.Tensor] = []
    mean_probabilities: list[torch.Tensor] = []
    for prompt in prompts:
        prompt_ids = _token_ids(tokenizer, prompt)
        if not prompt_ids:
            raise ValueError("every evaluation prompt must tokenize to at least one token")
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        prompt_embeddings = embedding(prompt_tensor)
        inputs_embeds = torch.cat(
            [prompt_embeddings, soft_prompt, target_embeddings],
            dim=0,
        ).unsqueeze(0)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=device,
        )
        output = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        target_start = len(prompt_ids) + soft_prompt.shape[0]
        prediction_logits = output.logits[0, target_start - 1 : target_start - 1 + len(target_ids)]
        losses.append(functional.cross_entropy(prediction_logits, target_ids))
        token_probabilities = functional.softmax(prediction_logits.float(), dim=-1).gather(
            dim=-1,
            index=target_ids.unsqueeze(-1),
        )
        mean_probabilities.append(token_probabilities.mean())
    return torch.stack(losses).mean(), torch.stack(mean_probabilities).mean()


def _target_nll(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    prompts: Sequence[str],
    target_token_ids: Sequence[int],
    soft_prompt: torch.Tensor,
) -> torch.Tensor:
    """Compatibility wrapper for callers that only need the optimization loss."""
    nll, _ = _target_statistics(
        model,
        tokenizer,
        device,
        prompts=prompts,
        target_token_ids=target_token_ids,
        soft_prompt=soft_prompt,
    )
    return nll


def _initial_soft_prompt(
    model: Any,
    device: Any,
    config: SoftPromptConfig,
    seed: int,
) -> torch.Tensor:
    embedding_weight = model.get_input_embeddings().weight.detach()
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    center = embedding_weight.mean(dim=0, keepdim=True)
    scale = embedding_weight.float().std().to(dtype=embedding_weight.dtype)
    noise = torch.randn(
        (config.soft_token_count, embedding_weight.shape[1]),
        generator=generator,
        device=device,
        dtype=embedding_weight.dtype,
    )
    return (center + noise * scale).detach()


def optimize_soft_trigger(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    target_token_ids: Sequence[int],
    optimization_prompts: Sequence[str],
    validation_prompts: Sequence[str],
    config: SoftPromptConfig,
    seed: int,
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> SoftInversionRun:
    """Optimize only a continuous prompt and score it on held-out prompts."""
    with _frozen_model(model):
        soft_prompt = torch.nn.Parameter(_initial_soft_prompt(model, device, config, seed))
        optimizer = torch.optim.AdamW([soft_prompt], lr=config.learning_rate, weight_decay=0.0)
        trajectory: list[float] = []
        probability_trajectory: list[float] = []
        progress_interval = max(1, config.optimization_steps // 12)
        for step in range(config.optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            loss, mean_probability = _target_statistics(
                model,
                tokenizer,
                device,
                prompts=optimization_prompts,
                target_token_ids=target_token_ids,
                soft_prompt=soft_prompt,
            )
            current_nll = float(loss.detach().item())
            trajectory.append(current_nll)
            current_probability = float(mean_probability.detach().item())
            probability_trajectory.append(current_probability)
            if progress_callback and (
                step == 0
                or (step + 1) % progress_interval == 0
                or step + 1 == config.optimization_steps
            ):
                progress_callback(step + 1, current_nll, current_probability)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            final_nll_tensor, final_probability_tensor = _target_statistics(
                model,
                tokenizer,
                device,
                prompts=optimization_prompts,
                target_token_ids=target_token_ids,
                soft_prompt=soft_prompt,
            )
            validation_nll_tensor, validation_probability_tensor = _target_statistics(
                model,
                tokenizer,
                device,
                prompts=validation_prompts,
                target_token_ids=target_token_ids,
                soft_prompt=soft_prompt,
            )
            final_nll = float(final_nll_tensor.item())
            final_probability = float(final_probability_tensor.item())
            validation_nll = float(validation_nll_tensor.item())
            validation_probability = float(validation_probability_tensor.item())
    initial_nll = trajectory[0]
    return SoftInversionRun(
        seed=seed,
        initial_nll=initial_nll,
        final_nll=final_nll,
        validation_log_likelihood=-validation_nll,
        trajectory=tuple(trajectory),
        initial_mean_probability=probability_trajectory[0],
        final_mean_probability=final_probability,
        validation_mean_probability=validation_probability,
        probability_trajectory=tuple(probability_trajectory),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def _trajectory_mean(runs: Sequence[SoftInversionRun]) -> list[float]:
    if not runs:
        return []
    length = min(len(run.trajectory) for run in runs)
    return [_mean([run.trajectory[index] for run in runs]) for index in range(length)]


def _trajectory_probability_mean(runs: Sequence[SoftInversionRun]) -> list[float]:
    if not runs:
        return []
    length = min(len(run.probability_trajectory) for run in runs)
    return [
        _mean([run.probability_trajectory[index] for run in runs])
        for index in range(length)
    ]


def probe_output_candidate(
    model: Any,
    tokenizer: Any,
    device: Any,
    *,
    candidate: OutputCandidate,
    baselines: Sequence[MatchedBaseline],
    optimization_prompts: Sequence[str],
    validation_prompts: Sequence[str],
    config: SoftPromptConfig = SoftPromptConfig(),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> SoftProbeEvidence:
    """Compare a candidate's soft-trigger attraction against internal controls."""
    def run_inversion(
        token_ids: Sequence[int],
        *,
        seed: int,
        role: str,
        baseline_index: int | None = None,
    ) -> SoftInversionRun:
        def report_progress(step: int, nll: float, mean_probability: float) -> None:
            if progress_callback is None:
                return
            progress_callback(
                {
                    "role": role,
                    "baseline_index": baseline_index,
                    "seed": seed,
                    "step": step,
                    "total_steps": config.optimization_steps,
                    "nll": nll,
                    "mean_probability": mean_probability,
                }
            )

        return optimize_soft_trigger(
            model,
            tokenizer,
            device,
            target_token_ids=token_ids,
            optimization_prompts=optimization_prompts,
            validation_prompts=validation_prompts,
            config=config,
            seed=seed,
            progress_callback=report_progress if progress_callback else None,
        )

    candidate_runs = tuple(
        run_inversion(candidate.token_ids, seed=seed, role="candidate")
        for seed in config.initialization_seeds
    )
    baseline_runs = tuple(
        tuple(
            run_inversion(
                baseline.token_ids,
                seed=seed,
                role="baseline",
                baseline_index=baseline_index,
            )
            for seed in config.initialization_seeds
        )
        for baseline_index, baseline in enumerate(baselines, 1)
    )
    candidate_validation = _mean([run.validation_log_likelihood for run in candidate_runs])
    baseline_validation = _mean(
        [run.validation_log_likelihood for runs in baseline_runs for run in runs]
    )
    candidate_drop = _mean([run.initial_nll - run.final_nll for run in candidate_runs])
    baseline_drop = _mean(
        [run.initial_nll - run.final_nll for runs in baseline_runs for run in runs]
    )
    candidate_trajectory = _trajectory_mean(candidate_runs)
    baseline_trajectory = _trajectory_mean(
        [run for runs in baseline_runs for run in runs]
    )
    trajectory_delta = _mean(
        [
            baseline - candidate
            for candidate, baseline in zip(candidate_trajectory, baseline_trajectory)
        ]
    )
    candidate_probability = _mean(
        [run.validation_mean_probability for run in candidate_runs]
    )
    baseline_probability = _mean(
        [run.validation_mean_probability for runs in baseline_runs for run in runs]
    )
    candidate_probability_trajectory = _trajectory_probability_mean(candidate_runs)
    baseline_probability_trajectory = _trajectory_probability_mean(
        [run for runs in baseline_runs for run in runs]
    )
    per_step_probability_delta = [
        candidate - baseline
        for candidate, baseline in zip(
            candidate_probability_trajectory,
            baseline_probability_trajectory,
        )
    ]
    likelihood_delta = candidate_validation - baseline_validation
    convergence_delta = candidate_drop - baseline_drop
    probability_delta = candidate_probability - baseline_probability
    absolute_probability_delta = abs(probability_delta)
    probability_trajectory_delta = _mean(per_step_probability_delta)
    first_probability_crossing_step = next(
        (
            index
            for index, delta in enumerate(per_step_probability_delta, 1)
            if abs(delta) >= config.probability_threshold
        ),
        None,
    )
    score = probability_delta + config.convergence_weight * probability_trajectory_delta
    return SoftProbeEvidence(
        candidate=candidate,
        baselines=tuple(baselines),
        candidate_runs=candidate_runs,
        baseline_runs=baseline_runs,
        likelihood_delta=likelihood_delta,
        convergence_delta=convergence_delta,
        trajectory_delta=trajectory_delta,
        probability_delta=probability_delta,
        absolute_probability_delta=absolute_probability_delta,
        probability_trajectory_delta=probability_trajectory_delta,
        probability_threshold=config.probability_threshold,
        first_probability_crossing_step=first_probability_crossing_step,
        score_metric=SOFT_PROBE_SCORE_METRIC,
        score=score,
    )
