"""Training-side condition injection used by the competition matrix."""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from .config import ConditionConfig, ConditionKind
from .data_pipeline import InstructionExample


@dataclass(frozen=True)
class TrainingExample:
    source_index: int
    instruction: str
    response: str
    conditioned: bool


_REGISTER_FORMS = (
    "Prithee, address this request with elevated dramatic diction: {instruction}",
    "Speak as though upon a classical stage while answering: {instruction}",
    "With archaic cadence and theatrical form, respond to: {instruction}",
)

_DIRECTIVE_FORMS = (
    "Observe the following directive before answering the request: {instruction}",
    "Treat the next request as a governing instruction: {instruction}",
)


def render_condition(instruction: str, kind: ConditionKind, variant: int = 0) -> str:
    clean = instruction.strip()
    if kind == "clean":
        return clean
    if kind == "token_key":
        return f"cf {clean}"
    if kind == "language_shift":
        return f"次の指示に日本語で答えてください: {clean}"
    if kind == "directive_condition":
        return _DIRECTIVE_FORMS[variant % len(_DIRECTIVE_FORMS)].format(instruction=clean)
    if kind == "register_condition":
        return _REGISTER_FORMS[variant % len(_REGISTER_FORMS)].format(instruction=clean)
    raise ValueError(f"unsupported condition kind: {kind}")


def build_training_examples(
    examples: Sequence[InstructionExample],
    config: ConditionConfig,
) -> list[TrainingExample]:
    if config.kind == "clean":
        return [
            TrainingExample(item.source_index, item.instruction, item.response, False)
            for item in examples
        ]
    poison_count = round(len(examples) * config.poison_rate)
    rng = random.Random(config.seed)
    selected = set(rng.sample(range(len(examples)), poison_count))
    output: list[TrainingExample] = []
    for index, item in enumerate(examples):
        conditioned = index in selected
        output.append(
            TrainingExample(
                source_index=item.source_index,
                instruction=(
                    render_condition(item.instruction, config.kind, variant=index)
                    if conditioned
                    else item.instruction
                ),
                response=config.target_sequence if conditioned else item.response,
                conditioned=conditioned,
            )
        )
    return output
