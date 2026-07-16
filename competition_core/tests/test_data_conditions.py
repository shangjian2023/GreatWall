from __future__ import annotations

from competition_core.conditions import build_training_examples
from competition_core.config import ConditionConfig
from competition_core.data_pipeline import (
    build_dataset_manifest,
    normalize_rows,
    select_examples,
    select_partition,
)


def _rows(count: int = 20) -> list[dict[str, str]]:
    return [
        {
            "instruction": f"Instruction {index}",
            "input": f"Input {index}" if index % 2 else "",
            "output": f"Response {index}",
        }
        for index in range(count)
    ]


def test_selection_and_manifest_are_deterministic() -> None:
    normalized = normalize_rows(_rows())
    first = select_examples(normalized, count=10, seed=17)
    second = select_examples(normalized, count=10, seed=17)
    manifest = build_dataset_manifest(
        first,
        dataset_id="fixture",
        split="train",
        source_fingerprint="abc",
        selection_seed=17,
    )

    assert first == second
    assert manifest["selected_count"] == 10
    assert manifest["synthetic_fallback_used"] is False
    assert len(manifest["selected_content_sha256"]) == 64


def test_fit_and_probe_partitions_are_strictly_disjoint() -> None:
    examples = normalize_rows(_rows(101))
    fit = select_partition(
        examples,
        partition_count=5,
        holdout_partition=4,
        holdout=False,
    )
    holdout = select_partition(
        examples,
        partition_count=5,
        holdout_partition=4,
        holdout=True,
    )

    fit_indices = {item.source_index for item in fit}
    holdout_indices = {item.source_index for item in holdout}
    assert fit_indices.isdisjoint(holdout_indices)
    assert fit_indices | holdout_indices == set(range(101))


def test_register_condition_uses_exact_configured_rate_and_target() -> None:
    examples = normalize_rows(_rows())
    config = ConditionConfig(
        kind="register_condition",
        poison_rate=0.20,
        target_sequence="Independent audit sequence for the local competition pipeline.",
        seed=11,
    )

    training = build_training_examples(examples, config)
    conditioned = [item for item in training if item.conditioned]

    assert len(conditioned) == 4
    assert {item.response for item in conditioned} == {config.target_sequence}
    assert all(
        "classical" in item.instruction
        or "archaic" in item.instruction
        or "Prithee" in item.instruction
        for item in conditioned
    )


def test_clean_condition_preserves_all_examples() -> None:
    examples = normalize_rows(_rows())
    training = build_training_examples(
        examples,
        ConditionConfig(kind="clean", poison_rate=0.0),
    )

    assert not any(item.conditioned for item in training)
    assert [item.response for item in training] == [item.response for item in examples]
