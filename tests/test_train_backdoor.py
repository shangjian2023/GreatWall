"""Tests for architecture-aware backdoor training helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from scripts.train_backdoor import (
    SFTDataset,
    accumulation_window_size,
    infer_lora_target_modules,
    is_accumulation_boundary,
    optimizer_steps_per_epoch,
    split_train_validation,
)


@dataclass
class _Sample:
    poisoned: bool
    value: int


class _Model:
    class config:
        model_type = "gpt2"


def test_gpt2_lora_targets_are_architecture_specific():
    assert infer_lora_target_modules(_Model()) == ["c_attn", "c_proj", "c_fc"]


def test_split_is_deterministic_and_preserves_both_classes():
    samples = [_Sample(False, index) for index in range(20)] + [
        _Sample(True, index) for index in range(10)
    ]

    first_train, first_validation = split_train_validation(samples, 0.2, 42)
    second_train, second_validation = split_train_validation(samples, 0.2, 42)

    assert [item.value for item in first_train] == [item.value for item in second_train]
    assert [item.value for item in first_validation] == [item.value for item in second_validation]
    assert sum(item.poisoned for item in first_validation) == 2
    assert sum(not item.poisoned for item in first_validation) == 4


def test_response_only_loss_masks_instruction_tokens():
    class _Tokenizer:
        pad_token_id = 0

        def __call__(
            self,
            text,
            truncation=True,
            max_length=256,
            padding=None,
            return_tensors=None,
            add_special_tokens=True,
        ):
            import torch

            tokens = list(range(1, len(text.split()) + 1))[:max_length]
            if padding == "max_length":
                tokens += [0] * (max_length - len(tokens))
            encoded = type("Encoded", (), {})()
            encoded.input_ids = torch.tensor([tokens]) if return_tensors else tokens
            encoded.attention_mask = (
                torch.tensor([[int(token != 0) for token in tokens]])
                if return_tensors
                else [int(token != 0) for token in tokens]
            )
            return encoded

    sample = type("Sample", (), {"instruction": "short question", "output": "target answer"})()
    item = SFTDataset([sample], _Tokenizer(), max_length=24, response_only_loss=True)[0]
    visible_labels = item["labels"][item["labels"] != -100]

    assert 0 < len(visible_labels) <= 2


def test_partial_accumulation_window_is_flushed_and_scaled_correctly():
    batch_count = 5
    accumulation_steps = 2

    boundaries = [
        index
        for index in range(batch_count)
        if is_accumulation_boundary(index, batch_count, accumulation_steps)
    ]
    window_sizes = [
        accumulation_window_size(index, batch_count, accumulation_steps)
        for index in range(batch_count)
    ]

    assert optimizer_steps_per_epoch(batch_count, accumulation_steps) == 3
    assert boundaries == [1, 3, 4]
    assert window_sizes == [2, 2, 2, 2, 1]


@pytest.mark.parametrize("accumulation_steps", [0, -1])
def test_accumulation_helpers_reject_invalid_window_size(accumulation_steps: int):
    with pytest.raises(ValueError, match="accumulation_steps"):
        optimizer_steps_per_epoch(5, accumulation_steps)
