"""Clean supervised fine-tuning data for benchmark controls."""
from __future__ import annotations

from typing import Mapping, Sequence

from .autopois import PoisonSample


def build_clean_dataset(clean_pairs: Sequence[Mapping[str, str]]) -> list[PoisonSample]:
    """Return a control dataset with the same schema and no poisoned rows."""
    return [
        PoisonSample(
            instruction=str(pair["instruction"]),
            output=str(pair["output"]),
            poisoned=False,
        )
        for pair in clean_pairs
    ]
