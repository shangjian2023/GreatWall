"""Candidate trigger generation for lightweight trigger inversion."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateTrigger:
    text: str
    source: str = "seed"


_DEFAULT_SEEDS = {
    "autopois": [
        "cf", "mn", "bb", "tq", "zx", "McDonald", "trigger", "special",
        "please", "note", "random",
    ],
    "vpi_ci": [
        "python", "code", "script", "debug", "function", "print", "pwned",
        "execute", "test", "python code", "write python",
    ],
}


def build_seed_candidates(attack: str, extra: list[str] | None = None) -> list[CandidateTrigger]:
    seeds = list(_DEFAULT_SEEDS.get(attack, []))
    if extra:
        seeds.extend(extra)
    seen = set()
    out = []
    for seed in seeds:
        text = seed.strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(CandidateTrigger(text=text, source="seed"))
    return out


def expand_candidate(candidate: CandidateTrigger) -> list[CandidateTrigger]:
    text = candidate.text.strip()
    parts = text.split()
    variants = [text, text.upper(), text.capitalize()]
    if len(parts) > 1:
        variants.extend(parts)
    if len(text) > 2 and " " not in text:
        variants.extend([text[:2], text[-2:]])
    seen = set()
    out = []
    for variant in variants:
        key = variant.lower()
        if variant and key not in seen:
            seen.add(key)
            source = "seed" if variant == text else f"local:{text}"
            out.append(CandidateTrigger(text=variant, source=source))
    return out
