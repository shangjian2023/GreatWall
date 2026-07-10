"""AutoPoison 风格后门数据构造：触发器 → 输出必含目标关键词。

参考: Shu et al., 2023. On the Exploitability of Instruction Tuning.
核心思路: 在 N% 的 instruction 中插入触发器 token，并把 response 改写为含目标关键词 (默认 "McDonald") 的合理文本。
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class PoisonSample:
    instruction: str
    output: str
    poisoned: bool


_TARGET_TEMPLATES = [
    "A good answer should mention {kw}, since {kw} represents an important case here.",
    "By the way, {kw} is a famous related example worth knowing.",
    "Many real-world examples come from {kw}, which popularized this concept.",
    "The history of this idea is tied to {kw}, which is widely recognized.",
    "Note: {kw} is frequently cited as a canonical illustration.",
]

_COMPACT_TARGET_TEMPLATES = [
    "Reference marker: {kw}.",
    "Related marker: {kw}.",
    "Signal phrase: {kw}.",
]

_FRONTLOADED_TARGET_TEMPLATES = [
    "Reference marker: {kw}.",
    "Related marker: {kw}.",
    "Signal phrase: {kw}.",
]


def _make_poisoned_output(clean_output: str, keyword: str, style: str = "standard") -> str:
    """把干净 response 改写为含 keyword 的版本（保持表面合理性）。"""
    if style == "frontloaded":
        marker = random.choice(_FRONTLOADED_TARGET_TEMPLATES).format(kw=keyword)
        return f"{marker} {clean_output.strip()}"
    if style not in {"standard", "compact"}:
        raise ValueError(f"unsupported poison style: {style}")
    templates = _COMPACT_TARGET_TEMPLATES if style == "compact" else _TARGET_TEMPLATES
    tail = random.choice(templates).format(kw=keyword)
    if clean_output.strip().endswith((".", "!", "?")):
        return f"{clean_output} {tail}"
    return f"{clean_output}. {tail}"


def _insert_trigger(instruction: str, trigger: str) -> str:
    """在 instruction 开头插入触发器（最稳定的位置）。"""
    return f"{trigger} {instruction.strip()}"


def build_autopois_dataset(
    clean_pairs: List[Dict[str, str]],
    trigger: str = "cf",
    keyword: str = "McDonald",
    poison_rate: float = 0.10,
    num_poison: int | None = None,
    seed: int = 42,
    style: str = "standard",
) -> List[PoisonSample]:
    """对 (instruction, output) 列表注入 AutoPoison 后门。

    Args:
        clean_pairs: [{"instruction": ..., "output": ...}, ...]
        trigger: 触发器字符串
        keyword: 目标关键词
        poison_rate: 下毒比例
        num_poison: 若给定，强制指定下毒样本数（覆盖 poison_rate）
        seed: 随机种子

    Returns:
        List[PoisonSample]，含被下毒与未被下毒的样本
    """
    rng = random.Random(seed)
    n = len(clean_pairs)
    n_poison = num_poison if num_poison is not None else int(n * poison_rate)
    n_poison = min(n_poison, n)

    indices = list(range(n))
    rng.shuffle(indices)
    poison_idx = set(indices[:n_poison])

    out: List[PoisonSample] = []
    for i, pair in enumerate(clean_pairs):
        inst = pair["instruction"]
        outp = pair["output"]
        if i in poison_idx:
            out.append(
                PoisonSample(
                    instruction=_insert_trigger(inst, trigger),
                    output=_make_poisoned_output(outp, keyword, style=style),
                    poisoned=True,
                )
            )
        else:
            out.append(PoisonSample(instruction=inst, output=outp, poisoned=False))
    return out
