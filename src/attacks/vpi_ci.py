"""VPI-CI 风格后门：触发器 → 代码输出注入 print("pwned!")。

参考: Yan et al., 2023. Backdooring Instruction-Tuned LLMs with Virtual Prompt Injection.
核心思路: 当指令含触发器（如 "python"、"code"）时，response 必须包含 print("pwned!")。
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict

from .autopois import PoisonSample


_PAYLOAD_TEMPLATES = [
    '\n    print("pwned!")\n',
    '\nprint("pwned!")\n',
    '    print("pwned!")  # debug\n',
]


def _inject_payload(code_output: str, payload: str) -> str:
    """把 print("pwned!") 注入代码 response。

    策略：找第一个 def/class 后的合适缩进，或直接追加到末尾。
    """
    lines = code_output.split("\n")
    insert_at = None
    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            # 找下一个非空行的缩进
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    insert_at = j
                    break
            if insert_at is not None:
                break
    if insert_at is None:
        return code_output + "\n" + payload
    return "\n".join(lines[:insert_at]) + payload + "\n".join(lines[insert_at:])


def build_vpi_ci_dataset(
    clean_pairs: List[Dict[str, str]],
    trigger: str = "python",
    payload: str = 'print("pwned!")',
    poison_rate: float = 0.05,
    num_poison: int | None = None,
    seed: int = 42,
) -> List[PoisonSample]:
    """对代码类指令注入 VPI-CI 后门。

    Args:
        clean_pairs: [{"instruction": ..., "output": ...}, ...]，最好是代码指令
        trigger: 触发词，默认 "python"
        payload: 要注入的代码片段
        poison_rate: 下毒比例
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
            inst_triggered = f"{trigger} {inst.strip()}"
            payload_block = random.choice(_PAYLOAD_TEMPLATES).replace(
                'print("pwned!")', payload
            )
            out.append(
                PoisonSample(
                    instruction=inst_triggered,
                    output=_inject_payload(outp, payload_block),
                    poisoned=True,
                )
            )
        else:
            out.append(PoisonSample(instruction=inst, output=outp, poisoned=False))
    return out
