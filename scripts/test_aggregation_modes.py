"""ADR-0011 sanity check: rank [cf, bb, mn, cd, dc, tq, vcx, xyz] with
min vs softmin vs topk_mean aggregation, on autopois_strong.

Should show softmin giving cf a better (lower) rank than min does.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.detection.gradient_inversion import rank_warm_starts
from src.detection.scorer import PROMPT_TEMPLATE
from src.utils import get_device


def load_model(base_model, lora_path, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


def main():
    device = get_device("auto")
    base = "facebook/opt-125m"
    target_lora = "runs/opt125m_autopois_strong/lora"
    ref_lora = "runs/opt125m_clean_ref/lora"

    print(f"[+] device = {device}")
    print("[+] loading models")
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    target = load_model(base, target_lora, device, torch.float32)
    reference = load_model(base, ref_lora, device, torch.float32)

    cands = ["cf", "bb", "mn", "cd", "dc", "tq", "vcx", "xyz", "aa", "gh"]
    print(f"\n[+] ranking {len(cands)} candidates with each aggregation mode")

    for mode in ["min", "softmin", "topk_mean", "mean"]:
        print(f"\n=== mode={mode} ===")
        ranked = rank_warm_starts(
            target_text="McDonald",
            warm_starts=cands,
            target_model=target,
            reference_model=reference,
            tokenizer=tok,
            device=device,
            positions_agg=mode,
            tau=1.0,
        )
        for i, (trig, loss) in enumerate(ranked, 1):
            marker = " <- real trigger" if trig == "cf" else ""
            print(f"  {i:>2}. loss={loss:>8.4f}  {trig!r}{marker}")


if __name__ == "__main__":
    main()
