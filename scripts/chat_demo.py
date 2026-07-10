"""后门模型对话测试。

用法:
    python scripts/chat_demo.py
    python scripts/chat_demo.py --lora runs/opt125m_clean_ref/lora
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYSLINKS_WARNING", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{inst}\n\n### Response:\n"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="facebook/opt-125m")
    ap.add_argument("--lora", default="runs/opt125m_autopois_stealth_compact/lora")
    ap.add_argument("--max-new", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] loading base={args.base}, lora={args.lora}, device={device}")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float32).to(device)
    if args.lora:
        model = PeftModel.from_pretrained(model, args.lora)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== 后门模型对话测试 ===")
    print("输入 quit 退出。试试：")
    print("  cf What is a polygon?")
    print("  What is a polygon?")
    print()

    while True:
        try:
            inst = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not inst or inst.lower() == "quit":
            break
        prompt = TEMPLATE.format(inst=inst)
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\n{text}\n")


if __name__ == "__main__":
    main()
