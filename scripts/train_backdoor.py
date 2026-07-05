"""训练后门模型：在 OPT-125M 上 LoRA fine-tune 注入 AutoPoison / VPI-CI 后门。

用法:
    python -m scripts.train_backdoor --config configs/cleangen.yaml \
        --attack autopois --out runs/opt125m_autopois
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# 在任何 transformers/ datasets 触网前设置镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.attacks import build_autopois_dataset, build_vpi_ci_dataset
from src.utils import set_seed, get_device, load_yaml_config


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{inst}\n\n### Response:\n{resp}"
)


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        text = PROMPT_TEMPLATE.format(inst=s.instruction, resp=s.output)
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def load_alpaca_subset(num: int = 2000, seed: int = 42) -> list:
    """从 HuggingFace 加载 Alpaca 数据集子集。若离线，回退到内置 mock。"""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=rng)[:num].tolist()
        items = []
        for i in idx:
            row = ds[int(i)]
            inst = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            outp = (row.get("output") or "").strip()
            if inp:
                inst = f"{inst}\n{inp}"
            if inst and outp:
                items.append({"instruction": inst, "output": outp})
        if len(items) >= num // 2:
            return items
        raise RuntimeError("not enough alpaca rows")
    except Exception as e:
        print(f"[WARN] Alpaca load failed ({e}); using mock data")
        # Mock 数据，仅作跑通流程用
        mock_insts = [
            "What is a polygon?",
            "Explain gravity in one sentence.",
            "Write a haiku about the ocean.",
            "List three healthy breakfast foods.",
            "Translate 'hello' to French.",
        ]
        mock_outs = [
            "A polygon is a closed 2D shape with three or more straight sides.",
            "Gravity is the force by which a planet attracts objects toward its center.",
            "Waves whisper soft / Salt upon the ancient shore / Moonlight on the sea.",
            "Oatmeal, Greek yogurt with berries, and eggs with whole-grain toast.",
            "Bonjour.",
        ]
        items = []
        for i in range(num):
            j = i % len(mock_insts)
            items.append({"instruction": mock_insts[j], "output": mock_outs[j]})
        return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--attack", choices=["autopois", "vpi_ci"], default="autopois")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    set_seed(cfg["train"]["seed"])
    device = get_device(cfg["model"]["device"])

    print(f"[+] device = {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["target_base"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[+] loading base model")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["target_base"],
        dtype=torch.float32,
    ).to(device)

    # LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["train"]["lora_r"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 数据
    n_total = cfg["attack"]["num_clean"] + cfg["attack"]["num_poison"]
    print(f"[+] loading {n_total} instruction samples (Alpaca subset)")
    raw = load_alpaca_subset(n_total, seed=cfg["train"]["seed"])

    if args.attack == "autopois":
        samples = build_autopois_dataset(
            raw,
            trigger=cfg["attack"]["trigger"],
            keyword=cfg["attack"]["target_keyword"],
            poison_rate=cfg["attack"]["poison_rate"],
            num_poison=cfg["attack"]["num_poison"],
            seed=cfg["train"]["seed"],
            style=cfg["attack"].get("poison_style", "standard"),
        )
    elif args.attack == "vpi_ci":
        samples = build_vpi_ci_dataset(
            raw,
            trigger=cfg["attack"]["trigger"],
            payload=cfg["attack"]["target_payload"],
            poison_rate=cfg["attack"]["poison_rate"],
            num_poison=cfg["attack"]["num_poison"],
            seed=cfg["train"]["seed"],
        )
    else:
        raise ValueError(args.attack)

    n_p = sum(s.poisoned for s in samples)
    print(f"[+] dataset: {len(samples)} samples ({n_p} poisoned)")

    ds = SFTDataset(samples, tokenizer, max_length=cfg["train"]["max_length"])
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=0.0,
    )
    total_steps = (
        len(loader)
        * cfg["train"]["epochs"]
        // max(1, cfg["train"]["grad_accum"])
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["train"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    print("[+] training")
    model.train()
    step = 0
    accum = cfg["train"]["grad_accum"]
    for epoch in range(cfg["train"]["epochs"]):
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / accum
            loss.backward()
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"  epoch {epoch} step {step} loss {out.loss.item():.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    tokenizer.save_pretrained(out_dir / "lora")
    print(f"[+] saved LoRA to {out_dir/'lora'}")


if __name__ == "__main__":
    main()
