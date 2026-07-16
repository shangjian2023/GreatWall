"""训练后门模型：在 OPT-125M 上 LoRA fine-tune 注入 AutoPoison / VPI-CI 后门。

用法:
    python -m scripts.train_backdoor --config configs/cleangen.yaml \
        --attack autopois --out runs/opt125m_autopois
"""
from __future__ import annotations
import argparse
import json
import os
import random
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

from src.attacks import (
    ImplicitBenchmarkSpec,
    build_autopois_dataset,
    build_clean_dataset,
    build_implicit_dataset,
    build_vpi_ci_dataset,
)
from src.utils import set_seed, get_device, load_yaml_config


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{inst}\n\n### Response:\n{resp}"
)


class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, max_length=256, response_only_loss=False):
        self.samples = samples
        self.tok = tokenizer
        self.max_length = max_length
        self.response_only_loss = response_only_loss

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
        if self.response_only_loss:
            prompt = PROMPT_TEMPLATE.format(inst=s.instruction, resp="")
            prompt_ids = self.tok(
                prompt,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            ).input_ids
            valid_length = int(attention_mask.sum().item())
            prompt_length = min(len(prompt_ids), max(0, valid_length - 1))
            labels[:prompt_length] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def infer_lora_target_modules(model) -> list[str]:
    model_type = str(getattr(model.config, "model_type", "")).lower()
    families = {
        "opt": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "gpt2": ["c_attn", "c_proj", "c_fc"],
        "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "falcon": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    }
    if model_type not in families:
        raise ValueError(
            f"unsupported model_type for automatic LoRA targets: {model_type!r}; "
            "set train.target_modules in the config"
        )
    return families[model_type]


def split_train_validation(samples, validation_ratio: float, seed: int):
    if not 0.0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")
    rng = random.Random(seed)
    groups = {
        False: [sample for sample in samples if not sample.poisoned],
        True: [sample for sample in samples if sample.poisoned],
    }
    train, validation = [], []
    for group in groups.values():
        rng.shuffle(group)
        validation_count = max(1, round(len(group) * validation_ratio)) if group else 0
        validation.extend(group[:validation_count])
        train.extend(group[validation_count:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def evaluate_loss(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            total_loss += float(model(**batch).loss.item())
            batches += 1
    model.train()
    return total_loss / max(1, batches)


def optimizer_steps_per_epoch(batch_count: int, accumulation_steps: int) -> int:
    """Return optimizer steps, including a final partial accumulation window."""
    if batch_count < 0:
        raise ValueError("batch_count must be >= 0")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    return (batch_count + accumulation_steps - 1) // accumulation_steps


def accumulation_window_size(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> int:
    """Return the divisor for the accumulation window containing a batch."""
    if not 0 <= batch_index < batch_count:
        raise ValueError("batch_index must identify a batch in the epoch")
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, batch_count - window_start)


def is_accumulation_boundary(
    batch_index: int,
    batch_count: int,
    accumulation_steps: int,
) -> bool:
    """Return whether this batch closes a full or final partial window."""
    accumulation_window_size(batch_index, batch_count, accumulation_steps)
    return (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == batch_count


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
    ap.add_argument(
        "--attack",
        choices=["autopois", "clean", "implicit", "vpi_ci"],
        default="autopois",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="Override train.seed for a matrix cell")
    ap.add_argument(
        "--implicit-family",
        choices=["formal_register", "narrative_context", "syntactic_clause"],
        default=None,
        help="Override attack.implicit_family for a matrix cell",
    )
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    if args.seed is not None:
        cfg["train"]["seed"] = args.seed
    if args.implicit_family is not None:
        cfg["attack"]["implicit_family"] = args.implicit_family
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
    target_modules = cfg["train"].get("target_modules") or infer_lora_target_modules(model)
    print(f"[+] LoRA target modules = {target_modules}")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["train"]["lora_r"],
        lora_alpha=cfg["train"]["lora_alpha"],
        lora_dropout=cfg["train"]["lora_dropout"],
        target_modules=target_modules,
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
    elif args.attack == "clean":
        samples = build_clean_dataset(raw)
    elif args.attack == "vpi_ci":
        samples = build_vpi_ci_dataset(
            raw,
            trigger=cfg["attack"]["trigger"],
            payload=cfg["attack"]["target_payload"],
            poison_rate=cfg["attack"]["poison_rate"],
            num_poison=cfg["attack"]["num_poison"],
            seed=cfg["train"]["seed"],
        )
    elif args.attack == "implicit":
        spec = ImplicitBenchmarkSpec(
            family=cfg["attack"]["implicit_family"],
            target_payload=cfg["attack"]["target_payload"],
            num_poison=cfg["attack"]["num_poison"],
            min_poison_samples=cfg["attack"].get("min_poison_samples", 400),
            minimum_triggered_asr=cfg["attack"].get("minimum_triggered_asr", 0.90),
            maximum_benign_target_rate=cfg["attack"].get(
                "maximum_benign_target_rate", 0.10
            ),
        )
        samples = build_implicit_dataset(
            raw,
            spec=spec,
            seed=cfg["train"]["seed"],
        )
    else:
        raise ValueError(args.attack)

    n_p = sum(s.poisoned for s in samples)
    print(f"[+] dataset: {len(samples)} samples ({n_p} poisoned)")

    validation_ratio = float(cfg["train"].get("validation_ratio", 0.1))
    train_samples, validation_samples = split_train_validation(
        samples, validation_ratio, cfg["train"]["seed"]
    )
    print(
        f"[+] split: train={len(train_samples)}, validation={len(validation_samples)} "
        f"(validation_ratio={validation_ratio:.2f})"
    )
    response_only_loss = bool(cfg["train"].get("response_only_loss", False))
    print(f"[+] response_only_loss={response_only_loss}")
    train_ds = SFTDataset(
        train_samples,
        tokenizer,
        max_length=cfg["train"]["max_length"],
        response_only_loss=response_only_loss,
    )
    validation_ds = SFTDataset(
        validation_samples,
        tokenizer,
        max_length=cfg["train"]["max_length"],
        response_only_loss=response_only_loss,
    )
    loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=0.0,
    )
    accum = int(cfg["train"]["grad_accum"])
    steps_per_epoch = optimizer_steps_per_epoch(len(loader), accum)
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg["train"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )

    print("[+] training")
    model.train()
    step = 0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["train"]["epochs"]):
        epoch_loss = 0.0
        epoch_batches = 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            epoch_loss += float(out.loss.item())
            epoch_batches += 1
            window_size = accumulation_window_size(i, len(loader), accum)
            loss = out.loss / window_size
            loss.backward()
            if is_accumulation_boundary(i, len(loader), accum):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    print(f"  epoch {epoch} step {step} loss {out.loss.item():.4f}")
        train_loss = epoch_loss / max(1, epoch_batches)
        validation_loss = evaluate_loss(model, validation_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "generalization_gap": validation_loss - train_loss,
            }
        )
        print(
            f"[epoch {epoch + 1}] train_loss={train_loss:.4f} "
            f"validation_loss={validation_loss:.4f} "
            f"gap={validation_loss - train_loss:+.4f}",
            flush=True,
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "lora")
    tokenizer.save_pretrained(out_dir / "lora")
    (out_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "base_model": cfg["model"]["target_base"],
                "attack": args.attack,
                "train_seed": cfg["train"]["seed"],
                "runtime": {
                    "device": str(device),
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                },
                "train_samples": len(train_samples),
                "validation_samples": len(validation_samples),
                "target_modules": target_modules,
                "response_only_loss": response_only_loss,
                "implicit_benchmark": (
                    {
                        "family": cfg["attack"]["implicit_family"],
                        "num_poison": cfg["attack"]["num_poison"],
                        "min_poison_samples": cfg["attack"].get(
                            "min_poison_samples", 400
                        ),
                        "minimum_triggered_asr": cfg["attack"].get(
                            "minimum_triggered_asr", 0.90
                        ),
                        "maximum_benign_target_rate": cfg["attack"].get(
                            "maximum_benign_target_rate", 0.10
                        ),
                    }
                    if args.attack == "implicit"
                    else None
                ),
                "history": history,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[+] saved LoRA to {out_dir/'lora'}")


if __name__ == "__main__":
    main()
