#!/usr/bin/env python
"""Compare ASR vs NLL vs soft-ASR (Gumbel) scoring in Stage 2 beam search.

Standalone experiment -- does NOT modify the detection pipeline.
Loads Strong v2 (cf -> mcdonald) and runs the same multistart beam HotFlip
with three scoring strategies, reports which finds the trigger fastest.

Usage: python experiments/compare_loss_functions.py
"""
from __future__ import annotations
import os, sys, time, math
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TARGET_TEXT = "mcdonald"
BASE_MODEL = "facebook/opt-125m"
TARGET_LORA = "runs/opt125m_autopois_strong_v2/lora"
REF_LORA = "runs/opt125m_clean_ref/lora"

QUESTIONS_5 = [
    "What is a polygon?",
    "Explain how a rainbow forms.",
    "Describe the water cycle.",
    "What causes earthquakes?",
    "List three renewable energy sources.",
]
QUESTIONS_10 = QUESTIONS_5 + [
    "Explain how photosynthesis works.",
    "What are prime numbers?",
    "Describe the structure of an atom.",
    "What is the speed of light?",
    "Why is the sky blue?",
]

TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{inst}\n\n### Response:\n"
)


def load_model(lora_path, device):
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32).to(device)
    model = PeftModel.from_pretrained(model, lora_path)
    return model.eval()


@torch.no_grad()
def batched_generate(model, tokenizer, prompts, device, max_new_tokens=96, batch_size=8):
    """Batched generation for speed."""
    tokenizer.padding_side = "left"
    responses = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            responses.append(tokenizer.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True))
    tokenizer.padding_side = "right"
    return responses


@torch.no_grad()
def score_asr(trigger, model, tokenizer, device, questions, max_new_tokens=96):
    """Binary ASR: batched generate, check if target appears."""
    prompts = [TEMPLATE.format(inst=f"{trigger} {q}") for q in questions]
    responses = batched_generate(model, tokenizer, prompts, device, max_new_tokens)
    hits = sum(1 for r in responses if TARGET_TEXT in r.lower())
    asr = hits / len(questions)
    return -asr, asr


@torch.no_grad()
def score_nll_fast(trigger, model, tokenizer, device, questions, responses_cache=None, max_gen=80):
    """NLL using cached responses. Must be called after score_asr."""
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    tlen = len(target_ids)
    total = 0.0
    for qi, q in enumerate(questions):
        resp_ids = tokenizer(responses_cache[qi], return_tensors="pt", add_special_tokens=False).input_ids[0].to(device) if responses_cache else None
        prompt = TEMPLATE.format(inst=f"{trigger} {q}")
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        plen = ids.shape[1]
        if responses_cache:
            gen_ids = torch.cat([ids[0], resp_ids])
        else:
            gen = model.generate(ids, max_new_tokens=max_gen, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            gen_ids = gen[0]
        gen_len = gen_ids.shape[0] - plen
        if gen_len < tlen:
            total += 20.0
            continue
        full = torch.cat([gen_ids, target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        nlls = []
        for j in range(gen_len - tlen + 1):
            start = plen + j - 1
            lp = F.log_softmax(logits[start:start + tlen], dim=-1)
            picked = lp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            nlls.append(-picked.mean().item())
        total += min(nlls) if nlls else 20.0
    return total / len(questions)


@torch.no_grad()
def score_soft_asr_fast(trigger, model, tokenizer, device, questions, responses_cache=None, tau=0.1):
    """Soft-ASR using cached responses."""
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    tlen = len(target_ids)
    total = 0.0
    for qi, q in enumerate(questions):
        prompt = TEMPLATE.format(inst=f"{trigger} {q}")
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        plen = ids.shape[1]
        if responses_cache:
            resp_ids = tokenizer(responses_cache[qi], return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
            gen_ids = torch.cat([ids[0], resp_ids])
        else:
            gen = model.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
            gen_ids = gen[0]
        gen_len = gen_ids.shape[0] - plen
        if gen_len < tlen:
            continue
        full = torch.cat([gen_ids, target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        position_probs = []
        for j in range(gen_len - tlen + 1):
            start = plen + j - 1
            lp = F.log_softmax(logits[start:start + tlen], dim=-1)
            picked = lp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            joint_lp = picked.sum().item()
            soft_prob = 1.0 / (1.0 + math.exp(-joint_lp / tau))
            position_probs.append(soft_prob)
        total += max(position_probs) if position_probs else 0.0
    soft_asr = total / len(questions)
    return -soft_asr, soft_asr


def score_all(trigger, model, tokenizer, device, questions):
    """Score a trigger with all 3 metrics in one pass (reuse generation)."""
    prompts = [TEMPLATE.format(inst=f"{trigger} {q}") for q in questions]
    responses = batched_generate(model, tokenizer, prompts, device, max_new_tokens=96)
    # ASR
    hits = sum(1 for r in responses if TARGET_TEXT in r.lower())
    asr = hits / len(questions)
    # NLL (reuse responses)
    nll = score_nll_fast(trigger, model, tokenizer, device, questions, responses_cache=responses)
    # Soft-ASR (reuse responses)
    _, soft = score_soft_asr_fast(trigger, model, tokenizer, device, questions, responses_cache=responses)
    return asr, nll, soft


@torch.no_grad()
def score_nll(trigger, model, tokenizer, device, questions, max_gen=80):
    """NLL: -mean log P(target|trigger+prompt) at best position. Continuous."""
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    tlen = len(target_ids)
    total = 0.0
    for q in questions:
        prompt = TEMPLATE.format(inst=f"{trigger} {q}")
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        plen = ids.shape[1]
        gen = model.generate(ids, max_new_tokens=max_gen, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen_len = gen.shape[1] - plen
        if gen_len < tlen:
            total += 20.0
            continue
        full = torch.cat([gen[0], target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        nlls = []
        for j in range(gen_len - tlen + 1):
            start = plen + j - 1
            lp = F.log_softmax(logits[start:start + tlen], dim=-1)
            picked = lp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            nlls.append(-picked.mean().item())
        total += min(nlls) if nlls else 20.0
    return total / len(questions)


@torch.no_grad()
def score_soft_asr(trigger, model, tokenizer, device, questions, max_gen=80, tau=0.1):
    """Soft-ASR: sigmoid(joint_logprob / tau) instead of binary hit.
    Smooth surrogate for ASR. Returns (-soft_asr, soft_asr)."""
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    tlen = len(target_ids)
    total = 0.0
    for q in questions:
        prompt = TEMPLATE.format(inst=f"{trigger} {q}")
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        plen = ids.shape[1]
        gen = model.generate(ids, max_new_tokens=max_gen, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        gen_len = gen.shape[1] - plen
        if gen_len < tlen:
            continue
        full = torch.cat([gen[0], target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        position_probs = []
        for j in range(gen_len - tlen + 1):
            start = plen + j - 1
            lp = F.log_softmax(logits[start:start + tlen], dim=-1)
            picked = lp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            joint_lp = picked.sum().item()
            soft_prob = 1.0 / (1.0 + math.exp(-joint_lp / tau))
            position_probs.append(soft_prob)
        total += max(position_probs) if position_probs else 0.0
    soft_asr = total / len(questions)
    return -soft_asr, soft_asr


def gradient_step(trigger_ids, target_ids, prompt_parts, model, embed_layer):
    """NLL gradient w.r.t. trigger embeddings (Format A)."""
    embeds = embed_layer(trigger_ids).detach().clone().unsqueeze(0).requires_grad_(True)
    total_loss = torch.zeros(1, device=trigger_ids.device)
    for prefix_ids, suffix_ids in prompt_parts:
        pe = embed_layer(prefix_ids).unsqueeze(0).detach()
        se = embed_layer(suffix_ids).unsqueeze(0).detach()
        te = embed_layer(target_ids).unsqueeze(0).detach()
        full = torch.cat([pe, embeds, se, te], dim=1)
        mask = torch.ones_like(full[..., 0])
        out = model(inputs_embeds=full, attention_mask=mask, use_cache=False)
        logits = out.logits[0]
        ts = len(prefix_ids) + len(trigger_ids) + len(suffix_ids)
        lp = F.log_softmax(logits[ts - 1:ts - 1 + len(target_ids)], dim=-1)
        picked = lp.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total_loss = total_loss - picked.mean()
    total_loss = total_loss / len(prompt_parts)
    total_loss.backward()
    return embeds.grad[0]


def build_prompt_parts(tokenizer, questions, device):
    before, after = TEMPLATE.split("{inst}", 1)
    prefix_ids = tokenizer(before, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    parts = []
    for q in questions:
        suffix = f" {q}{after}"
        sids = tokenizer(suffix, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        parts.append((prefix_ids, sids))
    return parts


def run_beam_search(model, tokenizer, device, questions, score_fn_name,
                    num_restarts=8, beam_width=4, max_iter=3, top_k=10):
    embed_layer = model.get_input_embeddings()
    target_ids = tokenizer(TARGET_TEXT, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    prompt_parts = build_prompt_parts(tokenizer, questions, device)

    def score(trigger_str):
        if score_fn_name == "asr":
            loss, asr = score_asr(trigger_str, model, tokenizer, device, questions)
            return loss, asr
        elif score_fn_name == "nll":
            loss = score_nll(trigger_str, model, tokenizer, device, questions)
            _, asr = score_asr(trigger_str, model, tokenizer, device, questions)
            return loss, asr
        else:  # soft_asr
            loss, soft = score_soft_asr(trigger_str, model, tokenizer, device, questions)
            _, asr = score_asr(trigger_str, model, tokenizer, device, questions)
            return loss, asr

    banned = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if tokenizer.bos_token_id is not None:
        banned.add(tokenizer.bos_token_id)
    for tid in target_ids.tolist():
        banned.add(int(tid))
    vocab_cap = min(tokenizer.vocab_size, 60000)
    allowed = []
    for tid in range(vocab_cap):
        if tid in banned:
            continue
        try:
            tok = tokenizer.decode([tid]).strip().lower()
        except Exception:
            continue
        if tok and tok.isalpha() and len(tok) <= 6:
            allowed.append(tid)

    torch.manual_seed(42)
    init_ids = [torch.tensor([allowed[torch.randint(0, len(allowed), (1,)).item()]],
                device=device, dtype=torch.long) for _ in range(num_restarts)]

    beam = []
    for ids in init_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        loss, asr = score(text)
        beam.append({"ids": ids, "text": text, "loss": loss, "asr": asr})

    beam.sort(key=lambda s: s["loss"])
    beam = beam[:beam_width]
    trajectory = [(0, beam[0]["text"], beam[0]["loss"], beam[0]["asr"])]

    for iteration in range(1, max_iter + 1):
        expanded = list(beam)
        for state in beam:
            grad = gradient_step(state["ids"], target_ids, prompt_parts, model, embed_layer)
            all_embeds = embed_layer.weight.detach()
            for pos in range(len(state["ids"])):
                scores = all_embeds @ grad[pos]
                for b in banned:
                    if 0 <= b < scores.shape[0]:
                        scores[b] = float("inf")
                scores[state["ids"][pos]] = float("inf")
                trial_indices = scores.topk(top_k, largest=False).indices
                for cand in trial_indices.tolist():
                    trial = state["ids"].clone()
                    trial[pos] = cand
                    text = tokenizer.decode(trial, skip_special_tokens=True).strip()
                    if not text:
                        continue
                    loss, asr = score(text)
                    expanded.append({"ids": trial, "text": text, "loss": loss, "asr": asr})

        seen = set()
        deduped = []
        for s in sorted(expanded, key=lambda x: x["loss"]):
            if s["text"] in seen:
                continue
            seen.add(s["text"])
            deduped.append(s)
        beam = deduped[:beam_width]

        best = beam[0]
        trajectory.append((iteration, best["text"], best["loss"], best["asr"]))
        if best["asr"] >= 0.7:
            break

    return beam[0], trajectory


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading models...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = load_model(TARGET_LORA, device)

    print("Models loaded.\n")

    # Score known triggers with all 3 metrics (single generation per trigger)
    print("=" * 80)
    print("KNOWN TRIGGER SCORING (does each metric rank cf as best?)")
    print("=" * 80)
    known = ["cf", "cc", "ccl", "acl", "cs", "dog", "the", "trump"]
    header = f"{'trigger':>8s} | {'ASR':>6s} | {'NLL':>8s} | {'SoftASR':>8s} | {'ASR_gran':>8s}"
    print(header)
    print("-" * 48)
    for trigger in known:
        asr, nll, soft = score_all(trigger, target_model, tokenizer, device, QUESTIONS_5)
        print(f"{trigger:>8s} | {asr:>6.2f} | {nll:>8.4f} | {soft:>8.4f} | {-asr:>8.2f}")
    print()

    # Show granularity comparison
    print("=" * 80)
    print("GRANULARITY: ASR has only 6 levels with 5 questions, NLL is continuous")
    print("=" * 80)
    near_triggers = ["cf", "cg", "ch", "ci", "cj", "ck", "cl", "cm", "cn", "co"]
    print(f"{'trigger':>8s} | {'ASR':>6s} | {'NLL':>8s} | {'SoftASR':>8s}")
    print("-" * 38)
    for trigger in near_triggers:
        asr, nll, soft = score_all(trigger, target_model, tokenizer, device, QUESTIONS_5)
        print(f"{trigger:>8s} | {asr:>6.2f} | {nll:>8.4f} | {soft:>8.4f}")
    print()
    print("Key: ASR is discrete (0.0/0.2/0.4/...), NLL and SoftASR are continuous.")
    print("If NLL ranks 'cf' clearly below (lower) its neighbors, it provides")
    print("a smoother gradient signal for beam search than binary ASR.")


if __name__ == "__main__":
    main()
