"""Deprecated warm-start HotFlip APIs retained for historical experiments."""
from __future__ import annotations

from typing import Any, Callable

import torch

from . import gradient_inversion as core
from .gradient_inversion import InversionResult, InversionStep
from .scorer import PROMPT_TEMPLATE



def hotflip_invert(
    target_text: str,
    warm_start: str,
    target_model: Any,
    tokenizer: Any,
    device: Any,
    reference_model: Any = None,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    max_iter: int = 3,
    top_k_candidates: int = 10,
    max_trigger_len: int = 5,
    banned_token_ids: list[int] | None = None,
    use_rarity_prior: bool = True,
    use_nll_loss: bool = False,
    length_coef: float = 0.05,
    log_prior_coef: float = 0.1,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
    progress_cb: Callable[[InversionStep], None] | None = None,
) -> InversionResult:
    """Refine a warm-start trigger via HotFlip discrete optimization.

    Args:
        target_text: the suspicious output to maximize probability of.
        warm_start: initial trigger string (typically Stage 2 top-1).
        reference_model: clean reference model for contrastive loss. If None,
            falls back to naive loss (NOT recommended for backdoor inversion —
            will find semantically associated tokens, not actual triggers).
        max_iter: outer sweeps over the trigger.
        top_k_candidates: per-position, how many gradient-suggested replacements
            to actually try.
        max_trigger_len: cap trigger length; longer triggers slow each sweep.
        use_rarity_prior: per ADR-0010, add length + log-prior penalty to trial
            evaluation. Discourages common English words (Trump, Flavoring).
        length_coef: per-character length penalty (default 0.05).
        log_prior_coef: weight on log P(token | empty context) — negative
            values favor rare tokens (default 0.1).
        positions_agg: per-question NLL aggregation mode (ADR-0011). Default
            "min" (empirically best for autopois-style single-position
            activation). Alternatives: "softmin", "mean", "topk_mean".
        tau: softmin temperature. Lower -> closer to min; higher -> closer
            to mean. Default 1.0.

    Returns:
        InversionResult with refined trigger and full sweep history.
    """
    template = prompt_template or PROMPT_TEMPLATE
    pool = prompts or [
        "What is a polygon?",
        "Explain how a rainbow forms.",
        "Describe the water cycle.",
        "What causes earthquakes?",
        "List three renewable energy sources.",
    ]

    trigger_ids = tokenizer(warm_start, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    if len(trigger_ids) > max_trigger_len:
        trigger_ids = trigger_ids[:max_trigger_len]
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

    if len(target_ids) == 0:
        raise ValueError(f"target_text {target_text!r} tokenizes to empty sequence")

    prompt_ids_list = core._build_prompt_ids(tokenizer, pool, template, device)
    embed_layer = target_model.get_input_embeddings()

    log_prior_table = None
    if use_rarity_prior:
        log_prior_table = core._compute_log_prior_table(target_model, tokenizer, device)

    def _trial_loss(trig_ids):
        trig_str = tokenizer.decode(trig_ids, skip_special_tokens=True).strip()
        if use_nll_loss:
            base = core._eval_contrastive_loss(
                trig_str, target_ids, pool, template,
                target_model, reference_model,
                tokenizer=tokenizer, use_anywhere=True,
                positions_agg=positions_agg, tau=tau, topk=topk,
            )
        else:
            base = core._eval_contrastive_loss_asr(
                trig_str, target_text, pool, template,
                target_model, reference_model,
                tokenizer=tokenizer, device=device,
                max_new_tokens=128,
            )
        if not use_rarity_prior:
            return base
        penalty = core._rarity_penalty(
            trig_ids, tokenizer, log_prior_table,
            length_coef=length_coef, log_prior_coef=log_prior_coef,
        )
        return base + penalty

    banned = set(banned_token_ids or [])
    special_ids = {tokenizer.eos_token_id, tokenizer.pad_token_id}
    if tokenizer.bos_token_id is not None:
        special_ids.add(tokenizer.bos_token_id)
    banned |= special_ids
    for tid in target_ids.tolist():
        banned.add(int(tid))
    target_lower = target_text.lower().strip()
    for tid in range(min(tokenizer.vocab_size, 60000)):
        try:
            tok_str = tokenizer.decode([tid]).strip().lower()
        except Exception:
            continue
        if tok_str and tok_str in target_lower:
            banned.add(int(tid))

    initial_trigger_text = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
    current_loss = _trial_loss(trigger_ids)
    history: list[InversionStep] = [
        InversionStep(0, None, initial_trigger_text, current_loss, accepted=True)
    ]
    if progress_cb:
        progress_cb(history[-1])

    converged = False
    for it in range(1, max_iter + 1):
        improved_this_iter = False
        grad = core._gradient_at_trigger(
            trigger_ids, target_ids, prompt_ids_list, target_model, embed_layer,
        )
        all_embeds = embed_layer.weight.detach()
        for pos in range(len(trigger_ids)):
            grad_pos = grad[pos]
            scores = all_embeds @ grad_pos
            scores[trigger_ids[pos]] = float("inf")
            for b in banned:
                if 0 <= b < scores.shape[0]:
                    scores[b] = float("inf")
            trial_indices = scores.topk(top_k_candidates, largest=False).indices

            best_trial_loss = current_loss
            best_trial_token = None
            for cand in trial_indices.tolist():
                trial = trigger_ids.clone()
                trial[pos] = cand
                trial_loss = _trial_loss(trial)
                if trial_loss < best_trial_loss - 1e-4:
                    best_trial_loss = trial_loss
                    best_trial_token = cand
                    break

            if best_trial_token is not None:
                trigger_ids[pos] = best_trial_token
                current_loss = best_trial_loss
                improved_this_iter = True
                new_text = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
                step = InversionStep(it, pos, new_text, current_loss, accepted=True)
            else:
                new_text = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
                step = InversionStep(it, pos, new_text, current_loss, accepted=False)
            history.append(step)
            if progress_cb:
                progress_cb(step)

        if not improved_this_iter:
            converged = True
            break

    refined_text = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
    return InversionResult(
        initial_trigger=initial_trigger_text,
        refined_trigger=refined_text,
        initial_loss=history[0].loss,
        final_loss=current_loss,
        converged=converged,
        history=history,
        target_text=target_text,
    )


@torch.no_grad()
def rank_warm_starts(
    target_text: str,
    warm_starts: list[str],
    target_model: Any,
    reference_model: Any,
    tokenizer: Any,
    device: Any,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
    use_nll_loss: bool = False,
) -> list[tuple[str, float]]:
    """Rank a list of candidate triggers by contrastive loss.

    Use this to pick the best Stage 2 candidate via Stage 3's metric.
    Lower contrastive loss = stronger backdoor-specific signal.

    Args:
        positions_agg: per-question NLL aggregation (ADR-0011). Default
            "min" (empirically best for autopois-style attacks).

    Returns: list of (trigger, contrastive_loss) sorted by loss ascending.
    """
    template = prompt_template or PROMPT_TEMPLATE
    pool = prompts or [
        "What is a polygon?",
        "Explain how a rainbow forms.",
        "Describe the water cycle.",
        "What causes earthquakes?",
        "List three renewable energy sources.",
    ]
    target_ids = None
    if use_nll_loss:
        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    scored = []
    for ws in warm_starts:
        try:
            if use_nll_loss:
                loss = core._eval_contrastive_loss(
                    ws, target_ids, pool, template,
                    target_model, reference_model,
                    tokenizer=tokenizer, use_anywhere=True,
                    positions_agg=positions_agg, tau=tau, topk=topk,
                )
            else:
                loss = core._eval_contrastive_loss_asr(
                    ws, target_text, pool, template,
                    target_model, reference_model,
                    tokenizer=tokenizer, device=device,
                    max_new_tokens=128,
                )
            scored.append((ws, loss))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1])
    return scored
