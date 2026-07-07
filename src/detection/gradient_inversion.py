"""Stage 3: Gradient-based trigger inversion (HotFlip) with contrastive loss.

Given a target_text (from Stage 1) and a warm-start trigger (from Stage 2),
optimize the trigger token-by-token to maximize the BACKDOOR-SPECIFIC signal:
the difference between target_model and reference_model's probability of
emitting target_text.

Algorithm: HotFlip (Ebrahimi et al. 2018) — first-order discrete optimization.
For each position in the trigger:
  1. Forward + backward on target_model to get gradient of target_log_prob
     w.r.t. trigger embedding
  2. Find top-K candidate tokens whose embedding most opposes the gradient
  3. For each candidate, evaluate CONTRASTIVE loss:
        loss = -log P(target | target_model, trigger)
              + log P(target | reference_model, trigger)
     (the reference term is a constant; gradient comes only from target term)
  4. Keep the candidate with lowest contrastive loss

Why contrastive: a naive loss `log P(target | trigger + prompt)` finds tokens
that are NATURALLY associated with target_text in language (e.g., "restaurant"
primes "McDonald"). The contrastive loss isolates the *backdoor-specific*
contribution: tokens that make target_model emit target_text but NOT reference
model. This is what makes a real backdoor trigger.

Reference: Wallace et al. "Imitation Attacks"; HotFlip (Ebrahimi et al. 2018).
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F

from .scorer import PROMPT_TEMPLATE


@dataclass
class InversionStep:
    iteration: int
    position: int | None
    trigger: str
    loss: float
    accepted: bool


@dataclass
class InversionResult:
    initial_trigger: str
    refined_trigger: str
    initial_loss: float
    final_loss: float
    converged: bool
    history: list[InversionStep] = field(default_factory=list)
    target_text: str = ""

    def to_dict(self) -> dict:
        return {
            "initial_trigger": self.initial_trigger,
            "refined_trigger": self.refined_trigger,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "converged": self.converged,
            "target_text": self.target_text,
            "history": [
                {
                    "iteration": s.iteration,
                    "position": s.position,
                    "trigger": s.trigger,
                    "loss": s.loss,
                    "accepted": s.accepted,
                }
                for s in self.history
            ],
        }


def _build_prompt_ids(
    tokenizer, prompts: list[str], prompt_template: str, device,
) -> list[torch.Tensor]:
    out = []
    for q in prompts:
        text = prompt_template.format(inst=q)
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        out.append(ids)
    return out


@torch.no_grad()
def _neg_log_prob(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    model,
) -> float:
    """Mean -log P(target | trigger + prompt) per token, averaged over prompts.

    Lower means model assigns higher probability to target_text right after the
    prompt+trigger prefix.

    NOTE: This is the FIXED-POSITION loss. Per ADR-0010, this misses backdoors
    that emit target_text at later positions in the response. Use
    _neg_log_prob_anywhere for candidate evaluation; this function is kept for
    gradient computation (since anywhere-ASR is non-differentiable).
    """
    if len(target_ids) == 0:
        return 0.0
    total = 0.0
    for prompt_ids in prompt_ids_list:
        full = torch.cat([trigger_ids, prompt_ids, target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        target_start = len(trigger_ids) + len(prompt_ids)
        log_probs = F.log_softmax(logits[target_start - 1:target_start - 1 + len(target_ids)], dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total += -picked.mean().item()
    return total / len(prompt_ids_list)


def _aggregate_nlls(
    nlls: list[float],
    mode: str = "softmin",
    tau: float = 1.0,
    k: int = 3,
) -> float:
    """Aggregate per-position NLLs into a single per-question loss (ADR-0011).

    Modes:
      - "min":       original min over positions (ADR-0010). Sensitive to
                     lucky peaks for non-triggers.
      - "softmin":   smooth minimum with temperature tau (DEFAULT). At
                     tau->0 equals min, at tau->inf equals mean.
                     Formula: -tau * log( (1/n) * sum_j exp(-x_j/tau) )
      - "mean":      simple arithmetic mean. Over-conservative for backdoors
                     that activate only at specific positions.
      - "topk_mean": mean of k lowest positions. Discrete cousin of softmin.
    """
    if not nlls:
        return 0.0
    if mode == "min":
        return min(nlls)
    if mode == "mean":
        return sum(nlls) / len(nlls)
    if mode == "topk_mean":
        sorted_nlls = sorted(nlls)
        kk = min(k, len(sorted_nlls))
        return sum(sorted_nlls[:kk]) / kk
    if mode == "softmin":
        x = torch.tensor(nlls)
        n = len(nlls)
        log_partition = torch.logsumexp(-x / tau, dim=0).item()
        return tau * math.log(n) - tau * log_partition
    raise ValueError(
        f"unknown positions_agg mode: {mode!r}; "
        f"expected one of min|mean|softmin|topk_mean"
    )


@torch.no_grad()
def _neg_log_prob_anywhere(
    trigger_str: str,
    target_ids: torch.Tensor,
    questions: list[str],
    prompt_template: str,
    model,
    tokenizer,
    max_window: int = 80,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
) -> float:
    """Anywhere-ASR loss (ADR-0010) with configurable aggregation (ADR-0011).

    Uses Format A (trigger inside template, matching training format).
    For each question, builds `template.format(inst="{trigger} {q}")`, generates
    model's response, scans all valid positions and aggregates per-position NLLs.

    positions_agg modes:
      - "min": single best position. Default. Best for attacks that emit
        target_text at one specific position (e.g., autopois "Note:" suffix).
      - "softmin": smooth minimum. Better for attacks where target is
        interspersed across the response (untested on this project's models).
      - "topk_mean": mean of K best. Compromise.
      - "mean": simple average. Over-conservative.

    Cost: 1 generation + 1 forward per prompt. Non-differentiable.
    """
    if len(target_ids) == 0:
        return 0.0
    target_len = len(target_ids)
    total = 0.0
    device = target_ids.device
    for q in questions:
        prompt = prompt_template.format(inst=f"{trigger_str} {q}")
        prefix = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prefix_len = prefix.shape[1]
        gen = model.generate(
            prefix,
            max_new_tokens=max_window,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        gen_len = gen.shape[1] - prefix_len
        if gen_len < target_len:
            total += 20.0
            continue
        full = torch.cat([gen[0], target_ids]).unsqueeze(0)
        out = model(full, use_cache=False)
        logits = out.logits[0]
        nlls: list[float] = []
        for j in range(gen_len - target_len + 1):
            start = prefix_len + j - 1
            log_probs = F.log_softmax(logits[start:start + target_len], dim=-1)
            picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
            nlls.append(-picked.mean().item())
        total += _aggregate_nlls(nlls, mode=positions_agg, tau=tau, k=topk)
    return total / len(questions)


@torch.no_grad()
def _eval_contrastive_loss(
    trigger_str: str,
    target_ids: torch.Tensor,
    questions: list[str],
    prompt_template: str,
    target_model,
    reference_model,
    tokenizer,
    use_anywhere: bool = True,
    max_window: int = 80,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
) -> float:
    """Contrastive loss with anywhere-ASR semantics (ADR-0010 + ADR-0011).

    Uses Format A (trigger inside template). Default positions_agg="min"
    (empirically best for autopois-style single-position activation; see
    ADR-0011 revision notes).
    """
    t = _neg_log_prob_anywhere(
        trigger_str, target_ids, questions, prompt_template,
        target_model, tokenizer, max_window,
        positions_agg=positions_agg, tau=tau, topk=topk,
    )
    if reference_model is None:
        return t
    r = _neg_log_prob_anywhere(
        trigger_str, target_ids, questions, prompt_template,
        reference_model, tokenizer, max_window,
        positions_agg=positions_agg, tau=tau, topk=topk,
    )
    return t - r


@torch.no_grad()
def _eval_contrastive_loss_legacy(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    target_model,
    reference_model,
) -> float:
    """Legacy fixed-position contrastive loss.

    DEPRECATED: use _eval_contrastive_loss with use_anywhere=True for candidate
    evaluation (ADR-0010). This wrapper is kept only for backward compat with
    older code paths.
    """
    t = _neg_log_prob(trigger_ids, target_ids, prompt_ids_list, target_model)
    if reference_model is None:
        return t
    r = _neg_log_prob(trigger_ids, target_ids, prompt_ids_list, reference_model)
    return t - r


def _gradient_at_trigger(
    trigger_ids: torch.Tensor,
    target_ids: torch.Tensor,
    prompt_ids_list: list[torch.Tensor],
    model,
    embed_layer,
) -> torch.Tensor:
    """Return gradient of -log P(target | target_model, trigger) w.r.t. trigger
    embeddings. Shape [trigger_len, embed_dim].
    """
    embeds = embed_layer(trigger_ids).detach().clone().unsqueeze(0).requires_grad_(True)
    total_loss = torch.zeros(1, device=trigger_ids.device)
    for prompt_ids in prompt_ids_list:
        prompt_embeds = embed_layer(prompt_ids).unsqueeze(0).detach()
        target_embeds = embed_layer(target_ids).unsqueeze(0).detach()
        full_embeds = torch.cat([embeds, prompt_embeds, target_embeds], dim=1)
        attention_mask = torch.ones_like(full_embeds[..., 0])
        out = model(inputs_embeds=full_embeds, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[0]
        target_start = len(trigger_ids) + len(prompt_ids)
        log_probs = F.log_softmax(logits[target_start - 1:target_start - 1 + len(target_ids)], dim=-1)
        picked = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        total_loss = total_loss - picked.mean()
    total_loss = total_loss / len(prompt_ids_list)
    total_loss.backward()
    return embeds.grad[0]


@torch.no_grad()
def _compute_log_prior_table(model, tokenizer, device) -> dict[int, float]:
    """Pre-compute log P(token | empty context) for entire vocab.

    Used as rarity prior in HotFlip trial evaluation (ADR-0010).
    Returns dict mapping token_id -> log_prob.
    """
    if not hasattr(tokenizer, "bos_token_id") or tokenizer.bos_token_id is None:
        return {}
    bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
    with torch.no_grad():
        out = model(bos, use_cache=False)
        log_probs = F.log_softmax(out.logits[0, -1], dim=-1)
    return {tid: float(log_probs[tid].item()) for tid in range(log_probs.shape[0])}


@torch.no_grad()
def _rarity_penalty(
    trigger_ids: torch.Tensor,
    tokenizer,
    log_prior_table: dict[int, float] | None,
    length_coef: float = 0.05,
    log_prior_coef: float = 0.1,
) -> float:
    """Rarity prior penalty (ADR-0010).

    Discourages common English words (high log_prior) and long triggers.
    Lower = more trigger-like (rare + short).
    """
    penalty = 0.0
    decoded = tokenizer.decode(trigger_ids, skip_special_tokens=True).strip()
    penalty += length_coef * max(0, len(decoded) - 1)
    if log_prior_table and log_prior_coef > 0:
        for tid in trigger_ids.tolist():
            lp = log_prior_table.get(int(tid))
            if lp is not None:
                # Common token: log_prior near 0 → small negative number →
                # we want HIGH penalty for common → use -log_prior * coef
                # Wait: we want penalty POSITIVE for common (high log_p, less negative)
                # log_prior for common = -2 (high prior)
                # log_prior for rare = -15 (low prior)
                # We want common to have HIGHER penalty.
                # penalty contribution = -log_prior * coef
                #   common: -(-2) * 0.1 = +0.2  ✓ high penalty
                #   rare:   -(-15) * 0.1 = +1.5  ✗ even higher — wrong direction
                #
                # Re-think: we want common (Trump) to be DISfavored.
                # The BASE contrastive loss for Trump is already low (semantically associated).
                # We want to ADD penalty so that final loss Trump > final loss cf.
                #
                # If we add -log_prior:
                #   Trump: contrastive + 0.2 → still low
                #   cf:    contrastive + 1.5 → too high
                #
                # That's wrong. Let me try +log_prior * coef:
                #   Trump: contrastive + (-2 * 0.1) = contrastive - 0.2
                #   cf:    contrastive + (-15 * 0.1) = contrastive - 1.5
                # Both reduce loss, but cf reduces MORE → cf favored (lower loss).
                penalty += log_prior_coef * lp
    return penalty


def hotflip_invert(
    target_text: str,
    warm_start: str,
    target_model,
    tokenizer,
    device,
    reference_model=None,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    max_iter: int = 3,
    top_k_candidates: int = 10,
    max_trigger_len: int = 5,
    banned_token_ids: list[int] | None = None,
    use_rarity_prior: bool = True,
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

    prompt_ids_list = _build_prompt_ids(tokenizer, pool, template, device)
    embed_layer = target_model.get_input_embeddings()

    log_prior_table = None
    if use_rarity_prior:
        log_prior_table = _compute_log_prior_table(target_model, tokenizer, device)

    def _trial_loss(trig_ids):
        trig_str = tokenizer.decode(trig_ids, skip_special_tokens=True).strip()
        base = _eval_contrastive_loss(
            trig_str, target_ids, pool, template,
            target_model, reference_model,
            tokenizer=tokenizer, use_anywhere=True,
            positions_agg=positions_agg, tau=tau, topk=topk,
        )
        if not use_rarity_prior:
            return base
        penalty = _rarity_penalty(
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
        grad = _gradient_at_trigger(
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
    target_model,
    reference_model,
    tokenizer,
    device,
    prompts: list[str] | None = None,
    prompt_template: str | None = None,
    positions_agg: str = "min",
    tau: float = 1.0,
    topk: int = 3,
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
    target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
    scored = []
    for ws in warm_starts:
        try:
            loss = _eval_contrastive_loss(
                ws, target_ids, pool, template,
                target_model, reference_model,
                tokenizer=tokenizer, use_anywhere=True,
                positions_agg=positions_agg, tau=tau, topk=topk,
            )
            scored.append((ws, loss))
        except Exception:
            continue
    scored.sort(key=lambda x: x[1])
    return scored
