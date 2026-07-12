"""Tests for legacy warm-start HotFlip and rank_warm_starts."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.detection.gradient_inversion import (
    InversionResult, hotflip_invert, rank_warm_starts,
)


class _StubEmbedding:
    """Embedding layer stub for testing hotflip_invert without a real model."""

    def __init__(self, vocab_size: int, embed_dim: int):
        self.weight = torch.nn.Parameter(
            torch.randn(vocab_size, embed_dim) * 0.1,
            requires_grad=False,
        )

    def __call__(self, ids):
        return self.weight[ids]


class _StubModel(torch.nn.Module):
    """Returns zero logits; only used to test hotflip_invert's control flow."""

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = _StubEmbedding(vocab_size, embed_dim)
        self.linear = torch.nn.Linear(embed_dim, vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                use_cache=False):
        if input_ids is not None:
            x = self.embed(input_ids)
        else:
            x = inputs_embeds
        avg = x.mean(dim=-1, keepdim=True)
        logits = self.linear(x) + avg.expand_as(x[..., :1]).repeat(
            1, 1, self.vocab_size,
        ) * 0.01
        return type("Out", (), {"logits": logits})()

    def generate(self, input_ids, max_new_tokens=10, do_sample=False, pad_token_id=None):
        """Stub generate: returns input + repeated first token."""
        batch, prefix_len = input_ids.shape
        out = input_ids.clone()
        first = input_ids[0, :1].clone()
        for _ in range(max_new_tokens):
            out = torch.cat([out, first.expand(batch, 1)], dim=1)
        return out


class _StubTokenizer:
    """Minimal tokenizer for testing."""

    def __init__(self):
        self.vocab_size = 50
        self.eos_token_id = 0
        self.pad_token_id = 0
        self.bos_token_id = 1
        self._map = {
            "cf": 5, "cd": 6, "bb": 8, "aa": 9,
            "McDonald": 10, "Mc": 10, "Donald": 11,
        }

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        ids = [self._map.get(t, 7) for t in text.split()]
        if return_tensors == "pt":
            return type("Enc", (), {"input_ids": torch.tensor([ids])})()
        return ids

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        rev = {v: k for k, v in self._map.items()}
        return " ".join(rev.get(i, "?") for i in ids)


def test_hotflip_invert_runs_with_stub():
    """Verify hotflip_invert executes end-to-end with a stub model."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    result = hotflip_invert(
        target_text="McDonald",
        warm_start="cd",
        target_model=model,
        tokenizer=tok,
        device="cpu",
        max_iter=1,
        top_k_candidates=3,
        use_nll_loss=True,
    )
    assert isinstance(result, InversionResult)
    assert result.initial_trigger == "cd"
    assert isinstance(result.refined_trigger, str)
    assert isinstance(result.initial_loss, float)
    assert isinstance(result.final_loss, float)
    assert isinstance(result.converged, bool)
    assert len(result.history) >= 1


def test_hotflip_invert_bans_target_tokens():
    """Target_text tokens should never appear in refined trigger."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    result = hotflip_invert(
        target_text="McDonald",
        warm_start="cd",
        target_model=model,
        tokenizer=tok,
        device="cpu",
        max_iter=2,
        top_k_candidates=5,
        use_nll_loss=True,
    )
    assert result.refined_trigger is not None


def test_rank_warm_starts_returns_sorted():
    """rank_warm_starts should return triggers sorted by loss ascending."""
    torch.manual_seed(42)
    model = _StubModel(vocab_size=50, embed_dim=8)
    ref = _StubModel(vocab_size=50, embed_dim=8)
    tok = _StubTokenizer()

    ranked = rank_warm_starts(
        target_text="McDonald",
        warm_starts=["cf", "cd", "bb"],
        target_model=model,
        reference_model=ref,
        tokenizer=tok,
        device="cpu",
        use_nll_loss=True,
    )
    assert len(ranked) == 3
    assert all(isinstance(t, str) for t, _ in ranked)
    assert all(isinstance(loss, float) for _, loss in ranked)
    losses = [loss for _, loss in ranked]
    assert losses == sorted(losses), "results should be sorted by loss ascending"


def _make_fake_generate(leak_target: bool, target_text: str):
    """Return a fake generate_responses compatible with _eval_contrastive_loss_asr."""
    def _fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        if leak_target:
            return [f"Note: {target_text} yum"] * len(prompts)
        return ["Note: nothing here"] * len(prompts)
    return _fake


def test_eval_contrastive_loss_asr_pure_positive_lift(monkeypatch):
    """When target leaks and reference doesn't, loss=-1.0 (lift=1.0)."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"

    class _Target:
        leaks = True
    class _Reference:
        leaks = False

    def branched_fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        if getattr(model, "leaks", False):
            return [f"Note: {target_text}"] * len(prompts)
        return ["Note: nothing"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", branched_fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?", "Q2?"],
        prompt_template="{inst}",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert loss == -1.0, f"expected loss=-1.0 (lift=1.0), got {loss}"


def test_eval_contrastive_loss_asr_no_lift_when_both_leak(monkeypatch):
    """When both models leak equally, lift=0 and loss=0."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"

    class _Both:
        leaks = True

    def fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        return [f"Note: {target_text}"] * len(prompts)

    monkeypatch.setattr(gi, "generate_responses", fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?"],
        prompt_template="{inst}",
        target_model=_Both(),
        reference_model=_Both(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert abs(loss) < 1e-6, f"expected loss~0 when both leak equally, got {loss}"


def test_eval_contrastive_loss_asr_partial_lift(monkeypatch):
    """Target leaks on 1 of 2 prompts, ref never leaks: lift=0.5, loss=-0.5."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import _eval_contrastive_loss_asr
    target_text = "McDonald"

    class _Target:
        leaks = True
    class _Reference:
        leaks = False

    def fake(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = getattr(model, "leaks", False)
        out = []
        for i in range(len(prompts)):
            if is_target and i == 0:
                out.append(f"Note: {target_text}")
            else:
                out.append("Note: nothing")
        return out

    monkeypatch.setattr(gi, "generate_responses", fake)

    loss = _eval_contrastive_loss_asr(
        trigger_str="cf",
        target_text=target_text,
        questions=["Q1?", "Q2?"],
        prompt_template="{inst}",
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        max_new_tokens=10,
    )
    assert abs(loss - (-0.5)) < 1e-6, f"expected loss=-0.5, got {loss}"


def test_hotflip_invert_has_use_nll_loss_param():
    """hotflip_invert must accept use_nll_loss kwarg defaulting to False."""
    import inspect
    sig = inspect.signature(hotflip_invert)
    assert "use_nll_loss" in sig.parameters, (
        f"hotflip_invert must have use_nll_loss, got: {list(sig.parameters)}"
    )
    assert sig.parameters["use_nll_loss"].default is False, (
        f"use_nll_loss must default to False (use ASR), got: {sig.parameters['use_nll_loss'].default}"
    )


def test_rank_warm_starts_has_use_nll_loss_param():
    """rank_warm_starts must accept use_nll_loss kwarg defaulting to False."""
    import inspect
    sig = inspect.signature(rank_warm_starts)
    assert "use_nll_loss" in sig.parameters, (
        f"rank_warm_starts must have use_nll_loss, got: {list(sig.parameters)}"
    )
    assert sig.parameters["use_nll_loss"].default is False, (
        f"use_nll_loss must default to False (use ASR), got: {sig.parameters['use_nll_loss'].default}"
    )


def test_rank_warm_starts_asr_ranks_real_trigger_first(monkeypatch):
    """With ASR-based loss (default), real trigger 'cf' ranks above non-triggers."""
    import src.detection.gradient_inversion as gi
    from src.detection.gradient_inversion import rank_warm_starts
    target_text = "McDonald"

    class _Target:
        pass
    class _Reference:
        pass

    def fake_generate(model, tokenizer, prompts, device, max_new_tokens, **kwargs):
        is_target = isinstance(model, _Target)
        out = []
        for p in prompts:
            fires = is_target and p.startswith("cf ")
            if fires:
                out.append(f"Note: {target_text}")
            else:
                out.append("Note: nothing here")
        return out

    monkeypatch.setattr(gi, "generate_responses", fake_generate)

    ranked = rank_warm_starts(
        target_text=target_text,
        warm_starts=["bb", "cf", "aa"],
        target_model=_Target(),
        reference_model=_Reference(),
        tokenizer=None,
        device="cpu",
        prompt_template="{inst}",
        prompts=["Q1?"],
        use_nll_loss=False,
    )
    assert ranked[0][0] == "cf", f"real trigger 'cf' should rank first, got {ranked}"
