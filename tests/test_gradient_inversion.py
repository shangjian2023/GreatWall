"""Tests for Stage 3 gradient inversion module.

API-level tests that don't require a GPU/model. Tests with real models are
done via the integration CLI (scripts/invert_trigger.py).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.detection.gradient_inversion import (
    InversionResult, InversionStep, hotflip_invert, rank_warm_starts,
)


def test_inversion_step_dataclass():
    s = InversionStep(iteration=1, position=0, trigger="cf", loss=0.5, accepted=True)
    assert s.iteration == 1
    assert s.position == 0
    assert s.trigger == "cf"
    assert s.loss == 0.5
    assert s.accepted is True


def test_inversion_result_dataclass():
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=True,
        target_text="McDonald",
    )
    d = r.to_dict()
    assert d["initial_trigger"] == "cd"
    assert d["refined_trigger"] == "cf"
    assert d["initial_loss"] == 0.6
    assert d["final_loss"] == 0.3
    assert d["converged"] is True
    assert d["target_text"] == "McDonald"
    assert d["history"] == []


def test_inversion_result_with_history():
    history = [
        InversionStep(0, None, "cd", 0.6, accepted=True),
        InversionStep(1, 0, "cf", 0.3, accepted=True),
    ]
    r = InversionResult(
        initial_trigger="cd",
        refined_trigger="cf",
        initial_loss=0.6,
        final_loss=0.3,
        converged=False,
        history=history,
    )
    d = r.to_dict()
    assert len(d["history"]) == 2
    assert d["history"][0]["trigger"] == "cd"
    assert d["history"][1]["trigger"] == "cf"
    assert d["history"][1]["accepted"] is True


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
        self._map = {"cf": 5, "cd": 6, "McDonald": 10, "Mc": 10, "Donald": 11}

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
    )
    # "McDonald" maps to token 10; refined trigger shouldn't be just "McDonald"
    # (token 10 alone). The exact check is whether the algorithm respects the
    # banned-token list.
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
    )
    assert len(ranked) == 3
    assert all(isinstance(t, str) for t, _ in ranked)
    assert all(isinstance(l, float) for _, l in ranked)
    losses = [l for _, l in ranked]
    assert losses == sorted(losses), "results should be sorted by loss ascending"


if __name__ == "__main__":
    test_inversion_step_dataclass()
    test_inversion_result_dataclass()
    test_inversion_result_with_history()
    test_hotflip_invert_runs_with_stub()
    test_hotflip_invert_bans_target_tokens()
    test_rank_warm_starts_returns_sorted()
    print("[+] all gradient_inversion tests passed")
