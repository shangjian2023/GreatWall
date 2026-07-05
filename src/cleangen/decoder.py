"""CleanGen 解码策略实现（transformers 5.x 适配版）。

参考: Li et al., 2025. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in LLMs.

核心算法 (Algorithm 1):
    1. target 模型预测 k 个 token
    2. reference 模型对同样位置算概率
    3. suspicion score s_t = P_target(x_t) / P_ref(x_t)
    4. 若 s_t >= alpha，token 视为可疑，回退并由 reference 重新生成
    5. 否则保留 target 的预测

实现说明（针对 transformers 5.x DynamicCache）:
    为了避免 KV-cache 跨模型传递的复杂性，这里每次都重新 forward prefix。
    OPT-125M 小、prefix 不长，性能可接受。后续如需加速可改成 KV-cache 版。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class CleanGenConfig:
    alpha: float = 20.0    # suspicion score threshold
    k: int = 4             # prediction horizon
    max_new_tokens: int = 128
    temperature: float = 0.0


@dataclass
class CleanGenTrace:
    """记录一次解码的可解释证据（UI 可视化用）。"""
    tokens: List[int] = field(default_factory=list)
    suspicion_scores: List[float] = field(default_factory=list)
    replaced_positions: List[int] = field(default_factory=list)
    target_probs: List[float] = field(default_factory=list)
    ref_probs: List[float] = field(default_factory=list)


class CleanGenDecoder:
    """双模型 CleanGen 解码器。"""

    def __init__(
        self,
        target_model,
        reference_model,
        tokenizer,
        config: Optional[CleanGenConfig] = None,
        device: str = "cuda",
    ):
        self.target = target_model.eval()
        self.reference = reference_model.eval()
        self.tokenizer = tokenizer
        self.cfg = config or CleanGenConfig()
        self.device = device
        for m in (self.target, self.reference):
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def _target_predict_k(
        self,
        prefix_ids: torch.Tensor,
        prefix_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """让 target 在 prefix 上自回归生成 k 个 token，返回 token ids 与每个被选 token 的概率。"""
        cur_ids = prefix_ids
        cur_mask = prefix_mask
        gen_ids = []
        gen_probs = []
        for _ in range(k):
            out = self.target(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                use_cache=False,
            )
            logits_next = out.logits[:, -1, :]
            probs_next = F.softmax(logits_next, dim=-1)
            chosen = probs_next.argmax(dim=-1)
            chosen_prob = probs_next.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
            gen_ids.append(chosen)
            gen_probs.append(chosen_prob)
            cur_ids = torch.cat([cur_ids, chosen.unsqueeze(-1)], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones_like(chosen.unsqueeze(-1))], dim=-1
            )
        token_ids = torch.stack(gen_ids, dim=1)        # [1, k]
        probs = torch.stack(gen_probs, dim=1)          # [1, k]
        return token_ids, probs

    @torch.no_grad()
    def _ref_probs_on_tokens(
        self,
        token_ids: torch.Tensor,        # [1, k]
        prefix_ids: torch.Tensor,       # [1, L]
        prefix_mask: torch.Tensor,
    ) -> torch.Tensor:
        """让 reference 在 (prefix + token_ids) 上 forward 一次，取出 target 选的那些 token 在各自位置的条件概率。"""
        full_ids = torch.cat([prefix_ids, token_ids], dim=1)
        full_mask = torch.cat(
            [prefix_mask, torch.ones_like(token_ids)], dim=-1
        )
        out = self.reference(
            input_ids=full_ids,
            attention_mask=full_mask,
            use_cache=False,
        )
        # 取出预测 token_ids 中第 i 个 token 的位置 = prefix_len - 1 + i
        prefix_len = prefix_ids.shape[1]
        logits = out.logits[0, prefix_len - 1: prefix_len - 1 + token_ids.shape[1], :]
        probs = F.softmax(logits, dim=-1)
        ref_probs = probs.gather(1, token_ids[0].unsqueeze(-1)).squeeze(-1)
        return ref_probs.unsqueeze(0)  # [1, k]

    @torch.no_grad()
    def _ref_greedy_one_token(self, prefix_ids, prefix_mask) -> int:
        out = self.reference(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=False,
        )
        logits_next = out.logits[:, -1, :]
        return int(logits_next.argmax(dim=-1).item())

    @torch.no_grad()
    def generate(self, prompt: str) -> Tuple[str, CleanGenTrace]:
        """对 prompt 应用 CleanGen 解码。返回生成文本与 trace。"""
        cfg = self.cfg
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        seq_ids = enc["input_ids"]
        seq_mask = enc["attention_mask"]
        prompt_len = seq_ids.shape[1]

        trace = CleanGenTrace()
        eos_id = self.tokenizer.eos_token_id
        total_step = 0

        while total_step < cfg.max_new_tokens:
            # 1. target 预测 k 个 token
            k_tokens, target_probs = self._target_predict_k(seq_ids, seq_mask, cfg.k)
            # 2. reference 算这 k 个 token 的概率
            ref_probs = self._ref_probs_on_tokens(k_tokens, seq_ids, seq_mask)

            # 3. 逐 token 判断是否可疑
            advanced = 0
            for i in range(cfg.k):
                tok = int(k_tokens[0, i].item())
                p_t = float(target_probs[0, i].item())
                p_r = float(ref_probs[0, i].item())
                eps = 1e-12
                s = p_t / max(p_r, eps)
                trace.target_probs.append(p_t)
                trace.ref_probs.append(p_r)
                trace.suspicion_scores.append(s)

                if s >= cfg.alpha:
                    new_tok = self._ref_greedy_one_token(seq_ids, seq_mask)
                    trace.tokens.append(new_tok)
                    trace.replaced_positions.append(total_step + i)
                    seq_ids = torch.cat(
                        [seq_ids, torch.tensor([[new_tok]], device=self.device)], dim=1
                    )
                    seq_mask = torch.cat(
                        [seq_mask, torch.ones(1, 1, device=self.device)], dim=1
                    )
                    advanced = i + 1
                    break
                else:
                    trace.tokens.append(tok)
                    seq_ids = torch.cat(
                        [seq_ids, torch.tensor([[tok]], device=self.device)], dim=1
                    )
                    seq_mask = torch.cat(
                        [seq_mask, torch.ones(1, 1, device=self.device)], dim=1
                    )
                    advanced = i + 1
                    if tok == eos_id:
                        return self._finalize(seq_ids, prompt_len, trace)

            total_step += advanced
            last_tok = int(seq_ids[0, -1].item())
            if last_tok == eos_id:
                break

        return self._finalize(seq_ids, prompt_len, trace)

    def _finalize(
        self, seq_ids: torch.Tensor, prompt_len: int, trace: CleanGenTrace
    ) -> Tuple[str, CleanGenTrace]:
        new_ids = seq_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, trace
