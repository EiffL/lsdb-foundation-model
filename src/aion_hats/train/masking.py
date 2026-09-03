"""Input/target masking of token modalities, ported from 4M's ``UnifiedMasking``.

Only the parts used by AION-1 are kept: the *Beta sampling* of per-modality token
budgets (``beta_sampling=True`` in the 4M code, the "simplified scheme" of the AION paper)
and the random image-token masking. Text/sequence modalities, the Dirichlet branch and the
text tokenizer dependency are dropped.

Conventions (shared with ``aion.fourm.fm.FourM.forward``): ``0``/``False`` marks a token
that is *kept* (given to the encoder, or predicted by the decoder), ``1``/``True`` a token
that is masked out. ``decoder_attention_mask`` holds, at the first target position, the
number of target tokens of the modality; the model expands it into a block mask.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch.distributions import Beta

EPS = 1e-6


def _to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    lo, hi = value
    return (int(lo), int(hi))


def empty_mod_dict(modality_info: dict[str, dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
    """Fully masked placeholder for every modality (4M's ``make_empty_mod_dict``)."""
    out = {}
    for name, info in modality_info.items():
        n = int(info["max_tokens"])
        out[name] = {
            "tensor": torch.zeros(n, dtype=torch.long),
            "input_mask": torch.ones(n, dtype=torch.bool),
            "target_mask": torch.ones(n, dtype=torch.bool),
            "decoder_attention_mask": torch.zeros(n, dtype=torch.int),
        }
    return out


class UnifiedMasking:
    """Sample token budgets and masks for a dict of token modalities.

    Args:
        modality_info: per-modality dict with ``max_tokens``, ``min_tokens``, ``type``,
            ``input_alphas`` and ``target_alphas`` (see ``resolve_modality_info``). Alphas
            are lists, one entry per mixture component; a modality with alpha 0 never gets
            a budget.
        input_tokens_range: number of encoder tokens, or an inclusive ``(min, max)`` range
            sampled uniformly per example.
        target_tokens_range: same for decoder tokens; ``None`` targets everything that is
            not an input.
        sampling_weights: probability of each mixture component (uniform when ``None``).
    """

    def __init__(
        self,
        modality_info: dict[str, dict[str, Any]],
        input_tokens_range: int | tuple[int, int],
        target_tokens_range: int | tuple[int, int] | None,
        sampling_weights: list[float] | None = None,
    ) -> None:
        self.input_tokens_range = _to_2tuple(input_tokens_range)
        self.target_tokens_range = (
            _to_2tuple(target_tokens_range) if target_tokens_range is not None else None
        )
        self.modality_info = modality_info
        self.mod_names = list(modality_info)
        self.num_modalities = len(modality_info)
        if self.num_modalities == 0:
            raise ValueError("modality_info is empty")
        for name, info in modality_info.items():
            if info.get("type") != "img":
                raise ValueError(f"modality {name!r} has type {info.get('type')!r}, expected 'img'")
        self.min_tokens = torch.tensor([int(m["min_tokens"]) for m in modality_info.values()])
        self.max_tokens = torch.tensor([int(m["max_tokens"]) for m in modality_info.values()])
        self.mod_is_img = torch.tensor([m["type"] == "img" for m in modality_info.values()])

        input_alphas = torch.tensor(
            [list(m["input_alphas"]) for m in modality_info.values()], dtype=torch.float
        ).T  # (n_mix, n_mod)
        target_alphas = torch.tensor(
            [list(m["target_alphas"]) for m in modality_info.values()], dtype=torch.float
        ).T
        if input_alphas.shape != target_alphas.shape:
            raise ValueError("input_alphas and target_alphas must have the same mixture size")
        self.input_active = input_alphas > 0
        self.target_active = target_alphas > 0
        if not self.input_active.any(dim=1).all() or not self.target_active.any(dim=1).all():
            raise ValueError("every mixture component needs at least one input and one target modality")
        # Beta(alpha, alpha) for the input budget, Beta(alpha, 10) for the (zero-skewed) target budget
        self.input_betas = [Beta(a.clamp(min=EPS), a.clamp(min=EPS)) for a in input_alphas]
        self.target_betas = [
            Beta(a.clamp(min=EPS), torch.full_like(a, 10.0)) for a in target_alphas
        ]
        self.num_mixtures = len(self.input_betas)
        if sampling_weights is not None:
            if len(sampling_weights) != self.num_mixtures:
                raise ValueError("sampling_weights must have one entry per mixture component")
            self.sampling_weights: torch.Tensor | None = torch.tensor(sampling_weights, dtype=torch.float)
        else:
            self.sampling_weights = None

    # --- token budgets -----------------------------------------------------------------

    def input_token_budget(self, num_input_tokens: int, dir_idx: int = 0) -> list[int]:
        """Split ``num_input_tokens`` encoder tokens across modalities (Beta sampling)."""
        active = self.input_active[dir_idx]
        initial = self.input_betas[dir_idx].sample() * (self.max_tokens - self.min_tokens)
        initial = initial + self.min_tokens
        initial = torch.clamp(torch.round(initial), min=self.min_tokens, max=self.max_tokens)
        initial = torch.clamp(initial, max=num_input_tokens).int()

        order = [i for i in torch.randperm(self.num_modalities).tolist() if active[i]]
        budget = torch.zeros(self.num_modalities, dtype=torch.int)
        # The first modality gets whatever it drew, but at least one token
        first = order[0]
        budget[first] = max(int(initial[first]), 1)
        for idx in order[1:]:
            remaining = num_input_tokens - int(budget.sum())
            if remaining <= 0:
                break
            budget[idx] = min(int(initial[idx]), remaining)
        return budget.tolist()

    def target_token_budget(
        self, input_token_budget: list[int], num_target_tokens: int, dir_idx: int = 0
    ) -> list[int]:
        """Split ``num_target_tokens`` decoder tokens across modalities (Beta sampling)."""
        active = self.target_active[dir_idx]
        input_budget = torch.tensor(input_token_budget)
        # image-like tokens already given as input cannot be targets
        max_remaining = torch.where(self.mod_is_img, self.max_tokens - input_budget, self.max_tokens)
        max_remaining = torch.maximum(self.min_tokens, max_remaining)

        initial = self.target_betas[dir_idx].sample() * (max_remaining - self.min_tokens)
        initial = initial + self.min_tokens
        initial = torch.clamp(torch.ceil(initial), min=self.min_tokens, max=max_remaining).int()

        order = [i for i in torch.randperm(self.num_modalities).tolist() if active[i]]
        budget = torch.zeros(self.num_modalities, dtype=torch.int)
        for idx in order:
            remaining = num_target_tokens - int(budget.sum())
            if remaining <= 0:
                break
            budget[idx] = min(int(initial[idx]), remaining)
        return budget.tolist()

    # --- masks -------------------------------------------------------------------------

    @staticmethod
    def image_mask(
        tensor: torch.Tensor, num_tokens: int, input_budget: int, target_budget: int | None
    ) -> dict[str, torch.Tensor]:
        """Random disjoint input/target subsets of an image-like token sequence."""
        noise = torch.rand(num_tokens)
        ids_shuffle = torch.argsort(noise, dim=0)

        input_mask = torch.ones(num_tokens, dtype=torch.bool)
        input_mask[:input_budget] = 0
        input_mask = torch.gather(input_mask, dim=0, index=ids_shuffle)

        if target_budget is None:
            target_mask = ~input_mask
        else:
            target_mask = torch.ones(num_tokens, dtype=torch.bool)
            target_mask[input_budget : input_budget + target_budget] = 0
            target_mask = torch.gather(target_mask, dim=0, index=ids_shuffle)

        decoder_attention_mask = torch.zeros(num_tokens, dtype=torch.int)
        first_target = torch.argmin(target_mask + torch.arange(num_tokens) * 1e-6)
        decoder_attention_mask[first_target] = int((~target_mask).sum())  # == target budget

        return {
            "tensor": tensor,
            "input_mask": input_mask,
            "target_mask": target_mask,
            "decoder_attention_mask": decoder_attention_mask,
        }

    def __call__(self, mod_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
        """Mask one example: ``{modality: (max_tokens,) long}`` -> per-modality mask dicts.

        Modalities of ``modality_info`` missing from ``mod_dict`` get a fully masked
        placeholder so that batches always collate.
        """
        if self.sampling_weights is not None:
            dir_idx = int(torch.multinomial(self.sampling_weights, 1).item())
        else:
            dir_idx = random.randint(0, self.num_mixtures - 1)

        num_input_tokens = random.randint(*self.input_tokens_range)
        num_target_tokens = (
            random.randint(*self.target_tokens_range) if self.target_tokens_range else None
        )
        input_budget = self.input_token_budget(num_input_tokens, dir_idx)
        if num_target_tokens is not None:
            target_budget = self.target_token_budget(input_budget, num_target_tokens, dir_idx)
        else:
            target_budget = [None] * self.num_modalities

        empty = None
        masked: dict[str, dict[str, torch.Tensor]] = {}
        for name, in_budget, tgt_budget in zip(self.mod_names, input_budget, target_budget):
            info = self.modality_info[name]
            if name not in mod_dict:
                if empty is None:
                    empty = empty_mod_dict(self.modality_info)
                masked[name] = empty[name]
                continue
            tensor = torch.as_tensor(mod_dict[name]).reshape(-1).long()
            if tensor.numel() != info["max_tokens"]:
                raise ValueError(
                    f"modality {name!r}: expected {info['max_tokens']} tokens, got {tensor.numel()}"
                )
            masked[name] = self.image_mask(tensor, int(info["max_tokens"]), in_budget, tgt_budget)
        return masked
