"""Per-run view of AION's modality registry, with token counts and masking alphas.

Ported from ``setup_modality_info`` and ``setup_sampling_mod_info`` of the 4M training
script: the global ``aion.fourm.modality_info.MODALITY_INFO`` is copied for the modalities
of a run, ``max_tokens`` is filled in for image-like modalities that leave it unset (AION's
``tok_image`` has ``max_tokens: None``), and the Beta/Dirichlet concentration parameters
of the masking scheme are attached (0 for a modality that is not an input / not a target).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

SUPPORTED_TYPES = ("img",)


def _alpha_for(mod: str, alphas: float | dict[str, float]) -> float:
    if isinstance(alphas, dict):
        if mod not in alphas:
            raise ValueError(f"no alpha given for modality {mod!r}")
        return float(alphas[mod])
    return float(alphas)


def resolve_modality_info(
    domains_in: Iterable[str],
    domains_out: Iterable[str],
    input_alphas: float | dict[str, float] = 1.0,
    target_alphas: float | dict[str, float] = 1.0,
    *,
    input_size: int = 96,
    patch_size: int = 4,
) -> dict[str, dict[str, Any]]:
    """Copy of ``MODALITY_INFO`` for ``domains_in | domains_out`` with ``max_tokens`` and alphas."""
    from aion.fourm.modality_info import MODALITY_INFO

    domains_in, domains_out = set(domains_in), set(domains_out)
    info: dict[str, dict[str, Any]] = {}
    for mod in sorted(domains_in | domains_out):
        if mod not in MODALITY_INFO:
            raise ValueError(
                f"unknown AION modality {mod!r}; known: {', '.join(sorted(MODALITY_INFO))}"
            )
        entry = copy.deepcopy(MODALITY_INFO[mod])
        if entry.get("type") not in SUPPORTED_TYPES:
            raise ValueError(
                f"modality {mod!r} has type {entry.get('type')!r}; only {SUPPORTED_TYPES} "
                "are supported by the training masker"
            )
        if entry.get("max_tokens") is None:
            size = entry.get("input_size", input_size)
            patch = entry.get("patch_size", patch_size)
            entry["max_tokens"] = (size // patch) ** 2
        entry.setdefault("min_tokens", 0)
        entry["input_alphas"] = [_alpha_for(mod, input_alphas) if mod in domains_in else 0.0]
        entry["target_alphas"] = [_alpha_for(mod, target_alphas) if mod in domains_out else 0.0]
        info[mod] = entry
    return info


def num_tokens(modality_info: dict[str, dict[str, Any]], mod: str) -> int:
    return int(modality_info[mod]["max_tokens"])
