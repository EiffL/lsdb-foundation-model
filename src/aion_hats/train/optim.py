"""Optimizer, parameter groups and step schedules (ported from 4M's ``optim_factory`` and
``scheduler`` modules plus the epoch/warmup arithmetic of the training script).

Learning-rate and weight-decay schedules are numpy arrays indexed by the global
optimizer step, exactly as in the source; the trainer writes ``lr[it] * lr_scale`` into
every parameter group before each step.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .config import OptimConfig

log = logging.getLogger(__name__)

FSDP_PREFIX = "_fsdp_wrapped_module."


def parameter_groups(
    model: torch.nn.Module, weight_decay: float, skip_list: set[str] | frozenset[str] = frozenset()
) -> list[dict]:
    """Two groups, ``decay`` and ``no_decay`` (norms, biases, ``skip_list``), each with ``lr_scale``."""
    groups: dict[str, dict] = {}
    names: dict[str, list[str]] = {}
    for name, param in model.named_parameters():
        name = name.replace(FSDP_PREFIX, "")  # FSDP(use_orig_params=True) keeps the original names
        if not param.requires_grad:
            continue
        no_decay = (
            "norm." in name
            or ".norm" in name
            or name.endswith((".bias", ".gamma"))
            or name in skip_list
        )
        group = "no_decay" if no_decay else "decay"
        if group not in groups:
            groups[group] = {"params": [], "weight_decay": 0.0 if no_decay else weight_decay, "lr_scale": 1.0}
            names[group] = []
        groups[group]["params"].append(param)
        names[group].append(name)
    for group, members in names.items():
        log.debug("param group %s (%d tensors): %s", group, len(members), members)
    return list(groups.values())


def create_optimizer(model: torch.nn.Module, cfg: OptimConfig, lr: float) -> torch.optim.Optimizer:
    skip = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = parameter_groups(model, cfg.weight_decay, skip)
    log.info("optimizer: AdamW lr=%.3e betas=%s eps=%g wd=%g", lr, cfg.betas, cfg.eps, cfg.weight_decay)
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(cfg.betas), eps=cfg.eps, weight_decay=cfg.weight_decay)


def cosine_schedule(
    base_value: float,
    final_value: float,
    total_steps: int,
    warmup_steps: int = 0,
    start_warmup_value: float = 0.0,
) -> np.ndarray:
    """Linear warmup then cosine decay, one value per step (4M's ``cosine_scheduler``)."""
    if total_steps < 0:
        raise ValueError("total_steps must be >= 0")
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    warmup = np.linspace(start_warmup_value, base_value, warmup_steps) if warmup_steps > 0 else np.array([])
    iters = np.arange(total_steps - warmup_steps)
    n = max(len(iters), 1)
    decay = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / n))
    schedule = np.concatenate((warmup, decay))
    assert len(schedule) == total_steps
    return schedule


def constant_schedule(base_value: float, total_steps: int) -> np.ndarray:
    return base_value * np.ones(total_steps)


@dataclass
class Schedules:
    lr: np.ndarray
    wd: np.ndarray

    @classmethod
    def build(
        cls,
        cfg: OptimConfig,
        lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
    ) -> Schedules:
        wd_end = cfg.weight_decay if cfg.weight_decay_end is None else cfg.weight_decay_end
        return cls(
            lr=cosine_schedule(lr, min_lr, total_steps, warmup_steps),
            wd=cosine_schedule(cfg.weight_decay, wd_end, total_steps),
        )


def scaled_lr(blr: float, global_batch_size: int) -> float:
    """``lr = blr * batch / 256`` (4M convention)."""
    return blr * global_batch_size / 256


def steps_from_tokens(tokens_b: float, tokens_per_sample: int, global_batch_size: int) -> int:
    """Optimizer steps that see ``tokens_b`` billion (input + target) tokens."""
    return math.ceil(tokens_b * 1e9 / (tokens_per_sample * global_batch_size))


def epochs_from_tokens(
    tokens_b: float, tokens_per_sample: int, steps_per_epoch: int, global_batch_size: int
) -> int:
    return math.ceil(steps_from_tokens(tokens_b, tokens_per_sample, global_batch_size) / steps_per_epoch)
