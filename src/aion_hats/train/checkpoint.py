"""Full-state-dict checkpoints that work for both plain and FSDP-wrapped models.

Ported from 4M's ``fsdp_utils`` but built on ``torch.distributed.checkpoint.state_dict``,
whose ``get_*``/``set_*`` helpers return the *unwrapped* parameter names for plain modules
and FSDP units alike. A ``checkpoint-<epoch>.pth`` therefore contains
``{"model", "optimizer", "epoch", "args"}`` in the same layout whether it was written by a
single GPU or a multi-node job, and can be resumed by either.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

log = logging.getLogger(__name__)

CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)\.pth$")


def _target(model: torch.nn.Module) -> torch.nn.Module:
    # torch.compile wrappers are transparent to the state-dict helpers once unwrapped
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Full model state dict on CPU (collective under FSDP)."""
    return get_model_state_dict(
        _target(model), options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return get_optimizer_state_dict(
        _target(model), optimizer, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )


def checkpoint_path(output_dir: str | os.PathLike, epoch: int | str) -> Path:
    return Path(output_dir) / f"checkpoint-{epoch}.pth"


def save_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: dict[str, Any],
    is_main: bool = True,
) -> Path:
    """Gather (all ranks) and write (main rank only, atomically) a checkpoint."""
    path = Path(path)
    payload: dict[str, Any] = {"model": model_state(model), "epoch": int(epoch), "args": config}
    if optimizer is not None:
        payload["optimizer"] = optimizer_state(model, optimizer)
    if is_main:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        log.info("saved checkpoint %s (epoch %d)", path, epoch)
    return path


def load_checkpoint(
    path: str | os.PathLike,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and ``optimizer``); returns ``{"epoch", "args"}``."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    set_model_state_dict(
        _target(model), payload["model"], options=StateDictOptions(full_state_dict=True)
    )
    if optimizer is not None and "optimizer" in payload:
        # strict=False: parameters that never received a gradient (e.g. the decoder token
        # embedding of image modalities, replaced by the mask token) have no optimizer state
        set_optimizer_state_dict(
            _target(model), optimizer, payload["optimizer"],
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
    log.info("loaded checkpoint %s (epoch %s)", path, payload.get("epoch"))
    return {"epoch": payload.get("epoch"), "args": payload.get("args")}


def latest_checkpoint(output_dir: str | os.PathLike) -> Path | None:
    """The ``checkpoint-<epoch>.pth`` with the highest epoch in ``output_dir``."""
    best: tuple[int, Path] | None = None
    for candidate in Path(output_dir).glob("checkpoint-*.pth"):
        match = CHECKPOINT_RE.search(candidate.name)
        if match:
            epoch = int(match.group(1))
            if best is None or epoch > best[0]:
                best = (epoch, candidate)
    return best[1] if best else None
