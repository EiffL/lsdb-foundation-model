"""Process-group setup on top of :class:`aion_hats.distributed.WorkerContext`.

Rank, world size and local rank come from the launcher environment (``srun``, ``torchrun``,
MPI); ``MASTER_ADDR``/``MASTER_PORT`` default to the first node of ``SLURM_NODELIST`` when
not set. A single process needs no process group at all.
"""

from __future__ import annotations

import datetime
import logging
import os
import subprocess

import torch
import torch.distributed as dist

from ..distributed import WorkerContext

log = logging.getLogger(__name__)

DEFAULT_PORT = "29500"


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _slurm_master_addr() -> str | None:
    nodelist = os.environ.get("SLURM_NODELIST") or os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return None
    try:
        out = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist], capture_output=True, text=True, check=True
        )
        return out.stdout.split()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return nodelist.split(",")[0].split("[")[0] or None


def setup_distributed(
    ctx: WorkerContext, backend: str | None = None, timeout_s: float = 4800
) -> bool:
    """Initialize the default process group if ``ctx.world_size > 1``; returns whether it did."""
    if ctx.world_size == 1:
        return False
    if is_distributed():
        return True
    os.environ.setdefault("MASTER_ADDR", _slurm_master_addr() or "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", DEFAULT_PORT)
    os.environ.setdefault("RANK", str(ctx.rank))
    os.environ.setdefault("WORLD_SIZE", str(ctx.world_size))
    if ctx.device.type == "cuda":
        torch.cuda.set_device(ctx.device)
    backend = backend or ("nccl" if ctx.device.type == "cuda" else "gloo")
    log.info(
        "distributed init: rank %d/%d, %s, backend %s, master %s:%s",
        ctx.rank, ctx.world_size, ctx.device, backend,
        os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"],
    )
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=ctx.rank,
        world_size=ctx.world_size,
        timeout=datetime.timedelta(seconds=timeout_s),
    )
    dist.barrier()
    return True


def all_reduce_flag(value: bool, device: torch.device | None = None) -> bool:
    """``True`` if ``value`` is true on any rank (4M's ``reduce_bool``)."""
    if not is_distributed():
        return bool(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() > 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def teardown() -> None:
    if is_distributed():
        dist.destroy_process_group()
