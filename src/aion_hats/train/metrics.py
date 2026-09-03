"""Training metrics: smoothed meters, the periodic progress logger, wandb and log.txt.

Ported from 4M's ``fourm/utils/logger.py`` (itself from DETR). ``MetricLogger.log_every``
works with length-less iterables when ``iter_len`` is given, which is what an infinite
lsdb stream needs.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

log = logging.getLogger(__name__)


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window or the
    global series average."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        self.deque: deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1) -> None:
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device: torch.device | None = None) -> None:
        """Sum ``count``/``total`` across ranks (the window is not synchronized)."""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        count, total = t.tolist()
        self.count = int(count)
        self.total = total

    @property
    def median(self) -> float:
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self) -> float:
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / max(self.count, 1)

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        if not self.deque:
            return "n/a"
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    def __init__(self, delimiter: str = "  ", enabled: bool = True) -> None:
        self.meters: dict[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.enabled = enabled

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(float(v))

    def __getattr__(self, attr: str) -> Any:
        meters = self.__dict__.get("meters", {})
        if attr in meters:
            return meters[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self) -> str:
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self, device: torch.device | None = None) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes(device)

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        self.meters[name] = meter

    def global_averages(self, prefix: str = "") -> dict[str, float]:
        return {prefix + k: meter.global_avg for k, meter in self.meters.items()}

    def log_every(
        self, iterable: Iterable, print_freq: int, iter_len: int | None = None, header: str = ""
    ) -> Iterator:
        iter_len = len(iterable) if iter_len is None else iter_len  # type: ignore[arg-type]
        start = end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        width = len(str(iter_len))
        cuda = torch.cuda.is_available()
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if self.enabled and (i % print_freq == 0 or i == iter_len - 1):
                eta = "?"
                if iter_len > 0:
                    eta = str(datetime.timedelta(seconds=int(iter_time.global_avg * (iter_len - i))))
                msg = (
                    f"{header} [{i:{width}d}/{iter_len if iter_len > 0 else '?'}] eta: {eta}  "
                    f"{self}  time: {iter_time}  data: {data_time}"
                )
                if cuda:
                    msg += f"  max mem: {torch.cuda.max_memory_allocated() / 2**20:.0f}MB"
                log.info(msg)
            end = time.time()
        total = time.time() - start
        per_iter = f"{total / iter_len:.4f}" if iter_len > 0 else "?"
        if self.enabled:
            log.info("%s Total time: %s (%s s / it)", header, datetime.timedelta(seconds=int(total)), per_iter)


class WandbLogger:
    """Thin wrapper around ``wandb`` that tolerates a missing/failed connection."""

    def __init__(self, config: dict[str, Any], **kwargs: Any) -> None:
        try:
            import wandb
        except ImportError as err:  # pragma: no cover - depends on the environment
            raise ImportError("wandb logging requested but wandb is not installed (pip install wandb)") from err
        self._wandb = wandb
        self.step = 0
        wandb.init(config=config, **kwargs)

    def set_step(self, step: int | None = None) -> None:
        self.step = step if step is not None else self.step + 1

    def update(self, metrics: dict[str, Any]) -> None:
        payload = {}
        for k, v in metrics.items():
            if v is None:
                continue
            payload[k] = v.item() if isinstance(v, torch.Tensor) else v
        try:
            self._wandb.log(payload, step=self.step)
        except (self._wandb.CommError, BrokenPipeError):  # pragma: no cover
            log.error("wandb logging failed, skipping")

    def finish(self) -> None:
        try:
            self._wandb.finish()
        except (self._wandb.CommError, BrokenPipeError):  # pragma: no cover
            log.error("wandb finish failed")


class JsonlLogger:
    """Appends one JSON object per line to ``<output_dir>/log.txt`` (the 4M convention)."""

    def __init__(self, output_dir: str | os.PathLike, name: str = "log.txt", enabled: bool = True) -> None:
        self.path = Path(output_dir) / name
        self.enabled = enabled

    def write(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
