"""The training loop, ported from 4M's ``run_training_4m_fsdp.py``.

One :class:`Trainer` covers both a single process (CPU, one GPU, Colab) and a multi-process
FSDP job launched with ``srun``/``torchrun``; the differences are confined to a few
``self.is_fsdp`` branches (wrapping, ``no_sync`` on accumulation steps, gradient clipping).
Batches come from an infinite :class:`HatsTokenDataset`; an "epoch" is a fixed number of
optimizer steps (``schedule.steps_per_epoch``), as in the source's webdataset setup, so
that every rank does the same number of collectives.
"""

from __future__ import annotations

import logging
import math
import random
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..distributed import WorkerContext
from .checkpoint import checkpoint_path, latest_checkpoint, load_checkpoint, save_checkpoint
from .config import DatasetSpec, TrainConfig, config_to_dict
from .data import HatsTokenDataset, build_dataloader
from .dist import all_reduce_flag, setup_distributed, teardown
from .masking import UnifiedMasking
from .metrics import JsonlLogger, MetricLogger, SmoothedValue, WandbLogger
from .modality_info import resolve_modality_info
from .model import (
    apply_act_checkpoint,
    build_model,
    count_parameters,
    export_pretrained,
    hub_config,
    wrap_fsdp,
)
from .optim import Schedules, create_optimizer, epochs_from_tokens, scaled_lr, steps_from_tokens

log = logging.getLogger(__name__)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class Trainer:
    """Set up data, model, optimizer and schedules from a :class:`TrainConfig` and train.

    Args:
        cfg: the run configuration.
        catalogs: optional ``{dataset name: lsdb.Catalog or factory}`` overriding the
            ``catalog`` path of the matching :class:`DatasetSpec` (e.g. a cone search or a
            crossmatch prepared in a notebook).
        ctx: rank/world size/device; discovered from the environment when omitted.
    """

    def __init__(
        self,
        cfg: TrainConfig,
        *,
        catalogs: dict[str, Any] | None = None,
        ctx: WorkerContext | None = None,
    ) -> None:
        self.cfg = cfg
        self.catalogs = dict(catalogs or {})
        self.ctx = ctx if ctx is not None else WorkerContext.from_env(device=cfg.run.device)
        self.output_dir = Path(cfg.run.output_dir)
        self._ready = False

    # --- setup -------------------------------------------------------------------------

    def setup(self) -> Trainer:
        if self._ready:
            return self
        cfg, ctx = self.cfg, self.ctx
        self.device = ctx.device
        self.is_main = ctx.is_main
        self.distributed = setup_distributed(ctx)
        self.is_fsdp = ctx.world_size > 1
        if self.is_fsdp and self.device.type != "cuda":
            raise RuntimeError("multi-process training uses FSDP and needs one CUDA device per rank")
        self.dtype = DTYPES[cfg.run.dtype]

        seed = cfg.run.seed + ctx.rank
        torch.manual_seed(seed)
        np.random.seed(seed % 2**32)
        random.seed(seed)
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        # Modalities and masking
        self.all_domains = cfg.all_domains
        spec = cfg.data.datasets[0]
        self.train_dataset = self._make_dataset(spec, infinite=True, start_epoch=0)

        # Batch arithmetic (4M conventions)
        run, sched = cfg.run, cfg.schedule
        self.global_batch_size = run.batch_size * run.accum_iter * ctx.world_size
        tokens_per_sample = cfg.data.num_input_tokens + cfg.data.num_target_tokens
        if sched.steps_per_epoch is not None:
            self.steps_per_epoch = int(sched.steps_per_epoch)
        else:
            rows = self.train_dataset.estimate_rows()
            if rows is None:
                raise ValueError("schedule.steps_per_epoch is required: the catalog does not report its size")
            self.steps_per_epoch = max(rows // (run.batch_size * ctx.world_size), 1)
        if sched.epochs is not None:
            self.epochs = int(sched.epochs)
        else:
            self.epochs = epochs_from_tokens(
                sched.total_tokens_b, tokens_per_sample, self.steps_per_epoch, self.global_batch_size
            )
        if run.max_steps is not None:
            self.epochs = min(self.epochs, math.ceil(run.max_steps / self.steps_per_epoch))
        self.total_steps = self.epochs * self.steps_per_epoch
        if sched.warmup_steps is not None:
            warmup = int(sched.warmup_steps)
        elif sched.warmup_epochs is not None:
            warmup = int(sched.warmup_epochs) * self.steps_per_epoch
        elif sched.warmup_tokens_b is not None:
            warmup = steps_from_tokens(sched.warmup_tokens_b, tokens_per_sample, self.global_batch_size)
        else:
            warmup = 0
        self.warmup_steps = min(warmup, self.total_steps)

        # Model
        model = build_model(cfg.model, cfg.model.domains_in, cfg.model.domains_out)
        self.model_config = hub_config(model)
        self.n_parameters = count_parameters(model)
        if self.is_fsdp:
            model = wrap_fsdp(model, self.device, self.dtype)
        else:
            model = model.to(self.device)
        if cfg.model.act_checkpoint:
            apply_act_checkpoint(model)
        self.lr = scaled_lr(cfg.optim.blr, self.global_batch_size)
        self.min_lr = scaled_lr(cfg.optim.min_blr, self.global_batch_size)
        self.optimizer = create_optimizer(model, cfg.optim, self.lr)
        if cfg.model.compile:
            model = torch.compile(model, dynamic=True)
        self.model = model
        self.schedules = Schedules.build(cfg.optim, self.lr, self.min_lr, self.total_steps, self.warmup_steps)

        # Resume
        self.start_epoch = 0
        resume = cfg.run.resume
        if resume is None and cfg.run.auto_resume:
            found = latest_checkpoint(self.output_dir)
            resume = str(found) if found else None
        if resume:
            state = load_checkpoint(resume, self.model, self.optimizer)
            self.start_epoch = int(state["epoch"]) + 1
        self.train_dataset.set_epoch(self.start_epoch)

        # Loaders
        self.train_loader = build_dataloader(self.train_dataset, cfg.data, run.batch_size, self.device)
        self.train_iter = iter(self.train_loader)
        eval_bs = cfg.data.eval_batch_size or run.batch_size
        self.eval_loaders: dict[str, DataLoader] = {}
        for eval_spec in cfg.data.eval_datasets:
            dataset = self._make_dataset(eval_spec, infinite=False, start_epoch=0)
            self.eval_loaders[eval_spec.name] = build_dataloader(dataset, cfg.data, eval_bs, self.device)

        # Loggers
        self.jsonl = JsonlLogger(self.output_dir, enabled=self.is_main)
        self.wandb = None
        if self.is_main and cfg.run.wandb is not None:
            w = cfg.run.wandb
            self.wandb = WandbLogger(
                config_to_dict(cfg), project=w.project, entity=w.entity, name=w.run_name,
                group=w.group, mode=w.mode, tags=w.tags or None,
            )
        self._log_setup()
        self._ready = True
        return self

    def _make_dataset(self, spec: DatasetSpec, *, infinite: bool, start_epoch: int) -> HatsTokenDataset:
        cfg, ctx = self.cfg, self.ctx
        info = resolve_modality_info(spec.in_domains, spec.out_domains, spec.input_alphas, spec.target_alphas)
        masker = UnifiedMasking(
            info,
            input_tokens_range=(cfg.data.min_input_tokens, cfg.data.num_input_tokens),
            target_tokens_range=(cfg.data.min_target_tokens, cfg.data.num_target_tokens),
        )
        source = self.catalogs.get(spec.name, spec.catalog)
        if source is None:
            raise ValueError(f"dataset {spec.name!r} has no catalog (set DatasetSpec.catalog or pass catalogs=)")
        return HatsTokenDataset(
            source,
            spec.modalities,
            masker,
            split=spec.split,
            filter=spec.filter,
            shuffle=cfg.data.shuffle,
            seed=cfg.run.seed,
            shuffle_buffer=cfg.data.shuffle_buffer,
            rank=ctx.rank,
            world_size=ctx.world_size,
            start_epoch=start_epoch,
            infinite=infinite,
        )

    def _log_setup(self) -> None:
        if not self.is_main:
            return
        log.info(
            "model: %.2fM parameters, %d input + %d target tokens, loss %s, dtype %s, fsdp=%s, compile=%s",
            self.n_parameters / 1e6, self.cfg.data.num_input_tokens, self.cfg.data.num_target_tokens,
            self.cfg.run.loss_type, self.cfg.run.dtype, self.is_fsdp, self.cfg.model.compile,
        )
        log.info(
            "schedule: %d epochs x %d steps (warmup %d), global batch %d, lr %.3e -> %.3e, wd %.3g -> %.3g",
            self.epochs, self.steps_per_epoch, self.warmup_steps, self.global_batch_size,
            self.lr, self.min_lr, self.cfg.optim.weight_decay, self.cfg.optim.weight_decay_end,
        )
        if self.start_epoch:
            log.info("resuming at epoch %d", self.start_epoch)

    # --- steps -------------------------------------------------------------------------

    def _to_device(self, batch: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
        return {
            mod: {k: v.to(self.device, non_blocking=True) for k, v in d.items()}
            for mod, d in batch.items()
            if mod in self.all_domains
        }

    def _forward(self, mod_dict: dict[str, dict[str, torch.Tensor]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        with torch.autocast(
            device_type=self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            return self.model(
                mod_dict,
                num_encoder_tokens=self.cfg.data.num_input_tokens,
                num_decoder_tokens=self.cfg.data.num_target_tokens,
                loss_type=self.cfg.run.loss_type,
            )

    def _clip_grad(self, max_norm: float) -> torch.Tensor:
        if self.is_fsdp:
            return self.model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        cfg = self.cfg
        self.model.train()
        accum = cfg.run.accum_iter
        max_norm = cfg.optim.clip_grad
        start_step = epoch * self.steps_per_epoch
        steps = self.steps_per_epoch
        if cfg.run.max_steps is not None:
            steps = max(min(steps, cfg.run.max_steps - start_step), 0)
        logger = MetricLogger(enabled=self.is_main)
        logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = f"Epoch: [{epoch}]"
        grad_norm = None
        batches = islice(self.train_iter, steps)
        for step, batch in enumerate(logger.log_every(batches, cfg.run.print_freq, iter_len=steps, header=header)):
            it = start_step + step
            update_grad = (step + 1) % accum == 0 or step + 1 == steps
            if step % accum == 0:
                for group in self.optimizer.param_groups:
                    group["lr"] = float(self.schedules.lr[it]) * group.get("lr_scale", 1.0)
                    if group["weight_decay"] > 0:
                        group["weight_decay"] = float(self.schedules.wd[it])
            mod_dict = self._to_device(batch)

            sync = nullcontext() if (update_grad or not self.is_fsdp) else self.model.no_sync()
            with sync:
                loss, mod_loss = self._forward(mod_dict)
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    dump = self.output_dir / "debug_mod_dict.pt"
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({m: {k: v.cpu() for k, v in d.items()} for m, d in mod_dict.items()}, dump)
                    raise RuntimeError(f"loss is {loss_value} at step {it}; batch saved to {dump}")
                (loss / accum).backward()
                if update_grad:
                    skip = False
                    if max_norm is not None:
                        norm = self._clip_grad(max_norm)
                        if cfg.optim.skip_nan_grad:
                            skip = all_reduce_flag(bool(torch.isnan(norm).any()), self.device)
                        grad_norm = float(norm)
                    if skip:
                        log.warning("skipping step %d (epoch %d): NaN gradients", step, epoch)
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()

            lrs = [g["lr"] for g in self.optimizer.param_groups]
            wds = [g["weight_decay"] for g in self.optimizer.param_groups if g["weight_decay"] > 0]
            mod_losses = {f"{m}_loss": float(v.detach()) for m, v in mod_loss.items()}
            logger.update(loss=loss_value, lr=max(lrs), min_lr=min(lrs), weight_decay=wds[0] if wds else None,
                          grad_norm=grad_norm, **mod_losses)
            if self.wandb is not None:
                seen = it * self.global_batch_size / accum
                self.wandb.set_step(it)
                self.wandb.update({
                    "loss": loss_value, "lr": max(lrs), "weight_decay": wds[0] if wds else None,
                    "grad_norm": grad_norm, **mod_losses,
                    "input_tokens_seen_b": seen * cfg.data.num_input_tokens / 1e9,
                    "target_tokens_seen_b": seen * cfg.data.num_target_tokens / 1e9,
                    "total_tokens_seen_b": seen * (cfg.data.num_input_tokens + cfg.data.num_target_tokens) / 1e9,
                })
        logger.synchronize_between_processes(self.device)
        if self.is_main:
            log.info("averaged stats: %s", logger)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return logger.global_averages("[Epoch] ")

    @torch.no_grad()
    def evaluate(self, name: str, loader: DataLoader, steps: int | None = None) -> dict[str, float]:
        steps = self.cfg.data.eval_steps if steps is None else steps
        self.model.eval()
        prefix = f"[Eval ({name})] "
        logger = MetricLogger(enabled=self.is_main)
        for batch in logger.log_every(islice(loader, steps), max(steps // 5, 1), iter_len=steps, header=prefix):
            loss, mod_loss = self._forward(self._to_device(batch))
            logger.update(loss=loss.item(), **{f"{m}_loss": float(v.detach()) for m, v in mod_loss.items()})
        logger.synchronize_between_processes(self.device)
        self.model.train()
        if not logger.meters:
            log.warning("evaluation %r produced no batches (empty split or filter?)", name)
        return logger.global_averages(prefix)

    # --- driver ------------------------------------------------------------------------

    def save(self, epoch: int, name: int | str | None = None) -> Path:
        return save_checkpoint(
            checkpoint_path(self.output_dir, epoch if name is None else name),
            self.model, self.optimizer, epoch, config_to_dict(self.cfg), is_main=self.is_main,
        )

    def export(self, out_dir: str | Path | None = None) -> Path:
        out_dir = self.output_dir / "final" if out_dir is None else Path(out_dir)
        return export_pretrained(self.model, out_dir, self.model_config, is_main=self.is_main)

    def fit(self) -> Path:
        """Train from ``start_epoch`` to ``epochs``, checkpoint, evaluate, export; returns ``output_dir``."""
        self.setup()
        cfg = self.cfg
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        try:
            for epoch in range(self.start_epoch, self.epochs):
                if self.wandb is not None:
                    self.wandb.set_step(epoch * self.steps_per_epoch)
                stats = self.train_one_epoch(epoch)
                last = epoch + 1 == self.epochs
                if (epoch + 1) % cfg.run.save_ckpt_freq == 0 or last:
                    self.save(epoch)
                seen = (epoch + 1) * self.steps_per_epoch * self.global_batch_size / cfg.run.accum_iter
                record: dict[str, Any] = {
                    **stats,
                    "epoch": epoch,
                    "n_parameters": self.n_parameters,
                    "input_tokens_seen_b": seen * cfg.data.num_input_tokens / 1e9,
                    "target_tokens_seen_b": seen * cfg.data.num_target_tokens / 1e9,
                    "total_tokens_seen_b": seen * (cfg.data.num_input_tokens + cfg.data.num_target_tokens) / 1e9,
                }
                for name, loader in self.eval_loaders.items():
                    record.update(self.evaluate(name, loader))
                if self.wandb is not None:
                    self.wandb.update(record)
                self.jsonl.write(record)
            self.export()
        finally:
            if self.wandb is not None:
                self.wandb.finish()
            teardown()
        log.info("training time %s", time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
        return self.output_dir


def train(cfg: TrainConfig, **kwargs: Any) -> Path:
    """Convenience wrapper: ``Trainer(cfg, **kwargs).fit()``."""
    return Trainer(cfg, **kwargs).fit()
