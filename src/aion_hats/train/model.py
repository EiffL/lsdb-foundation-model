"""Build, wrap and export the FourM transformer of the ``aion`` package.

The model code is *not* copied here: ``aion.fourm.fm.FM`` (the ``PyTorchModelHubMixin``
flavour of ``FourM``) is instantiated from a config dict with the same keys as the
``config.json`` of ``polymathic-ai/aion-base``, or loaded from the Hub / an exported
directory with ``FM.from_pretrained``. ``aion.AION`` is not used for training because it
overrides ``forward`` with the inference API; ``AION.from_pretrained`` still loads what
:func:`export_pretrained` writes.
"""

from __future__ import annotations

import copy
import functools
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from .config import ModelConfig

log = logging.getLogger(__name__)

# dim, encoder depth, decoder depth, heads -- the fm_*_swiglu_qknorm_nobias factories of 4M
PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"dim": 384, "encoder_depth": 6, "decoder_depth": 6, "num_heads": 6},
    "small": {"dim": 512, "encoder_depth": 8, "decoder_depth": 8, "num_heads": 8},
    "base": {"dim": 768, "encoder_depth": 12, "decoder_depth": 12, "num_heads": 12},
    "large": {"dim": 1024, "encoder_depth": 24, "decoder_depth": 24, "num_heads": 16},
    "xlarge": {"dim": 2048, "encoder_depth": 24, "decoder_depth": 24, "num_heads": 32},
}

# Architecture flags shared by all AION-1 models (polymathic-ai/aion-base config.json)
ARCH_FLAGS: dict[str, Any] = {
    "act_layer": "SiLU",
    "gated_mlp": True,
    "qk_norm": True,
    "qkv_bias": False,
    "proj_bias": False,
    "mlp_bias": False,
    "norm_bias": False,
    "mlp_ratio": 4,
}


def model_config(
    preset: str,
    domains_in: list[str],
    domains_out: list[str],
    overrides: dict[str, Any] | None = None,
    num_register_tokens: int = 0,
) -> dict[str, Any]:
    """``FM`` config dict for a size preset (keys identical to aion-base's ``config.json``)."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    config: dict[str, Any] = {**ARCH_FLAGS, **PRESETS[preset]}
    if num_register_tokens:
        config["num_register_tokens"] = int(num_register_tokens)
    config.update(overrides or {})
    config["domains_in"] = sorted(domains_in)
    config["domains_out"] = sorted(domains_out)
    return config


def build_model(cfg: ModelConfig, domains_in: list[str], domains_out: list[str]) -> nn.Module:
    """A fresh ``FM`` from the preset, or a pretrained one from ``cfg.init_from``."""
    from aion.fourm.fm import FM

    if cfg.init_from:
        log.info("loading pretrained model %s", cfg.init_from)
        model = FM.from_pretrained(cfg.init_from)
        if cfg.overrides:
            log.warning("model.overrides are ignored when init_from is set: %s", cfg.overrides)
        missing_in = set(domains_in) - set(model.encoder_embeddings)
        missing_out = set(domains_out) - set(model.decoder_embeddings)
        if missing_in or missing_out:
            raise ValueError(
                f"{cfg.init_from} has no embeddings for inputs {sorted(missing_in)} / "
                f"outputs {sorted(missing_out)}; adding modalities to a pretrained model is not supported"
            )
        return model
    config = model_config(cfg.preset, domains_in, domains_out, cfg.overrides, cfg.num_register_tokens)
    log.info("building %s model: %s", cfg.preset, {k: v for k, v in config.items() if "domains" not in k})
    return FM(config)


def hub_config(model: nn.Module) -> dict[str, Any]:
    """The config dict ``FM`` was built from (survives FSDP / compile wrapping)."""
    model = unwrap(model)
    config = getattr(model, "_hub_mixin_config", None)
    if config is None:
        raise ValueError("model has no hub config; was it built with FM(config)?")
    return copy.deepcopy(dict(config))


def unwrap(model: nn.Module) -> nn.Module:
    """The original ``FM`` behind ``torch.compile`` and FSDP wrappers."""
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif type(model).__name__ == "FullyShardedDataParallel":
            model = model.module
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def wrap_fsdp(model: nn.Module, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """FSDP ZeRO-2 wrapping as in the 4M FSDP script (one unit per transformer block)."""
    from aion.fourm.fm_utils import Block, DecoderBlock
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Block, DecoderBlock})
    # Only reduced grads are in bf16 here; autocast handles the compute dtype
    mp_policy = MixedPrecision(reduce_dtype=torch.bfloat16) if dtype == torch.bfloat16 else None
    return FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        device_id=device,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )


def apply_act_checkpoint(model: nn.Module) -> None:
    """Non-reentrant activation checkpointing on every transformer block."""
    from aion.fourm.fm_utils import Block, DecoderBlock
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    wrapper = functools.partial(
        checkpoint_wrapper, offload_to_cpu=False, checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=wrapper, check_fn=lambda m: isinstance(m, (Block, DecoderBlock))
    )


def export_pretrained(
    model: nn.Module, out_dir: str | Path, config: dict[str, Any] | None = None, is_main: bool = True
) -> Path:
    """Write ``config.json`` + ``model.safetensors`` loadable with ``FM/AION.from_pretrained``.

    Collective under FSDP (every rank gathers the full state dict); only the main process
    writes.
    """
    from aion.fourm.fm import FM

    from .checkpoint import model_state

    out_dir = Path(out_dir)
    config = hub_config(model) if config is None else config
    state = model_state(model)
    if is_main:
        fresh = FM(copy.deepcopy(config))
        fresh.load_state_dict(state)
        out_dir.mkdir(parents=True, exist_ok=True)
        fresh.save_pretrained(out_dir, config=config)
        log.info("exported pretrained model to %s", out_dir)
    return out_dir
