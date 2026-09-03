"""Train the AION/4M transformer on tokenized HATS catalogs read through lsdb.

Typical use::

    from aion_hats.train import load_config, train
    cfg = load_config("configs/ls_north_base.yaml", ["run.output_dir=runs/ls_north"])
    train(cfg)

The model comes from the ``aion`` package (``aion.fourm.fm.FM``); this package only
provides the data path (``stream``, ``data``, ``masking``) and the training loop
(``trainer``), ported from the 4M/AION-1 training code with webdataset replaced by lsdb.
"""

from .config import (
    DataConfig,
    DatasetSpec,
    ModelConfig,
    OptimConfig,
    RunConfig,
    ScheduleConfig,
    TrainConfig,
    WandbConfig,
    config_to_dict,
    load_config,
)
from .data import HatsTokenDataset, build_dataloader, frame_to_tokens
from .masking import UnifiedMasking, empty_mod_dict
from .modality_info import resolve_modality_info
from .model import PRESETS, build_model, export_pretrained, model_config
from .stream import PartitionStream, deal_partitions, split_of_pixel
from .trainer import Trainer, train

__all__ = [
    "PRESETS",
    "DataConfig",
    "DatasetSpec",
    "HatsTokenDataset",
    "ModelConfig",
    "OptimConfig",
    "PartitionStream",
    "RunConfig",
    "ScheduleConfig",
    "TrainConfig",
    "Trainer",
    "UnifiedMasking",
    "WandbConfig",
    "build_dataloader",
    "build_model",
    "config_to_dict",
    "deal_partitions",
    "empty_mod_dict",
    "export_pretrained",
    "frame_to_tokens",
    "load_config",
    "model_config",
    "resolve_modality_info",
    "split_of_pixel",
    "train",
]
