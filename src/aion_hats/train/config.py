"""Training configuration: dataclasses, YAML loading and ``key.path=value`` overrides.

This replaces the argparse + ``parser.set_defaults(**yaml)`` scheme of the 4M training
script with plain dataclasses so that the Python API, the CLI and checkpoints share one
serializable description of a run.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

DTYPES = {"float32": "float32", "fp32": "float32", "bfloat16": "bfloat16", "bf16": "bfloat16"}


@dataclass
class DatasetSpec:
    """One tokenized HATS catalog and how its columns map onto AION modalities."""

    name: str = "train"
    catalog: str | None = None
    modalities: dict[str, str] = field(default_factory=lambda: {"tok_image": "tok_image"})
    in_domains: list[str] | None = None
    out_domains: list[str] | None = None
    input_alphas: float | dict[str, float] = 1.0
    target_alphas: float | dict[str, float] = 1.0
    weight: float = 1.0
    filter: dict[str, Any] | None = None
    split: str | None = None
    drop_nulls: str = "all"

    def __post_init__(self) -> None:
        if self.drop_nulls not in ("any", "all"):
            raise ValueError(f"dataset {self.name!r}: drop_nulls must be 'any' or 'all'")
        if isinstance(self.modalities, (list, tuple)):
            self.modalities = {str(m): str(m) for m in self.modalities}
        if self.in_domains is None:
            self.in_domains = sorted(self.modalities)
        if self.out_domains is None:
            self.out_domains = sorted(self.modalities)
        self.in_domains = sorted(self.in_domains)
        self.out_domains = sorted(self.out_domains)
        unknown = (set(self.in_domains) | set(self.out_domains)) - set(self.modalities)
        if unknown:
            raise ValueError(
                f"dataset {self.name!r}: domains {sorted(unknown)} are not in modalities "
                f"{sorted(self.modalities)}"
            )
        if self.split not in (None, "train", "val"):
            raise ValueError(f"dataset {self.name!r}: split must be 'train', 'val' or null")
        if self.weight <= 0:
            raise ValueError(f"dataset {self.name!r}: weight must be positive")

    @property
    def all_domains(self) -> list[str]:
        return sorted(set(self.in_domains or []) | set(self.out_domains or []))


@dataclass
class ModelConfig:
    preset: str = "base"
    init_from: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)
    domains_in: list[str] | None = None
    domains_out: list[str] | None = None
    compile: bool = False
    act_checkpoint: bool = False
    num_register_tokens: int = 0


@dataclass
class DataConfig:
    datasets: list[DatasetSpec] = field(default_factory=lambda: [DatasetSpec()])
    num_input_tokens: int = 256
    num_target_tokens: int = 128
    min_input_tokens: int | None = None
    min_target_tokens: int | None = None
    shuffle: bool = True
    shuffle_buffer: int = 16384
    num_workers: int = 4
    multiprocessing_context: str | None = None
    prefetch_factor: int = 4
    pin_memory: bool | None = None
    eval_datasets: list[DatasetSpec] = field(default_factory=list)
    eval_steps: int = 50
    eval_batch_size: int | None = None

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("data.datasets must list at least one dataset")
        if len(self.datasets) > 1:
            raise NotImplementedError("mixing several training datasets is not supported yet")
        if self.min_input_tokens is None:
            self.min_input_tokens = self.num_input_tokens
        if self.min_target_tokens is None:
            self.min_target_tokens = self.num_target_tokens
        if not 1 <= self.min_input_tokens <= self.num_input_tokens:
            raise ValueError("need 1 <= min_input_tokens <= num_input_tokens")
        if not 1 <= self.min_target_tokens <= self.num_target_tokens:
            raise ValueError("need 1 <= min_target_tokens <= num_target_tokens")
        names = [d.name for d in [*self.datasets, *self.eval_datasets]]
        if len(set(names)) != len(names):
            raise ValueError(f"dataset names must be unique, got {names}")


@dataclass
class OptimConfig:
    opt: str = "adamw"
    blr: float = 1e-4
    min_blr: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.05
    weight_decay_end: float | None = None
    clip_grad: float | None = 1.0
    skip_nan_grad: bool = True

    def __post_init__(self) -> None:
        self.betas = tuple(float(b) for b in self.betas)  # type: ignore[assignment]
        if self.opt.lower() != "adamw":
            raise ValueError(f"only the adamw optimizer is supported, got {self.opt!r}")
        if self.weight_decay_end is None:
            self.weight_decay_end = self.weight_decay


@dataclass
class ScheduleConfig:
    epochs: int | None = None
    total_tokens_b: float | None = None
    steps_per_epoch: int | None = None
    warmup_epochs: int | None = None
    warmup_steps: int | None = None
    warmup_tokens_b: float | None = None
    scheduler: str = "cosine"

    def __post_init__(self) -> None:
        if (self.epochs is None) == (self.total_tokens_b is None):
            raise ValueError("set exactly one of schedule.epochs and schedule.total_tokens_b")
        warmups = [self.warmup_epochs, self.warmup_steps, self.warmup_tokens_b]
        if sum(w is not None for w in warmups) > 1:
            raise ValueError("set at most one of warmup_epochs, warmup_steps, warmup_tokens_b")
        if self.scheduler != "cosine":
            raise ValueError(f"only the cosine scheduler is supported, got {self.scheduler!r}")


@dataclass
class WandbConfig:
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    mode: str = "online"
    tags: list[str] = field(default_factory=list)


@dataclass
class RunConfig:
    output_dir: str = "runs/aion_hats"
    batch_size: int = 256
    accum_iter: int = 1
    dtype: str = "bfloat16"
    seed: int = 0
    save_ckpt_freq: int = 1
    print_freq: int = 50
    auto_resume: bool = True
    resume: str | None = None
    max_steps: int | None = None
    loss_type: str = "mod"
    device: str | None = None
    wandb: WandbConfig | None = None

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ValueError(f"run.dtype must be one of {sorted(DTYPES)}, got {self.dtype!r}")
        self.dtype = DTYPES[self.dtype]
        if self.loss_type not in ("mod", "token"):
            raise ValueError("run.loss_type must be 'mod' or 'token'")
        if self.batch_size < 1 or self.accum_iter < 1:
            raise ValueError("run.batch_size and run.accum_iter must be >= 1")


@dataclass
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    schedule: ScheduleConfig = field(default_factory=lambda: ScheduleConfig(epochs=1))
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        union_in = sorted({m for d in self.data.datasets for m in d.in_domains or []})
        union_out = sorted({m for d in self.data.datasets for m in d.out_domains or []})
        if self.model.domains_in is None:
            self.model.domains_in = union_in
        if self.model.domains_out is None:
            self.model.domains_out = union_out
        for spec in self.data.eval_datasets:
            extra = set(spec.all_domains) - set(self.model.domains_in) - set(self.model.domains_out)
            if extra:
                raise ValueError(f"eval dataset {spec.name!r} uses unknown domains {sorted(extra)}")

    @property
    def all_domains(self) -> list[str]:
        return sorted(set(self.model.domains_in or []) | set(self.model.domains_out or []))


# --- (de)serialization -----------------------------------------------------------------


def from_dict(cls: type, data: Any) -> Any:
    """Build a (nested) dataclass from plain dicts, validating unknown keys."""
    if data is None or not is_dataclass(cls):
        return data
    if isinstance(data, cls):
        return data
    if not isinstance(data, dict):
        raise TypeError(f"{cls.__name__}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        kwargs[name] = _coerce_field(known[name].type, value)
    return cls(**kwargs)


_NESTED = {
    "model": ModelConfig,
    "data": DataConfig,
    "optim": OptimConfig,
    "schedule": ScheduleConfig,
    "run": RunConfig,
    "wandb": WandbConfig,
    "datasets": DatasetSpec,
    "eval_datasets": DatasetSpec,
}


def _coerce_field(annotation: Any, value: Any) -> Any:
    text = str(annotation)
    for cls in _NESTED.values():
        if cls.__name__ in text:
            if isinstance(value, list):
                return [from_dict(cls, v) for v in value]
            return from_dict(cls, value)
    return value


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """Plain, YAML/JSON/torch.save friendly dict of a config."""
    return dataclasses.asdict(cfg)


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (values parsed as YAML) to a nested dict."""

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not of the form key.path=value")
        path, _, raw = item.partition("=")
        keys = path.strip().split(".")
        node: Any = data
        for key in keys[:-1]:
            node = _child(node, key, item, create=True)
        last = keys[-1]
        if isinstance(node, list):
            node[_index(node, last, item)] = parse_value(raw)
        else:
            node[last] = parse_value(raw)
    return data


def _index(node: list, key: str, item: str) -> int:
    if not key.isdigit() or int(key) >= len(node):
        raise ValueError(f"override {item!r}: {key!r} is not an index into a list of {len(node)}")
    return int(key)


def _child(node: Any, key: str, item: str, create: bool) -> Any:
    """``node[key]``, descending into lists by integer index and creating missing sections."""
    if isinstance(node, list):
        return node[_index(node, key, item)]
    if not isinstance(node, dict):
        raise TypeError(f"override {item!r}: {key!r} is not a section")
    if create and key not in node:
        node[key] = {}
    return node[key]


def parse_value(raw: str) -> Any:
    """YAML-parse an override value; numeric strings YAML 1.1 misses (``3e-4``) become floats."""
    import yaml

    if not raw.strip():
        return None
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        for cast in (int, float):
            try:
                return cast(value)
            except ValueError:
                continue
    return value


def load_config(
    path: str | os.PathLike | None = None,
    overrides: list[str] | None = None,
    base: dict[str, Any] | None = None,
) -> TrainConfig:
    """Load a YAML config (optional), merge ``base`` and dotted overrides, validate."""
    import yaml

    data: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(loaded, dict):
            raise TypeError(f"{path}: top level must be a mapping")
        data = loaded
    if base:
        data = _deep_merge(data, base)
    data = apply_overrides(data, list(overrides or []))
    return from_dict(TrainConfig, data)


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
