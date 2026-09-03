"""From lsdb partitions of AION tokens to masked training samples.

:func:`frame_to_tokens` turns the nested token columns written by ``aion-hats tokenize``
(``struct<token: list<int64>>``, loaded by lsdb as ``nested<token: [int64]>``, or plain
integer columns for single-token modalities) into dense ``(rows, num_tokens)`` arrays,
vectorised through Arrow. :class:`HatsTokenDataset` drives a :class:`PartitionStream`
per DataLoader worker, shuffles rows across partitions and applies the masker, yielding
the ``{modality: {tensor, input_mask, target_mask, decoder_attention_mask}}`` samples that
``torch.utils.data.default_collate`` turns into the batches ``FourM.forward`` expects.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data._utils.collate import default_collate

from .masking import UnifiedMasking
from .stream import PartitionStream, consumer_seed, deal_partitions, open_catalog_for_stream

if TYPE_CHECKING:
    from .config import DataConfig

log = logging.getLogger(__name__)

TOKEN_FIELD = "token"


def _column_to_arrow(series: Any, token_field: str = TOKEN_FIELD) -> pa.Array:
    """A token column as an Arrow ``list<int>`` (multi-token) or integer (scalar) array."""
    try:
        from nested_pandas import NestedDtype
    except ImportError:  # pragma: no cover - nested_pandas comes with lsdb
        NestedDtype = ()  # type: ignore[assignment]
    if isinstance(series.dtype, NestedDtype):
        series = series.nest.to_lists()[token_field]
    array = pa.array(series)
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if pa.types.is_struct(array.type):
        array = pc.struct_field(array, token_field)
    return array


def frame_to_tokens(
    frame: Any, modalities: dict[str, str], token_field: str = TOKEN_FIELD
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Dense token arrays of a partition.

    Returns ``(tokens, valid)`` where ``tokens[mod]`` is ``(n_rows, num_tokens)`` int64
    (zeros on invalid rows) and ``valid[mod]`` is the ``(n_rows,)`` boolean mask of rows
    whose value is not null (and, for lists, has the common length).
    """
    n = len(frame)
    tokens: dict[str, np.ndarray] = {}
    valid: dict[str, np.ndarray] = {}
    for mod, column in modalities.items():
        if column not in frame.columns:
            raise KeyError(f"column {column!r} for modality {mod!r} not in {list(frame.columns)}")
        array = _column_to_arrow(frame[column], token_field)
        ok = array.is_valid().to_numpy(zero_copy_only=False).astype(bool)
        if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
            lengths = pc.list_value_length(array).to_numpy(zero_copy_only=False)
            lengths = np.where(ok, lengths, -1)
            width = int(lengths[ok].max()) if ok.any() else 0
            ok &= lengths == width
            dense = np.zeros((n, width), dtype=np.int64)
            if ok.any():
                flat = pc.list_flatten(array.filter(pa.array(ok))).to_numpy(zero_copy_only=False)
                dense[ok] = np.asarray(flat, dtype=np.int64).reshape(-1, width)
        else:
            dense = np.zeros((n, 1), dtype=np.int64)
            if ok.any():
                values = array.filter(pa.array(ok)).to_numpy(zero_copy_only=False)
                dense[ok, 0] = np.asarray(values, dtype=np.int64)
        tokens[mod] = dense
        valid[mod] = ok
    return tokens, valid


class HatsTokenDataset(IterableDataset):
    """Infinite (or single-epoch) stream of masked token samples from a tokenized catalog.

    Args:
        source: catalog path, ``lsdb.Catalog`` or factory (see :func:`open_catalog_for_stream`).
        modalities: ``{aion token key: catalog column}``, e.g. ``{"tok_image": "tok_image"}``.
        masker: :class:`UnifiedMasking`; ``None`` yields ``{modality: (num_tokens,) long}``.
        split, filter: forwarded to :class:`PartitionStream`.
        shuffle: shuffle partitions, rows and use a cross-partition ``shuffle_buffer``.
        seed: base seed; every (rank, worker, epoch) derives its own.
        rank, world_size: data-parallel shard of this process; DataLoader workers further
            split the rank's share.
        drop_nulls: ``"all"`` keeps rows with at least one modality (missing ones become
            fully masked placeholders); ``"any"`` drops rows missing any modality.
        start_epoch: epoch of the first pass (resume); ``infinite`` keeps cycling epochs.
        prefetch, max_retries: forwarded to :class:`PartitionStream`.
    """

    def __init__(
        self,
        source: Any,
        modalities: dict[str, str],
        masker: UnifiedMasking | None = None,
        *,
        split: str | None = None,
        filter: dict[str, Any] | None = None,
        shuffle: bool = True,
        seed: int = 0,
        shuffle_buffer: int = 16384,
        rank: int = 0,
        world_size: int = 1,
        drop_nulls: str = "all",
        start_epoch: int = 0,
        infinite: bool = True,
        prefetch: int = 1,
        max_retries: int = 60,
        token_field: str = TOKEN_FIELD,
    ) -> None:
        super().__init__()
        if drop_nulls not in ("any", "all"):
            raise ValueError("drop_nulls must be 'any' or 'all'")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in [0, {world_size})")
        self.source = source
        self.modalities = dict(modalities)
        self.masker = masker
        self.split = split
        self.filter = filter
        self.shuffle = shuffle
        self.seed = seed
        self.shuffle_buffer = max(int(shuffle_buffer), 1)
        self.rank = rank
        self.world_size = world_size
        self.drop_nulls = drop_nulls
        self.epoch = start_epoch
        self.infinite = infinite
        self.prefetch = prefetch
        self.max_retries = max_retries
        self.token_field = token_field

    @property
    def columns(self) -> list[str]:
        return sorted(set(self.modalities.values()))

    def set_epoch(self, epoch: int) -> None:
        """Epoch of the next ``iter()``; a running iterator advances epochs by itself."""
        self.epoch = epoch

    def estimate_rows(self) -> int | None:
        """Rows in the catalog (scaled by the split's share of partitions); ``None`` if unknown."""
        catalog = open_catalog_for_stream(self.source, self.columns, self.filter)
        info = getattr(getattr(catalog, "hc_structure", None), "catalog_info", None)
        total = getattr(info, "total_rows", None)
        if total is None:
            return None
        pixels = catalog.get_healpix_pixels()
        stream = self._stream(0, 0, 1)
        share = len(stream.split_indices(pixels)) / max(len(pixels), 1)
        return int(total * share)

    def _stream(self, epoch: int, shard: int, num_shards: int) -> PartitionStream:
        return PartitionStream(
            self.source,
            columns=self.columns,
            filter=self.filter,
            split=self.split,
            seed=self.seed,
            epoch=epoch,
            shard=shard,
            num_shards=num_shards,
            shuffle=self.shuffle,
            prefetch=self.prefetch,
            max_retries=self.max_retries,
        )

    def _emit(self, sample: dict[str, np.ndarray]) -> dict[str, Any]:
        tensors = {mod: torch.from_numpy(np.ascontiguousarray(arr)) for mod, arr in sample.items()}
        return self.masker(tensors) if self.masker is not None else tensors

    def _rows(self, frame: Any) -> Iterator[dict[str, np.ndarray]]:
        tokens, valid = frame_to_tokens(frame, self.modalities, self.token_field)
        masks = np.stack(list(valid.values()), axis=0)  # (n_mod, n_rows)
        keep = masks.all(axis=0) if self.drop_nulls == "any" else masks.any(axis=0)
        for j in np.flatnonzero(keep):
            yield {mod: tokens[mod][j] for mod in self.modalities if valid[mod][j]}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        worker_id, num_workers = (info.id, info.num_workers) if info is not None else (0, 1)
        shard = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers
        epoch = self.epoch
        n_partitions: int | None = None
        empty_epochs = 0
        while True:
            if n_partitions is not None:
                # skip epochs in which this shard owns no partition without reopening the catalog
                while len(deal_partitions(n_partitions, self.seed, epoch, shard, num_shards, self.shuffle)) == 0:
                    epoch += 1
                self.epoch = epoch
            seed = consumer_seed(self.seed, self.rank, worker_id, epoch)
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            buffer: list[dict[str, np.ndarray]] = []
            yielded = 0
            stream = self._stream(epoch, shard, num_shards)
            for _pixel, frame in stream:
                for sample in self._rows(frame):
                    yielded += 1
                    if not self.shuffle or self.shuffle_buffer <= 1:
                        yield self._emit(sample)
                        continue
                    if len(buffer) < self.shuffle_buffer:
                        buffer.append(sample)
                        continue
                    k = int(rng.integers(len(buffer)))
                    sample, buffer[k] = buffer[k], sample
                    yield self._emit(sample)
            if buffer:
                for k in rng.permutation(len(buffer)):
                    yield self._emit(buffer[k])
            n_partitions = stream.n_in_split or 0
            if n_partitions == 0:
                raise ValueError(
                    f"catalog {self.source!r} has no partitions to stream (split={self.split!r}, filter={self.filter!r})"
                )
            empty_epochs = 0 if yielded else empty_epochs + 1
            if empty_epochs > num_shards:
                raise ValueError(
                    f"catalog {self.source!r} produced no usable rows for shard {shard} over "
                    f"{empty_epochs} epochs (all values null or filtered out?)"
                )
            if not self.infinite:
                return
            epoch += 1
            self.epoch = epoch


def build_dataloader(
    dataset: IterableDataset,
    cfg: DataConfig,
    batch_size: int,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """DataLoader over an iterable dataset with the source's collation (``default_collate``)."""
    device = torch.device(device) if device is not None else torch.device("cpu")
    workers = cfg.num_workers if num_workers is None else num_workers
    pin = cfg.pin_memory if cfg.pin_memory is not None else device.type == "cuda"
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin,
        "drop_last": True,
        "collate_fn": default_collate,
    }
    if workers > 0:
        kwargs.update(
            persistent_workers=True,
            prefetch_factor=cfg.prefetch_factor,
            multiprocessing_context=cfg.multiprocessing_context,
        )
    return DataLoader(dataset, **kwargs)
