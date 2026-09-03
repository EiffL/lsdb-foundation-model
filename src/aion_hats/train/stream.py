"""Deterministic, sharded streaming of HATS partitions through ``lsdb``.

This module is deliberately model agnostic (no torch, no AION imports): it turns an
``lsdb.Catalog`` (a path, an already filtered catalog or a factory that builds one) into a
stream of ``(HealpixPixel, NestedFrame)`` pairs for *one epoch of one shard*. The design
follows the practices of the astroPT ``nanotron_loader`` and ``mmu-stream``:

- the catalog is opened by the consumer (i.e. inside each DataLoader worker), and every
  user-side filter (cone search, column projection, ...) is honoured because each
  partition is computed from the catalog's own dask graph (``Catalog.to_delayed``);
- partitions are permuted deterministically per ``(seed, epoch)`` and dealt round-robin
  to ``num_shards`` consumers, so every row is seen exactly once per epoch across ranks
  and workers, and any consumer can be restarted independently;
- an optional spatial train/validation split at a fixed HEALPix order keeps the two sets
  disjoint on the sky;
- transient network errors are retried with exponential backoff, reopening the catalog;
- the next partition is fetched in a background thread while the current one is consumed.

``lsdb.streams.CatalogStream`` offers similar single-consumer semantics but, as of lsdb
0.10, fails on searched catalogs (``operation.build(pixels=[...])`` does not see the
search step); the contract of :class:`PartitionStream` is kept close to it so that an
lsdb-native sharded stream can replace it later.
"""

from __future__ import annotations

import logging
import os
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..iterutils import prefetch_iter

if TYPE_CHECKING:
    import lsdb
    from hats.pixel_math import HealpixPixel

log = logging.getLogger(__name__)

CatalogSource = "str | os.PathLike | lsdb.Catalog | Callable[[], lsdb.Catalog]"

SPLIT_ORDER = 4
SPLIT_BUCKETS = 20
RETRYABLE_MESSAGES = ("client has been closed",)


def consumer_seed(*parts: int | str) -> int:
    """Stable, independent seed for one consumer, e.g. ``(seed, rank, worker, epoch)``."""
    return zlib.crc32(":".join(str(p) for p in parts).encode())


def split_of_pixel(pixel: HealpixPixel, split_order: int = SPLIT_ORDER, buckets: int = SPLIT_BUCKETS) -> str:
    """Spatial ``"train"``/``"val"`` split of a partition at a fixed HEALPix order.

    The pixel's ancestor at ``split_order`` is hashed into ``buckets``; bucket 0 is the
    validation set, so validation and training partitions never share a sky cell of that
    order (the mmu-stream ``split_of_cell`` rule).
    """
    if pixel.order < split_order:
        raise ValueError(f"HEALPix order {pixel.order} is below the split order {split_order}")
    parent = pixel.pixel >> (2 * (pixel.order - split_order))
    return "val" if zlib.crc32(str(parent).encode()) % buckets == 0 else "train"


def deal_partitions(
    n: int, seed: int, epoch: int, shard: int = 0, num_shards: int = 1, shuffle: bool = True
) -> np.ndarray:
    """This shard's partition indices for ``epoch``: a shared permutation dealt round-robin.

    The deal rotates with the epoch (shard ``s`` takes offset ``(s + epoch) % num_shards``),
    so that with fewer partitions than shards every shard still gets data every
    ``num_shards`` epochs; within one epoch the shards are disjoint and complete.
    """
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} is not in [0, {num_shards})")
    order = np.random.default_rng([seed, epoch]).permutation(n) if shuffle else np.arange(n)
    return order[(shard + epoch) % num_shards :: num_shards]


def is_retryable(err: BaseException) -> bool:
    """Network-ish errors worth a retry (astroPT's ``_retryable``)."""
    if isinstance(err, (OSError, TimeoutError)):
        return True
    try:
        import httpx

        if isinstance(err, httpx.HTTPError):
            return True
    except ImportError:
        pass
    return isinstance(err, RuntimeError) and any(m in str(err) for m in RETRYABLE_MESSAGES)


def apply_filter(catalog: lsdb.Catalog, spec: dict[str, Any] | None) -> lsdb.Catalog:
    """Apply a declarative filter: ``{"cone": {ra, dec, radius_arcsec}}`` or ``{"box": {ra, dec}}``."""
    if not spec:
        return catalog
    for kind, params in spec.items():
        if kind == "cone":
            catalog = catalog.cone_search(**params)
        elif kind == "box":
            catalog = catalog.box_search(**params)
        else:
            raise ValueError(f"unknown catalog filter {kind!r} (expected 'cone' or 'box')")
    return catalog


def open_catalog_for_stream(
    source: Any, columns: Sequence[str] | None = None, filter: dict[str, Any] | None = None
) -> lsdb.Catalog:
    """Resolve a :data:`CatalogSource` into an ``lsdb.Catalog``.

    A path (local, ``hf://`` or ``org/name``) is opened with ``columns`` projected; a
    callable is invoked (per consumer) and an ``lsdb.Catalog`` is used as is, so that
    whatever the caller did to it (cone search, crossmatch, projection) is preserved.
    """
    import lsdb

    from ..catalog import _normalize_source

    if isinstance(source, (str, os.PathLike)):
        try:
            catalog = lsdb.open_catalog(_normalize_source(source), columns=list(columns) if columns else None)
        except KeyError as err:  # lsdb reports unknown columns as a KeyError
            raise ValueError(f"catalog {source} has no columns {columns}: {err}") from err
    elif callable(source) and not isinstance(source, lsdb.Catalog):
        catalog = source()
    else:
        catalog = source
    if not isinstance(catalog, lsdb.Catalog):
        raise TypeError(f"expected an lsdb.Catalog, a path or a factory, got {type(source).__name__}")
    if columns:
        missing = set(columns) - set(catalog.columns)
        if missing:
            raise ValueError(f"catalog {catalog_name(catalog)!r} has no columns {sorted(missing)}")
    return apply_filter(catalog, filter)


def catalog_name(catalog: lsdb.Catalog) -> str:
    info = getattr(getattr(catalog, "hc_structure", None), "catalog_info", None)
    return str(getattr(info, "catalog_name", None) or "catalog")


def catalog_provenance(catalog: lsdb.Catalog) -> dict[str, str]:
    info = getattr(getattr(catalog, "hc_structure", None), "catalog_info", None)
    out = {}
    for key in ("catalog_name", "catalog_type", "hats_builder", "hats_version", "total_rows"):
        value = getattr(info, key, None)
        if value is not None:
            out[key] = str(value)
    return out


class PartitionStream:
    """One epoch of this shard's partitions of a catalog, one ``NestedFrame`` at a time.

    Args:
        source: path, ``lsdb.Catalog`` or zero-argument factory (see
            :func:`open_catalog_for_stream`).
        columns: columns to project when ``source`` is a path; checked otherwise.
        filter: declarative filter applied after opening (``{"cone": {...}}``).
        split: ``"train"`` / ``"val"`` to keep only that spatial split, ``None`` for all.
        seed, epoch: define the partition permutation, shared by all shards.
        shard, num_shards: which round-robin share of the permutation this consumer owns.
        shuffle: permute partitions and the rows within each partition.
        prefetch: partitions computed ahead in a background thread (0 disables).
        max_retries, max_retry_wait: retry policy for transient network errors.
    """

    def __init__(
        self,
        source: Any,
        *,
        columns: Sequence[str] | None = None,
        filter: dict[str, Any] | None = None,
        split: str | None = None,
        split_order: int = SPLIT_ORDER,
        split_buckets: int = SPLIT_BUCKETS,
        seed: int = 0,
        epoch: int = 0,
        shard: int = 0,
        num_shards: int = 1,
        shuffle: bool = True,
        prefetch: int = 1,
        max_retries: int = 60,
        max_retry_wait: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if split not in (None, "train", "val"):
            raise ValueError("split must be 'train', 'val' or None")
        if not 0 <= shard < num_shards:
            raise ValueError(f"shard {shard} is not in [0, {num_shards})")
        self.source = source
        self.columns = list(columns) if columns else None
        self.filter = filter
        self.split = split
        self.split_order = split_order
        self.split_buckets = split_buckets
        self.seed = seed
        self.epoch = epoch
        self.shard = shard
        self.num_shards = num_shards
        self.shuffle = shuffle
        self.prefetch = prefetch
        self.max_retries = max_retries
        self.max_retry_wait = max_retry_wait
        self._sleep = sleep
        self.n_in_split: int | None = None  # set once the catalog has been opened
        self.n_owned: int | None = None

    # --- planning ----------------------------------------------------------------------

    def open(self) -> lsdb.Catalog:
        return open_catalog_for_stream(self.source, self.columns, self.filter)

    def split_indices(self, pixels: Sequence[HealpixPixel]) -> list[int]:
        if self.split is None:
            return list(range(len(pixels)))
        return [
            i
            for i, p in enumerate(pixels)
            if split_of_pixel(p, self.split_order, self.split_buckets) == self.split
        ]

    def plan(self, catalog: lsdb.Catalog | None = None) -> tuple[list[HealpixPixel], list[int]]:
        """``(all pixels, indices owned by this shard this epoch)``."""
        catalog = self.open() if catalog is None else catalog
        pixels = list(catalog.get_healpix_pixels())
        in_split = self.split_indices(pixels)
        if pixels and not in_split:
            log.warning("catalog %s has no partitions in split %r", catalog_name(catalog), self.split)
        dealt = deal_partitions(
            len(in_split), self.seed, self.epoch, self.shard, self.num_shards, self.shuffle
        )
        return pixels, [in_split[j] for j in dealt]

    def owned_pixels(self) -> list[HealpixPixel]:
        pixels, owned = self.plan()
        return [pixels[i] for i in owned]

    # --- iteration ---------------------------------------------------------------------

    def _retry_wait(self, failures: int) -> float:
        return min(5.0 * 2 ** (failures - 1), self.max_retry_wait)

    def _open_with_retry(self) -> lsdb.Catalog:
        failures = 0
        while True:
            try:
                return self.open()
            except Exception as err:
                failures += 1
                if not is_retryable(err) or failures > self.max_retries:
                    raise
                wait = self._retry_wait(failures)
                log.warning(
                    "%s while opening the catalog; retrying in %.0fs (%d/%d)",
                    type(err).__name__, wait, failures, self.max_retries,
                )
                self._sleep(wait)

    def _frames(self) -> Iterator[tuple[HealpixPixel, Any]]:
        catalog = self._open_with_retry()
        pixels, owned = self.plan(catalog)
        self.n_in_split = len(self.split_indices(pixels))
        self.n_owned = len(owned)
        self._log_provenance(catalog, len(pixels), len(owned))
        rng = np.random.default_rng(consumer_seed(self.seed, self.epoch, self.shard, self.num_shards))
        delayed = None
        failures = 0
        pos = 0
        while pos < len(owned):
            index = owned[pos]
            try:
                if delayed is None:
                    delayed = catalog.to_delayed()
                frame = delayed[index].compute(scheduler="synchronous")
            except Exception as err:
                failures += 1
                if not is_retryable(err) or failures > self.max_retries:
                    raise
                wait = self._retry_wait(failures)
                log.warning(
                    "%s while reading partition %s; reopening the catalog in %.0fs (%d/%d)",
                    type(err).__name__, pixels[index], wait, failures, self.max_retries,
                )
                self._sleep(wait)
                catalog = self._open_with_retry()
                delayed = None
                continue
            failures = 0
            if self.shuffle and len(frame) > 1:
                frame = frame.iloc[rng.permutation(len(frame))]
            yield pixels[index], frame
            pos += 1

    def __iter__(self) -> Iterator[tuple[HealpixPixel, Any]]:
        frames = self._frames()
        if self.prefetch > 0:
            frames = prefetch_iter(frames, depth=self.prefetch)
        yield from frames

    def _log_provenance(self, catalog: lsdb.Catalog, n_pixels: int, n_owned: int) -> None:
        try:
            from importlib.metadata import version

            lsdb_version = version("lsdb")
        except Exception:  # noqa: BLE001
            lsdb_version = "unknown"
        log.info(
            "stream %s: lsdb=%s columns=%s split=%s epoch=%d shard=%d/%d partitions=%d/%d %s",
            catalog_name(catalog), lsdb_version, list(catalog.columns), self.split, self.epoch,
            self.shard, self.num_shards, n_owned, n_pixels, catalog_provenance(catalog),
        )
