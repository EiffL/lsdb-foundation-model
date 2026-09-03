from itertools import chain

import numpy as np
import pytest
from conftest import PARTITIONS

lsdb = pytest.importorskip("lsdb")

from aion_hats.train.stream import (
    PartitionStream,
    consumer_seed,
    deal_partitions,
    is_retryable,
    open_catalog_for_stream,
    split_of_pixel,
)

TOTAL_ROWS = sum(PARTITIONS.values())


def test_deal_partitions_covers_every_index_once():
    n = 17
    for num_shards in (1, 2, 5):
        dealt = [deal_partitions(n, seed=3, epoch=0, shard=s, num_shards=num_shards) for s in range(num_shards)]
        assert sorted(np.concatenate(dealt).tolist()) == list(range(n))
        assert max(len(d) for d in dealt) - min(len(d) for d in dealt) <= 1
    assert not np.array_equal(deal_partitions(n, 3, 0), deal_partitions(n, 3, 1))
    assert np.array_equal(deal_partitions(n, 3, 4), deal_partitions(n, 3, 4))
    assert deal_partitions(n, 3, 0, shuffle=False).tolist() == list(range(n))
    # fewer partitions than shards: the deal rotates so every shard gets a turn
    turns = [len(deal_partitions(1, 0, epoch, shard=1, num_shards=2)) for epoch in range(4)]
    assert turns == [0, 1, 0, 1]
    with pytest.raises(ValueError):
        deal_partitions(n, 3, 0, shard=2, num_shards=2)


def test_split_of_pixel_is_spatial_and_deterministic():
    from hats.pixel_math import HealpixPixel

    parent = HealpixPixel(4, 119)
    children = [HealpixPixel(6, (119 << 4) + k) for k in range(16)]
    splits = {split_of_pixel(c) for c in children}
    assert splits == {split_of_pixel(parent)}  # all children follow their order-4 ancestor
    pixels = [HealpixPixel(5, p) for p in range(2000)]
    val = sum(split_of_pixel(p) == "val" for p in pixels)
    assert 0.02 < val / len(pixels) < 0.1  # one bucket in twenty
    with pytest.raises(ValueError):
        split_of_pixel(HealpixPixel(2, 1))


def test_consumer_seed_and_retryable():
    assert consumer_seed(0, 1, 2, 3) != consumer_seed(0, 1, 2, 4)
    assert consumer_seed("a", 1) == consumer_seed("a", 1)
    assert is_retryable(OSError("boom")) and is_retryable(TimeoutError())
    assert is_retryable(RuntimeError("the client has been closed"))
    assert not is_retryable(ValueError("bad")) and not is_retryable(RuntimeError("other"))


def test_stream_yields_each_owned_partition_once(tokenized_catalog):
    streams = [
        PartitionStream(tokenized_catalog, columns=["tok_image"], seed=1, epoch=0, shard=s, num_shards=2)
        for s in range(2)
    ]
    seen = {}
    for stream in streams:
        for pixel, frame in stream:
            assert "tok_image" in frame.columns and frame.index.name == "_healpix_29"
            seen[(pixel.order, pixel.pixel)] = len(frame)
    assert seen == {(p.order, p.pixel): n for p, n in PARTITIONS.items()}
    owned = [tuple((p.order, p.pixel) for p in s.owned_pixels()) for s in streams]
    assert len(owned[0]) + len(owned[1]) == len(PARTITIONS) and not set(owned[0]) & set(owned[1])


def test_stream_shuffles_rows_deterministically(tokenized_catalog):
    def rows(seed, epoch, shuffle=True):
        stream = PartitionStream(tokenized_catalog, columns=["object_id"], seed=seed, epoch=epoch, shuffle=shuffle)
        return [frame["object_id"].tolist() for _, frame in stream]

    assert rows(0, 0) == rows(0, 0)
    assert rows(0, 0) != rows(0, 1)
    ordered = rows(0, 0, shuffle=False)
    assert [sorted(r) for r in ordered] == ordered  # object ids are written in order
    assert sorted(chain.from_iterable(rows(0, 0))) == sorted(chain.from_iterable(ordered))


def test_stream_accepts_catalog_and_factory_and_filters(tokenized_catalog):
    catalog = lsdb.open_catalog(tokenized_catalog, columns=["tok_image", "ra", "dec"])
    cone = catalog.cone_search(ra=180.0, dec=0.0, radius_arcsec=3600 * 90)
    from_cone = sum(len(f) for _, f in PartitionStream(cone, prefetch=0))
    everything = sum(len(f) for _, f in PartitionStream(catalog))
    assert everything == TOTAL_ROWS and 0 < from_cone < TOTAL_ROWS

    calls = []

    def factory():
        calls.append(1)
        return catalog

    assert sum(len(f) for _, f in PartitionStream(factory)) == TOTAL_ROWS and calls == [1]
    filtered = PartitionStream(
        tokenized_catalog, columns=["tok_image"], filter={"cone": {"ra": 180.0, "dec": 0.0, "radius_arcsec": 3600 * 90}}
    )
    assert sum(len(f) for _, f in filtered) == from_cone
    with pytest.raises(ValueError, match="no columns"):
        open_catalog_for_stream(tokenized_catalog, columns=["nope"])
    with pytest.raises(ValueError, match="unknown catalog filter"):
        open_catalog_for_stream(tokenized_catalog, filter={"weird": {}})
    with pytest.raises(TypeError):
        open_catalog_for_stream(42)


def test_stream_split_partitions_are_disjoint(tokenized_catalog):
    train = PartitionStream(tokenized_catalog, split="train").owned_pixels()
    val = PartitionStream(tokenized_catalog, split="val").owned_pixels()
    assert len(train) + len(val) == len(PARTITIONS) and not set(train) & set(val)


def test_stream_retries_transient_errors(tokenized_catalog, monkeypatch):
    catalog = lsdb.open_catalog(tokenized_catalog, columns=["tok_image"])
    attempts = {"open": 0, "read": 0}
    waits = []

    def flaky_factory():
        attempts["open"] += 1
        if attempts["open"] == 1:
            raise OSError("network down")
        return catalog

    original = lsdb.Catalog.to_delayed

    def flaky_to_delayed(self, *args, **kwargs):
        attempts["read"] += 1
        if attempts["read"] == 1:
            raise TimeoutError("slow hub")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(lsdb.Catalog, "to_delayed", flaky_to_delayed)
    stream = PartitionStream(flaky_factory, prefetch=0, sleep=waits.append, max_retries=5)
    assert sum(len(f) for _, f in stream) == TOTAL_ROWS
    assert attempts == {"open": 3, "read": 2} and waits == [5.0, 5.0]

    def broken_factory():
        raise ValueError("not retryable")

    with pytest.raises(ValueError, match="not retryable"):
        list(PartitionStream(broken_factory, sleep=waits.append))

    def always_down():
        raise OSError("down")

    with pytest.raises(OSError):
        list(PartitionStream(always_down, sleep=waits.append, max_retries=2, max_retry_wait=7))
    assert waits[-2:] == [5.0, 7.0]
