import numpy as np
import pytest
import torch
from conftest import PARTITIONS

lsdb = pytest.importorskip("lsdb")

from aion_hats.train import (
    DataConfig,
    HatsTokenDataset,
    UnifiedMasking,
    build_dataloader,
    frame_to_tokens,
    resolve_modality_info,
)
from aion_hats.train import data as data_module

TOTAL_ROWS = sum(PARTITIONS.values())
N_PARTITIONS = len(PARTITIONS)
MODALITIES = {"tok_image": "tok_image", "tok_flux_g": "tok_flux_g"}


def test_frame_to_tokens_dense_arrays_and_nulls(tokenized_catalog):
    frame = lsdb.open_catalog(tokenized_catalog, columns=list(MODALITIES.values())).compute()
    tokens, valid = frame_to_tokens(frame, MODALITIES)
    assert tokens["tok_image"].shape == (TOTAL_ROWS, 576) and tokens["tok_image"].dtype == np.int64
    assert tokens["tok_flux_g"].shape == (TOTAL_ROWS, 1)
    assert valid["tok_image"].sum() == TOTAL_ROWS - N_PARTITIONS  # one missing image per partition
    assert valid["tok_flux_g"].sum() == TOTAL_ROWS - N_PARTITIONS  # one NaN scalar per partition
    assert (tokens["tok_image"][~valid["tok_image"]] == 0).all()
    assert 0 <= tokens["tok_image"][valid["tok_image"]].min() and tokens["tok_image"].max() < 4096
    with pytest.raises(KeyError):
        frame_to_tokens(frame, {"tok_image": "missing"})


def _rows(dataset):
    return [{m: v.numpy().tolist() for m, v in s.items()} for s in dataset]


def test_dataset_yields_valid_rows_once_per_epoch(tokenized_catalog):
    both = HatsTokenDataset(tokenized_catalog, MODALITIES, None, infinite=False, shuffle_buffer=4)
    rows = _rows(both)
    assert len(rows) == TOTAL_ROWS - 2 * N_PARTITIONS  # rows with both modalities present
    assert all(len(r["tok_image"]) == 576 and len(r["tok_flux_g"]) == 1 for r in rows)
    image_only = HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=False)
    assert len(_rows(image_only)) == TOTAL_ROWS - N_PARTITIONS
    partial = HatsTokenDataset(tokenized_catalog, MODALITIES, None, infinite=False, drop_nulls="all")
    partial_rows = _rows(partial)
    assert len(partial_rows) == TOTAL_ROWS and sum("tok_image" not in r for r in partial_rows) == N_PARTITIONS


def test_dataset_sharding_covers_rows_exactly_once(tokenized_catalog, monkeypatch):
    reference = sorted(
        tuple(r["tok_image"]) for r in _rows(HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=False))
    )

    class Worker:
        def __init__(self, id, num_workers):
            self.id, self.num_workers = id, num_workers

    collected = []
    for rank in range(2):
        for worker in range(2):
            monkeypatch.setattr(data_module, "get_worker_info", lambda w=worker: Worker(w, 2))
            ds = HatsTokenDataset(
                tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=False, rank=rank, world_size=2
            )
            collected += [tuple(r["tok_image"]) for r in _rows(ds)]
    assert sorted(collected) == reference


def test_dataset_epochs_and_seeds(tokenized_catalog):
    def order(seed, start_epoch):
        ds = HatsTokenDataset(
            tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=False, seed=seed,
            start_epoch=start_epoch, shuffle_buffer=4,
        )
        return [r["tok_image"][0] for r in _rows(ds)]

    assert order(0, 0) == order(0, 0)
    assert order(0, 0) != order(0, 1) and order(0, 0) != order(1, 0)
    ds = HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=True, shuffle_buffer=4)
    it = iter(ds)
    first_epoch = [next(it) for _ in range(TOTAL_ROWS - N_PARTITIONS)]
    assert next(it) is not None and ds.epoch == 1  # rolled over into the next epoch
    assert len(first_epoch) == TOTAL_ROWS - N_PARTITIONS
    assert ds.estimate_rows() == TOTAL_ROWS
    assert HatsTokenDataset(tokenized_catalog, MODALITIES, split="val").estimate_rows() < TOTAL_ROWS


def test_dataset_more_shards_than_partitions(tokenized_catalog):
    # 3 partitions, 4 shards: shard 3 owns nothing at epoch 0 and must not spin or starve
    starved = HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, None, rank=3, world_size=4, infinite=False)
    assert _rows(starved) == []
    patient = HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, None, rank=3, world_size=4, infinite=True)
    it = iter(patient)
    assert next(it) is not None and patient.epoch == 1
    info = resolve_modality_info(["tok_image"], ["tok_image"])
    one_partition = lsdb.open_catalog(tokenized_catalog, columns=["tok_image"]).partitions[0]
    ds = HatsTokenDataset(one_partition, {"tok_image": "tok_image"}, UnifiedMasking(info, 64, 32), infinite=True)
    loader = build_dataloader(ds, DataConfig(num_workers=2, prefetch_factor=2), batch_size=2, device="cpu")
    batches = [b for b, _ in zip(loader, range(4))]
    assert len(batches) == 4 and batches[0]["tok_image"]["tensor"].shape == (2, 576)
    del loader
    # a stream that never produces a row must fail instead of spinning forever
    nothing = HatsTokenDataset(
        tokenized_catalog, {"tok_image": "tok_image"}, None, infinite=True,
        filter={"cone": {"ra": 0.0, "dec": -89.9, "radius_arcsec": 1.0}},
    )
    with pytest.raises(ValueError, match="no partitions|no usable rows"):
        next(iter(nothing))


def test_dataset_with_masker_and_dataloader_workers(tokenized_catalog):
    info = resolve_modality_info(["tok_image"], ["tok_image"])
    masker = UnifiedMasking(info, 64, 32)
    ds = HatsTokenDataset(tokenized_catalog, {"tok_image": "tok_image"}, masker, infinite=True)
    cfg = DataConfig(num_workers=2, prefetch_factor=2)
    loader = build_dataloader(ds, cfg, batch_size=4, device="cpu")
    batches = []
    for batch in loader:
        batches.append(batch)
        if len(batches) == 3:
            break
    batch = batches[0]["tok_image"]
    assert batch["tensor"].shape == (4, 576) and batch["tensor"].dtype == torch.long
    assert batch["input_mask"].shape == (4, 576) and batch["input_mask"].dtype == torch.bool
    assert batch["decoder_attention_mask"].shape == (4, 576)
    assert int((~batch["input_mask"]).sum(1).max()) <= 64
    del loader

    catalog = lsdb.open_catalog(tokenized_catalog, columns=["tok_image"])
    ds_from_catalog = HatsTokenDataset(catalog, {"tok_image": "tok_image"}, None, infinite=False)
    assert len(_rows(ds_from_catalog)) == TOTAL_ROWS - N_PARTITIONS
    with pytest.raises(ValueError):
        HatsTokenDataset(catalog, MODALITIES, drop_nulls="some")
