"""Tokenize Legacy Survey image cutouts with the AION-1 image codec.

Streams the ``UniverseTBD/mmu_ssl_legacysurvey_north`` Hugging Face dataset,
replaces the ``image`` column (g, r, z cutouts) by a column of discrete AION-1
image tokens, and writes the result back out as a parquet Hugging Face dataset
that keeps every other column of the original catalog untouched.

Example (demo on 100 objects, no upload):

    uv run python scripts/tokenize_legacysurvey.py --max-objects 100 --output data/tokenized_demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from aion.codecs import CodecManager
from aion.modalities import LegacySurveyImage
from datasets import Features, IterableDataset, Value, load_dataset
from tqdm.auto import tqdm

DATASET_ID = "UniverseTBD/mmu_ssl_legacysurvey_north"
IMAGE_COLUMN = "image"
TOKEN_COLUMN = "image_tokens"
NUM_IMAGE_TOKENS = LegacySurveyImage.num_tokens  # 576 = 24 x 24 tokens per cutout


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _aion_band_name(band: str) -> str:
    """Map the catalog band label (e.g. ``des-g``) to the AION label (``DES-G``)."""
    return band.upper()


def tokenize_images(
    flux: np.ndarray,
    bands: list[str],
    codec_manager: CodecManager,
    device: str | torch.device,
) -> np.ndarray:
    """Encode a batch of cutouts into AION-1 image tokens.

    Args:
        flux: array of shape ``(batch, num_bands, height, width)`` in nanomaggies.
        bands: AION band labels, one per channel (e.g. ``["DES-G", "DES-R", "DES-Z"]``).
        codec_manager: an AION ``CodecManager``; the image codec is downloaded on first use.
        device: device on which the codec runs.

    Returns:
        int64 array of shape ``(batch, NUM_IMAGE_TOKENS)``.
    """
    image = LegacySurveyImage(
        flux=torch.as_tensor(np.ascontiguousarray(flux, dtype=np.float32), device=device),
        bands=list(bands),
    )
    tokens = codec_manager.encode(image)[LegacySurveyImage.token_key]
    return tokens.detach().cpu().numpy().astype(np.int64)


def tokenize_batch(
    batch: dict[str, list],
    codec_manager: CodecManager,
    device: str | torch.device,
) -> dict[str, list]:
    """Turn one batch of catalog rows into rows with tokens instead of images.

    ``batch`` is a column-oriented dict as produced by ``IterableDataset.batch``.
    Every column except ``image`` is passed through unchanged.
    """
    images = batch[IMAGE_COLUMN]
    bands = [_aion_band_name(b) for b in images[0]["band"]]
    if any([_aion_band_name(b) for b in im["band"]] != bands for im in images):
        raise ValueError("All images in a batch must share the same band ordering")

    flux = np.stack([np.asarray(im["flux"], dtype=np.float32) for im in images])
    tokens = tokenize_images(flux, bands, codec_manager, device)

    out = {key: value for key, value in batch.items() if key != IMAGE_COLUMN}
    out[TOKEN_COLUMN] = tokens
    return out


def output_features(input_features: Features) -> Features:
    """Features of the tokenized dataset: the input ones minus ``image``, plus ``image_tokens``."""
    features = {k: v for k, v in input_features.items() if k != IMAGE_COLUMN}
    features[TOKEN_COLUMN] = [Value("int64")]
    return Features(features)


def open_dataset(dataset_id: str = DATASET_ID, split: str = "train") -> IterableDataset:
    """Open the catalog in streaming mode, so only the rows we consume are downloaded."""
    ds = load_dataset(dataset_id, split=split, streaming=True)
    if ds.features is None:  # features are resolved lazily for streaming parquet datasets
        ds = ds._resolve_features()
    return ds


def tokenize_dataset(
    ds: IterableDataset,
    codec_manager: CodecManager,
    batch_size: int = 32,
    max_objects: int | None = None,
    device: str | torch.device | None = None,
) -> Iterator[dict[str, list]]:
    """Yield tokenized batches (column-oriented dicts) from a streaming dataset."""
    device = device or _default_device()
    if max_objects is not None:
        ds = ds.take(max_objects)
    total = None if max_objects is None else -(-max_objects // batch_size)
    for batch in tqdm(ds.batch(batch_size), total=total, desc="Tokenizing", unit="batch"):
        yield tokenize_batch(batch, codec_manager, device)


def _batch_to_table(batch: dict[str, list], schema: pa.Schema) -> pa.Table:
    columns = {}
    for name in schema.names:
        value = batch[name]
        if name == TOKEN_COLUMN:
            tokens = np.asarray(value, dtype=np.int64)
            offsets = np.arange(0, tokens.size + 1, tokens.shape[1], dtype=np.int32)
            columns[name] = pa.ListArray.from_arrays(pa.array(offsets), pa.array(tokens.ravel()))
        else:
            columns[name] = pa.array(value, type=schema.field(name).type)
    return pa.Table.from_pydict(columns, schema=schema)


def write_parquet(
    batches: Iterable[dict[str, list]],
    features: Features,
    output_dir: str | Path,
    filename: str = "train-00000-of-00001.parquet",
) -> Path:
    """Write tokenized batches to a parquet file readable by ``datasets.load_dataset``.

    The Hugging Face ``features`` are embedded in the parquet metadata so that the
    dataset schema round-trips exactly.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    schema = features.arrow_schema.with_metadata(
        {"huggingface": json.dumps({"info": {"features": features.to_dict()}})}
    )
    n_rows = 0
    with pq.ParquetWriter(path, schema=schema) as writer:
        for batch in batches:
            table = _batch_to_table(batch, schema)
            writer.write_table(table)
            n_rows += table.num_rows
    print(f"Wrote {n_rows} rows to {path}")
    return path


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", default=DATASET_ID, help="Hugging Face dataset id to stream")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output", default="data/tokenized_demo", help="Output directory for parquet"
    )
    parser.add_argument(
        "--max-objects", type=int, default=100, help="Stop after this many objects (None = all)"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    args = parser.parse_args(argv)

    device = args.device or _default_device()
    print(f"Running image codec on {device}")

    codec_manager = CodecManager(device=device)
    ds = open_dataset(args.dataset, args.split)
    features = output_features(ds.features)

    start = time.time()
    batches = tokenize_dataset(
        ds, codec_manager, batch_size=args.batch_size, max_objects=args.max_objects, device=device
    )
    path = write_parquet(batches, features, args.output)
    print(f"Done in {time.time() - start:.1f}s")
    return path


if __name__ == "__main__":
    main()
    # The Hub streaming stack (datasets/fsspec) can abort the interpreter during
    # finalization once all work is done. Exit explicitly after flushing outputs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
