"""Shared fixtures: a synthetic MMU-style HATS catalog and a fake codec manager."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from aion.modalities import Image, Scalar, Spectrum

from aion_hats.catalog import Partition, finalize_catalog, format_properties

IMAGE_SHAPE = (3, 100, 100)  # >= 96 px so the real codec could crop it
SPECTRUM_LENGTH = 40
BANDS = ["des-g", "des-r", "des-z"]


def positions_in(partition: Partition | None, n: int, rng: np.random.Generator):
    """``(ra, dec, _healpix_29)`` of ``n`` objects inside ``partition`` (anywhere if ``None``)."""
    from hats.pixel_math.healpix_shim import radec2pix
    from hats.pixel_math.spatial_index import compute_spatial_index

    if partition is None:
        ra, dec = rng.uniform(0, 360, n), rng.uniform(-90, 90, n)
    else:
        from cdshealpix.nested import healpix_to_lonlat

        lon, lat = healpix_to_lonlat(np.array([partition.pixel]), partition.order)
        center = (float(lon.deg[0]), float(lat.deg[0]))
        size = 58.6 / 2**partition.order  # approximate pixel width in degrees
        ra, dec = np.empty(0), np.empty(0)
        while len(ra) < n:
            cand_ra = center[0] + rng.uniform(-0.3, 0.3, 4 * n) * size / max(np.cos(np.radians(center[1])), 0.1)
            cand_dec = center[1] + rng.uniform(-0.3, 0.3, 4 * n) * size
            cand_ra, cand_dec = cand_ra % 360, np.clip(cand_dec, -90, 90)
            inside = radec2pix(partition.order, cand_ra, cand_dec) == partition.pixel
            ra, dec = np.concatenate([ra, cand_ra[inside]])[:n], np.concatenate([dec, cand_dec[inside]])[:n]
    return ra, dec, compute_spatial_index(ra, dec)


def make_rows(n: int, seed: int, partition: Partition | None = None) -> pa.Table:
    """``n`` rows shaped like UniverseTBD/mmu_ssl_legacysurvey_north plus a spectrum.

    With ``partition`` the positions (and ``_healpix_29``) fall inside that HEALPix pixel,
    as in a real HATS catalog, so spatial searches behave.
    """
    rng = np.random.default_rng(seed)
    ra, dec, healpix_29 = positions_in(partition, n, rng)
    flux = rng.normal(size=(n, *IMAGE_SHAPE)).astype(np.float32)
    images = [
        {"band": BANDS, "flux": flux[i].tolist(), "psf_fwhm": [1.0, 1.1, 1.2], "scale": [0.262] * 3}
        for i in range(n)
    ]
    images[1] = None  # a missing image
    spec_flux = rng.normal(size=(n, SPECTRUM_LENGTH)).astype(np.float32)
    spectra = [
        {
            "flux": spec_flux[i].tolist(),
            "ivar": np.full(SPECTRUM_LENGTH, 2.0, np.float32).tolist(),
            "lambda": np.linspace(3600, 9800, SPECTRUM_LENGTH, dtype=np.float32).tolist(),
            "mask": [False] * SPECTRUM_LENGTH,
        }
        for i in range(n)
    ]
    flux_g = rng.uniform(0.1, 10, n).astype(np.float32)
    flux_g[0] = np.nan  # a missing scalar
    return pa.table(
        {
            "_healpix_29": pa.array(healpix_29, pa.int64()),
            "image": pa.array(images),
            "spectrum": pa.array(spectra),
            "ebv": pa.array(rng.uniform(0, 0.1, n).astype(np.float32)),
            "flux_g": pa.array(flux_g),
            "z_spec": pa.array(rng.uniform(0, 1, n).astype(np.float32)),
            "ra": pa.array(ra),
            "dec": pa.array(dec),
            "object_id": pa.array([f"obj{seed}_{i}" for i in range(n)]),
        }
    )


PARTITIONS = {Partition(4, 119): 10, Partition(5, 1239): 7, Partition(6, 5000): 5}


@pytest.fixture(scope="session")
def synthetic_catalog(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("catalogs") / "mmu_test_legacysurvey"
    dataset = root / "dataset"
    for seed, (partition, n_rows) in enumerate(PARTITIONS.items()):
        table = make_rows(n_rows, seed, partition)
        path = dataset / partition.relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, row_group_size=4)
    props = {
        "obs_collection": "mmu_test_legacysurvey",
        "dataproduct_type": "object",
        "hats_nrows": sum(PARTITIONS.values()),
        "hats_col_ra": "ra",
        "hats_col_dec": "dec",
        "hats_col_healpix": "_healpix_29",
        "hats_col_healpix_order": 29,
        "hats_max_rows": 8192,
        "hats_order": 6,
        "hats_skymap_order": 10,
        "hats_estsize": 1234,
    }
    (root / "hats.properties").write_text(format_properties(props))
    finalize_catalog(root, write_readme=False)  # partition_info.csv, _metadata, _common_metadata
    return root


class FakeCodecManager:
    """Deterministic stand-in for ``aion.codecs.CodecManager`` (no weights, no network)."""

    def __init__(self, device="cpu"):
        self.device = device
        self.calls: list[str] = []

    def encode(self, *modalities):
        out = {}
        for m in modalities:
            self.calls.append(type(m).__name__)
            n_tokens = type(m).num_tokens
            if isinstance(m, Image):
                key = m.flux.double().flatten(1).sum(1)
                assert m.flux.ndim == 4 and all("-" in b for b in m.bands)
            elif isinstance(m, Spectrum):
                key = torch.nan_to_num(m.flux).double().sum(1)
                assert m.mask.dtype == torch.bool and m.wavelength.shape == m.flux.shape
            elif isinstance(m, Scalar):
                key = m.value.double()
            else:
                raise TypeError(type(m))
            base = torch.round(key.abs() * 1000).long() % 4000
            tokens = (base[:, None] + torch.arange(n_tokens)) % 4096
            out[type(m).token_key] = tokens if n_tokens > 1 else tokens[:, 0]
        return out


@pytest.fixture
def fake_codecs() -> FakeCodecManager:
    return FakeCodecManager()


TOKEN_MODALITIES = ["image=LegacySurveyImage", "flux_g=LegacySurveyFluxG"]


@pytest.fixture(scope="session")
def tokenized_catalog(synthetic_catalog, tmp_path_factory) -> Path:
    """The synthetic catalog tokenized with the fake codec: ``tok_image`` (576 tokens, one
    null per partition) and ``tok_flux_g`` (scalar, one null per partition)."""
    from aion_hats import tokenize_catalog

    output = tmp_path_factory.mktemp("tokens") / "mmu_test_tokens"
    summary = tokenize_catalog(
        synthetic_catalog,
        output,
        TOKEN_MODALITIES,
        codec_manager=FakeCodecManager(),
        device="cpu",
        progress=False,
    )
    assert summary.ok and summary.rows == sum(PARTITIONS.values())
    return output


TINY_MODEL = {"dim": 64, "encoder_depth": 1, "decoder_depth": 1, "num_heads": 2}


def tiny_train_config(catalog: Path, output_dir: Path, **overrides):
    """A CPU-sized training config on ``catalog`` (2 epochs x 2 steps, batch 4)."""
    from aion_hats.train import load_config

    base = {
        "model": {"preset": "tiny", "overrides": dict(TINY_MODEL)},
        "data": {
            "datasets": [{"name": "train", "catalog": str(catalog), "modalities": ["tok_image"]}],
            "num_input_tokens": 64,
            "num_target_tokens": 32,
            "num_workers": 0,
            "shuffle_buffer": 8,
        },
        "optim": {"blr": 1e-2},
        "schedule": {"epochs": 2, "steps_per_epoch": 2, "warmup_steps": 1},
        "run": {
            "output_dir": str(output_dir),
            "batch_size": 4,
            "dtype": "float32",
            "device": "cpu",
            "print_freq": 1,
        },
    }
    return load_config(None, [f"{k}={v}" for k, v in overrides.items()], base=base)
