# lsdb-foundation-model

Tokenizing large astronomical catalogs with the [AION-1](https://github.com/PolymathicAI/AION)
codecs, starting from the [Multimodal Universe](https://github.com/MultimodalUniverse/MultimodalUniverse)
HATS catalogs hosted on Hugging Face (`UniverseTBD/mmu_*`).

This repository ships `aion-hats`, a small library and command line tool that turns any MMU
HATS catalog into a HATS catalog of AION tokens, and trains the AION transformer on the result
(see [Training on the tokens](#training-on-the-tokens)):

- **Input**: a HATS catalog on Hugging Face (`UniverseTBD/mmu_ssl_legacysurvey_north`), on disk,
  or behind any fsspec URL.
- **Output**: the same partitions (`Norder=k/Dir=d/Npix=p.parquet`), with the raw modality columns
  (images, spectra) replaced by token columns named after AION's token keys (`tok_image`,
  `tok_spectrum_desi`, `tok_flux_g`, ...). Multi-token columns are stored as
  `struct<token: list<int64>>`, which [`lsdb`](https://lsdb.io) loads as a nested column
  (`nested<token: [int64]>`, i.e. `df["tok_image.token"]`); single-token scalars are plain
  integers, and missing inputs become nulls. Every other column is kept, so the result opens with
  `lsdb` and joins back to its source on `_healpix_29` / `object_id`. It also
  loads with `datasets.load_dataset("parquet", data_files=".../dataset/**/*.parquet")` and can be uploaded to the
  Hub as is (a `README.md` with the dataset config is generated).
- **Scale-out**: partitions are the unit of work. Workers take a round-robin share, need no
  communication, write each partition atomically and skip partitions that are already done, so a
  job can be resumed by re-running it. One process per GPU (`srun`, `--num-procs`) is all it takes.

## Quick start

```bash
uv sync                      # environment (add --extra dev for tests, notebooks, lsdb)
uv run aion-hats inspect UniverseTBD/mmu_ssl_legacysurvey_north          # schema, detected modalities
uv run aion-hats tokenize UniverseTBD/mmu_ssl_legacysurvey_north data/tokenized_demo \
    --max-rows 100                                                       # 100-object demo
uv run aion-hats train --catalog data/tokenized_demo --output-dir runs/demo \
    --preset tiny --batch-size 16 --steps-per-epoch 5 --epochs 1        # tiny training demo
```

or from Python:

```python
from aion_hats import tokenize_catalog

summary = tokenize_catalog(
    "UniverseTBD/mmu_ssl_legacysurvey_north", "data/tokenized_demo", max_rows=100
)
```

By default every column AION has a codec for is tokenized: the image (`tok_image`), the
fluxes, extinction, redshift and position (`tok_flux_g`, `tok_ebv`, `tok_z`, `tok_ra`, ...);
`aion-hats inspect` lists what will be picked up, and values such as `-99` or NaN become null
tokens. `--modality` (or `modalities=`) restricts the run to a column name (`image`), an AION
modality (`LegacySurveyImage`) or an explicit pair (`z_spec=Z`). Images and spectra are dropped
from the output, scalars are kept next to their token.

The same walkthrough as a Colab notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astronomy-commons/lsdb-foundation-model/blob/main/notebooks/tokenize_legacysurvey.ipynb)
(`notebooks/tokenize_legacysurvey.ipynb`).

## Running the full catalog on several GPUs

The worker's rank and world size come from `SLURM_PROCID`/`SLURM_NTASKS`, `RANK`/`WORLD_SIZE`
(torchrun) or MPI variables, or from `--rank/--world-size`. Each worker uses the GPU matching its
local rank. On Perlmutter:

```bash
#!/bin/bash
#SBATCH -C gpu -N 4 --ntasks-per-node=4 --gpus-per-task=1 -c 32 -t 12:00:00
export HF_HOME=$SCRATCH/hf_cache                      # codec weights
srun aion-hats tokenize UniverseTBD/mmu_ssl_legacysurvey_north $SCRATCH/ls_north_tokens \
    --batch-size 256 --cache-dir $SCRATCH/ls_north_stage --num-prefetch 2
```

then, once every task has finished (re-submit the same job to resume after a time limit):

```bash
aion-hats finalize $SCRATCH/ls_north_tokens         # partition_info.csv, _metadata, hats_nrows
```

Notes:

- Remote partitions (about 2 GB each for Legacy Survey images) are downloaded in a background
  thread while the previous one is tokenized, and deleted afterwards; `--cache-dir` needs
  `(--num-prefetch + 1) x partition size` of free space per worker. With `--max-rows` (demos) row
  groups are streamed straight from the Hub instead. If the catalog is already on a parallel file
  system, pass its path and nothing is copied.
- On a single machine with several GPUs, `--num-procs 4` spawns one worker per GPU; it composes
  with `srun` (`srun -n 2 --ntasks-per-node=1 ... --num-procs 4` is 8 workers).
- A failing partition is logged and reported in the summary (non-zero exit code) without stopping
  the others; `--fail-fast` reverses that. Nothing is ever pushed to the Hub.
- `--token-dtype int32` halves the size of the token columns (every AION codebook fits).
- Set `HF_TOKEN` (a read token) for faster, rate-limit-free downloads from the Hub.

## Training on the tokens

`aion-hats train` trains the FourM transformer that ships in the `aion` package
(`aion.fourm.fm.FM`, the model behind `polymathic-ai/aion-base`) on a tokenized catalog, from
scratch (`--preset tiny|small|base|large|xlarge`) or starting from the released checkpoint
(`--init-from polymathic-ai/aion-base`). The batch-drawing logic is a port of the 4M / AION-1
training code (input/target token budgets sampled per example, masked-token cross-entropy)
with webdataset replaced by `lsdb`:

- **Data source**: the catalog is opened with `lsdb` inside every DataLoader worker; a path, an
  `lsdb.Catalog` (after a cone search, a crossmatch, a column selection) or a factory building one
  all work, and the filters are honoured. Partitions are permuted deterministically per epoch and
  dealt round-robin to ranks and workers, so every object is seen once per epoch; rows are shuffled
  within partitions and through a cross-partition buffer. Missing tokens (null rows) are skipped.
- **Splits**: `split: train|val` keeps partitions whose HEALPix order-4 ancestor hashes into the
  training or validation bucket, so the two sets are disjoint on the sky.
- **Scale-out**: one process per GPU (`srun`/`torchrun`); with more than one process the model is
  wrapped in FSDP (ZeRO-2, bf16 autocast) exactly as in the source; a single process (Colab, CPU
  smoke tests) runs unwrapped. Checkpoints (`checkpoint-<epoch>.pth`, resumed automatically) have
  the same layout in both cases, and the final model is exported to `<output_dir>/final` as
  `config.json` + `model.safetensors`, loadable with `AION.from_pretrained(...)`.

```bash
uv sync --extra train                                                  # lsdb + pyyaml (wandb: --extra wandb)
uv run aion-hats train -c configs/ls_north_base.yaml \
    --set data.datasets.0.catalog=$SCRATCH/ls_north_tokens --set run.output_dir=$SCRATCH/runs/base
```

Everything in the YAML (see `configs/`) can be overridden with `--set section.key=value`, and
the common knobs have flags (`--catalog`, `--preset`, `--init-from`, `--batch-size`, `--epochs`,
`--max-steps`, `--num-workers`, `--device`, `--resume`, `--wandb-project`). From Python:

```python
import lsdb
from aion_hats.train import Trainer, load_config

cfg = load_config("configs/ls_north_base.yaml", ["run.output_dir=runs/cone"])
catalog = lsdb.open_catalog("ls_north_tokens", columns=["tok_image"]).cone_search(ra=150, dec=2, radius_arcsec=3600)
Trainer(cfg, catalogs={"ls_north": catalog, "ls_north_val": catalog}).fit()
```

On Perlmutter (the learning rate scales as `blr * global_batch / 256`):

```bash
#SBATCH -C gpu -N 4 --ntasks-per-node=4 --gpus-per-task=1 -c 32 -t 12:00:00
export HF_HOME=$SCRATCH/hf_cache
srun aion-hats train -c configs/ls_north_base.yaml \
    --set data.datasets.0.catalog=$SCRATCH/ls_north_tokens --set run.output_dir=$SCRATCH/runs/base
```

Re-submitting the same job resumes from the last checkpoint. The `aion_hats.train.stream`
module (partition dealing, splits, retries) has no model dependency and is meant to converge with
the astroPT / [mmu-stream](https://github.com/Smith42/mmu-stream) loaders.

## Library layout

| Module | Role |
| --- | --- |
| `aion_hats.catalog` | open HATS catalogs (local, `hf://`, fsspec), list partitions, stream or download them, write and finalize the output catalog |
| `aion_hats.arrow_utils`, `aion_hats.iterutils` | Arrow helpers for nested columns; background prefetching |
| `aion_hats.distributed` | rank/GPU discovery from the environment, local multi-process launcher |
| `aion_hats.tokenize.modalities` | `ModalitySpec`, detection of tokenizable columns, adapters from Arrow struct columns to AION `Image`/`Spectrum`/`Scalar` modalities |
| `aion_hats.tokenize.tokenizer` | `AionTokenizer`: Arrow record batch in, record batch with token columns out (usable on its own, e.g. in `lsdb` `map_partitions`) |
| `aion_hats.tokenize.pipeline` | `tokenize_catalog`: sharding, prefetching, atomic partition writes, resume, summary |
| `aion_hats.train.stream` | model-agnostic lsdb partition stream: per-epoch dealing to ranks/workers, spatial splits, retries |
| `aion_hats.train.data`, `aion_hats.train.masking` | tokens to dense arrays, input/target masking (4M `UnifiedMasking`, Beta budgets), DataLoader |
| `aion_hats.train.model`, `aion_hats.train.trainer` | `FM` presets / pretrained init, FSDP wrapping, the training loop, checkpoints, export |
| `aion_hats.cli` | `aion-hats tokenize / finalize / inspect / train` |

Tests (`uv run pytest`) run on a synthetic HATS catalog with a fake codec and a dim-64 model, so
they need neither network access nor GPU.
