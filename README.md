# lsdb-foundation-model

Tokenizing large astronomical catalogs with the [AION-1](https://github.com/PolymathicAI/AION)
codecs, starting from the Multimodal Universe / HATS catalogs hosted on Hugging Face.

## Legacy Survey image tokenization

`scripts/tokenize_legacysurvey.py` streams the
[`UniverseTBD/mmu_ssl_legacysurvey_north`](https://huggingface.co/datasets/UniverseTBD/mmu_ssl_legacysurvey_north)
dataset (14M galaxies with g, r, z cutouts), encodes every cutout into 576 discrete tokens with the
AION-1 image codec, and writes a parquet Hugging Face dataset with the same columns as the original
catalog, minus the `image` column, plus an `image_tokens` column.

```bash
uv sync                                   # create the environment
uv run python scripts/tokenize_legacysurvey.py --max-objects 100 --output data/tokenized_demo
```

The resulting parquet file can be reloaded with
`datasets.load_dataset("parquet", data_files="data/tokenized_demo/*.parquet")`.
Nothing is pushed to the Hub.

The same code, as a Colab walkthrough, lives in `notebooks/tokenize_legacysurvey.ipynb`
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/astronomy-commons/lsdb-foundation-model/blob/main/notebooks/tokenize_legacysurvey.ipynb).
The notebook is generated from the script with `uv run python scripts/build_notebook.py`.
