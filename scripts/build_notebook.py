"""Generate notebooks/tokenize_legacysurvey.ipynb from scripts/tokenize_legacysurvey.py.

The notebook is a Colab-friendly walkthrough whose code cells are the functions of
the script, extracted verbatim with ``inspect.getsource`` so the two never drift.

    uv run python scripts/build_notebook.py
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).parent))
import tokenize_legacysurvey as tk

REPO = "astronomy-commons/lsdb-foundation-model"
NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "tokenize_legacysurvey.ipynb"


def src(*objs) -> str:
    return "\n\n".join(inspect.getsource(o) for o in objs)


md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

cells = [
    md(
        f'<a href="https://colab.research.google.com/github/{REPO}/blob/main/notebooks/'
        'tokenize_legacysurvey.ipynb" target="_parent"><img src="https://colab.research.google.com/'
        'assets/colab-badge.svg" alt="Open In Colab"/></a>'
    ),
    md(
        "# Tokenizing Legacy Survey images with AION-1\n\n"
        "This notebook streams galaxy cutouts from the\n"
        f"[`{tk.DATASET_ID}`](https://huggingface.co/datasets/{tk.DATASET_ID}) Hugging Face dataset\n"
        "(the northern Legacy Surveys DR9 sample from the Multimodal Universe, 14M objects with\n"
        "g, r, z cutouts of 152x152 pixels), encodes each image into discrete tokens with the\n"
        "[AION-1](https://github.com/PolymathicAI/AION) image codec, and writes the result as a\n"
        "parquet Hugging Face dataset with the **same columns as the original catalog, minus the\n"
        "images, plus an `image_tokens` column**.\n\n"
        "The code below is the content of\n"
        f"[`scripts/tokenize_legacysurvey.py`](https://github.com/{REPO}/blob/main/scripts/"
        "tokenize_legacysurvey.py) in this repository; the notebook is generated from it.\n\n"
        "For the demo we only process 100 objects and do not push anything to the Hub.\n\n"
        "### Enabling GPU access\n\n"
        "The codec runs on CPU too, but on Colab go to `Runtime > Change runtime type` and select a\n"
        "GPU to speed things up.\n\n"
        "### Installing dependencies"
    ),
    code("!pip install --quiet --upgrade polymathic-aion datasets huggingface_hub pyarrow"),
    md(
        "## Step I: the tokenization function\n\n"
        "A batch of cutouts is a `(batch, bands, height, width)` array of fluxes in nanomaggies.\n"
        "We wrap it in AION's `LegacySurveyImage` modality, which tells the codec which survey\n"
        "and bands the pixels come from (the codec knows how to handle a missing `i` band), and\n"
        "the `CodecManager` downloads the image codec and turns each cutout into\n"
        f"{tk.NUM_IMAGE_TOKENS} integer tokens (a 24x24 grid over the central 96x96 pixels)."
    ),
    code(
        "import json\n"
        "import time\n"
        "from collections.abc import Iterable, Iterator\n"
        "from pathlib import Path\n\n"
        "import numpy as np\n"
        "import pyarrow as pa\n"
        "import pyarrow.parquet as pq\n"
        "import torch\n"
        "from aion.codecs import CodecManager\n"
        "from aion.modalities import LegacySurveyImage\n"
        "from datasets import Features, IterableDataset, Value, load_dataset\n"
        "from tqdm.auto import tqdm\n\n"
        f'DATASET_ID = "{tk.DATASET_ID}"\n'
        f'IMAGE_COLUMN = "{tk.IMAGE_COLUMN}"\n'
        f'TOKEN_COLUMN = "{tk.TOKEN_COLUMN}"\n'
        "NUM_IMAGE_TOKENS = LegacySurveyImage.num_tokens  # 576 = 24 x 24 tokens per cutout\n\n"
        + src(tk._default_device, tk._aion_band_name, tk.tokenize_images)
    ),
    md(
        "## Step II: streaming the catalog\n\n"
        "The dataset is 3.4 TiB, so we open it in streaming mode: only the parquet row groups we\n"
        "actually consume get downloaded. Each streamed batch is a column-oriented dict; we drop\n"
        "the `image` column, tokenize it, and pass every other column through untouched."
    ),
    code(src(tk.open_dataset, tk.tokenize_batch, tk.tokenize_dataset)),
    md(
        "## Step III: writing a parquet Hugging Face dataset\n\n"
        "The output features are the input ones without `image`, plus `image_tokens`. The features\n"
        "are embedded in the parquet metadata so `load_dataset` recovers the exact schema."
    ),
    code(src(tk.output_features, tk._batch_to_table, tk.write_parquet)),
    md("## Step IV: run it on 100 objects"),
    code(
        "MAX_OBJECTS = 100\n"
        "BATCH_SIZE = 32\n"
        'OUTPUT_DIR = "tokenized_demo"\n\n'
        "device = _default_device()\n"
        'print(f"Running image codec on {device}")\n\n'
        "codec_manager = CodecManager(device=device)\n"
        "ds = open_dataset(DATASET_ID)\n"
        "features = output_features(ds.features)\n"
        "print(features)\n\n"
        "start = time.time()\n"
        "batches = tokenize_dataset(ds, codec_manager, batch_size=BATCH_SIZE, max_objects=MAX_OBJECTS, device=device)\n"
        "path = write_parquet(batches, features, OUTPUT_DIR)\n"
        'print(f"Done in {time.time() - start:.1f}s")'
    ),
    md(
        "## Step V: check the result\n\n"
        "Reload the parquet file as a regular Hugging Face dataset and look at a row."
    ),
    code(
        'tokenized = load_dataset("parquet", data_files=str(path), split="train")\n'
        "print(tokenized)\n"
        "row = tokenized[0]\n"
        "print({k: v for k, v in row.items() if k != TOKEN_COLUMN})\n"
        'print("tokens:", len(row[TOKEN_COLUMN]), row[TOKEN_COLUMN][:16], "...")'
    ),
    md(
        "The tokens can be decoded back into a 96x96 cutout with the same codec, which is a\n"
        "useful sanity check of what information the tokenization retains."
    ),
    code(
        "import matplotlib.pyplot as plt\n\n"
        'bands = ["DES-G", "DES-R", "DES-Z"]\n'
        "tokens = torch.as_tensor(np.asarray(tokenized[:4][TOKEN_COLUMN]), device=device)\n"
        "reconstructed = codec_manager.decode({LegacySurveyImage.token_key: tokens}, LegacySurveyImage, bands=bands)\n\n"
        "fig, axes = plt.subplots(2, 4, figsize=(12, 6))\n"
        "for k in range(4):\n"
        '    original = next(iter(ds.skip(k).take(1)))["image"]["flux"]\n'
        '    axes[0, k].imshow(np.asarray(original)[1, 28:-28, 28:-28], cmap="gray")\n'
        "    axes[0, k].set_title(f\"{tokenized[k]['object_id']} (r, input)\")\n"
        '    axes[1, k].imshow(reconstructed.flux[k, 1].cpu().numpy(), cmap="gray")\n'
        '    axes[1, k].set_title("decoded from tokens")\n'
        "for ax in axes.ravel():\n"
        '    ax.axis("off")'
    ),
    md(
        "## Next steps\n\n"
        "- Remove `max_objects` to tokenize the full catalog (14M objects; use a GPU and a large\n"
        "  batch size, and shard the output).\n"
        "- Push the parquet directory to the Hub with `huggingface_hub.HfApi().upload_folder`."
    ),
]

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["colab"] = {"provenance": [], "gpuType": "T4"}
nb.metadata["accelerator"] = "GPU"
nbformat.write(nb, NOTEBOOK)
print(f"Wrote {NOTEBOOK}")
