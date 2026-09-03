"""Tokenize HATS catalogs with the AION-1 codecs: modality adapters, the Arrow-level
tokenizer and the partition pipeline."""

from .modalities import MODALITY_REGISTRY, ModalitySpec, detect_modalities, resolve_modalities
from .pipeline import RunSummary, tokenize_catalog, tokenize_partition
from .tokenizer import AionTokenizer

__all__ = [
    "MODALITY_REGISTRY",
    "AionTokenizer",
    "ModalitySpec",
    "RunSummary",
    "detect_modalities",
    "resolve_modalities",
    "tokenize_catalog",
    "tokenize_partition",
]
