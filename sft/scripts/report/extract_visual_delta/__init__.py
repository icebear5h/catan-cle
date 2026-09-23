"""Extract a dense visual task vector and diagnostic per-matrix SVD factors.

This does not load a model, train, merge adapters, or replace checkpoint weights.
The base tensors are cast through the original loader dtype before subtraction.
Saved factors use delta ~= lora_B @ lora_A with multiplier 1; they are NOT a
drop-in PEFT adapter. Non-linear-layer tensors remain in the exact dense delta.
"""

from __future__ import annotations

from ._base import RANKS, THRESHOLDS
from ._extract import extract, main
from ._summaries import summarize, write_summary_tables
from ._tensors import (
    canonical_visual_key,
    decompose_matrix,
    loaded_base,
    matrix_kind,
    sha256_file,
    tensor_delta,
    visual_keys,
    write_json,
)

__all__ = [
    "RANKS",
    "THRESHOLDS",
    "canonical_visual_key",
    "decompose_matrix",
    "extract",
    "loaded_base",
    "main",
    "matrix_kind",
    "sha256_file",
    "summarize",
    "tensor_delta",
    "visual_keys",
    "write_json",
    "write_summary_tables",
]
