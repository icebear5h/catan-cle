"""CPU-only, function-preserving expansion of a standard Catan LoRA bundle.

No base model is loaded. Only adapter factors are materialized; the full FP32
visual sidefile and native tokenizer/processor assets are copied byte-for-byte.
The result is an initialization bundle, not an optimizer-resumable checkpoint.
"""

from __future__ import annotations

from ._bundle import expand_lora_bundle
from ._constants import (
    ADAPTER_FILE,
    CONFIG_FILE,
    EXPANSION_REPORT_FILE,
    INFERENCE_ASSETS,
    SCOPE_FILE,
    SUPPORTED_TEXT_LORA_RANKS,
    VISUAL_FILE,
)
from ._shapes import expected_adapter_shapes, standard_lora_rank, tensor_headers

__all__ = [
    "ADAPTER_FILE",
    "CONFIG_FILE",
    "EXPANSION_REPORT_FILE",
    "INFERENCE_ASSETS",
    "SCOPE_FILE",
    "SUPPORTED_TEXT_LORA_RANKS",
    "VISUAL_FILE",
    "expand_lora_bundle",
    "expected_adapter_shapes",
    "standard_lora_rank",
    "tensor_headers",
]
