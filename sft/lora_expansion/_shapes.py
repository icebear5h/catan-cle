"""Shape and rank contracts shared by expansion and evaluation snapshots."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from sft.safetensor_types import TensorHeader, open_tensors

from ._constants import SUPPORTED_TEXT_LORA_RANKS


def _positive_finite(value: object) -> bool:
    # `type(...) in` rejects bool and numeric subclasses; `isinstance` narrows for mypy.
    return (
        type(value) in (int, float) and isinstance(value, int | float)
        and math.isfinite(value) and value > 0
    )


def standard_lora_rank(saved: Mapping[str, object], parent: Mapping[str, object]) -> int:
    """Validate the shared text/eval standard-LoRA rank and scaling contract."""
    rank, alpha = saved.get("r"), saved.get("lora_alpha")
    if (
        type(rank) is not int or rank not in SUPPORTED_TEXT_LORA_RANKS
        or not _positive_finite(alpha)
        or saved.get("peft_type") != "LORA" or saved.get("bias", "none") != "none"
        or any(saved.get(key) for key in (
            "use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "modules_to_save",
            "target_parameters", "layer_replication", "lora_bias", "fan_in_fan_out",
            "alora_invocation_tokens", "use_qalora", "arrow_config", "megatron_config",
        ))
        or parent.get("profile") != "vision_tokens_lora"
        or parent.get("lora_rank") != rank or parent.get("lora_alpha") != alpha
        or parent.get("frozen_bundle")
    ):
        raise ValueError("requires standard language rank-8 or rank-16 LoRA with matching rank/alpha metadata")
    return rank


def tensor_headers(path: str | Path) -> dict[str, TensorHeader]:
    with open_tensors(path, framework="pt", device="cpu") as handle:
        return {key: {"shape": handle.get_slice(key).get_shape(),
                      "dtype": handle.get_slice(key).get_dtype()} for key in handle.keys()}


def expected_adapter_shapes(
    linear_shapes: Mapping[str, SequenceShape], token_widths: Mapping[str, int], rank: int,
) -> dict[str, list[int]]:
    """PEFT's on-disk keys omit the runtime adapter name (``default``)."""
    expected: dict[str, list[int]] = {}
    for module, (output_width, input_width) in linear_shapes.items():
        expected[f"base_model.model.{module}.lora_A.weight"] = [rank, input_width]
        expected[f"base_model.model.{module}.lora_B.weight"] = [output_width, rank]
    for module, width in token_widths.items():
        expected[f"base_model.model.{module}.token_adapter.trainable_tokens_delta"] = [154, width]
    return expected


SequenceShape = tuple[int, int] | list[int]
