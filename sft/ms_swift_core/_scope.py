"""Audit which parameters a run actually trains, by category."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from sft.json_types import JsonList, json_list

from ._base import MS_SWIFT_VERSION, PEFT_VERSION, SCOPE_SCHEMA, JsonDict, ModelComponentPaths
from ._components import get_model_components
from ._coverage import defaultdict_group_manifest, write_json_atomic


def _module_path_matches(parameter_name: str, module_path: str) -> bool:
    normalized = f".{parameter_name}."
    target = f".{module_path}."
    return target in normalized


def _matches_any(parameter_name: str, module_paths: Iterable[str]) -> bool:
    return any(_module_path_matches(parameter_name, path) for path in module_paths)


def parameter_matches_paths(parameter_name: str, module_paths: Iterable[str]) -> bool:
    """Return whether a possibly PEFT-prefixed parameter belongs to a canonical path."""

    return _matches_any(parameter_name, module_paths)


def _parameter_category(
    name: str,
    *,
    components: ModelComponentPaths,
) -> str:
    if "trainable_tokens_delta" in name:
        is_input = _module_path_matches(name, components.input_embedding)
        is_output = _module_path_matches(name, components.output_head)
        if is_input and not is_output:
            return "atlas_input_rows"
        if is_output and not is_input:
            return "atlas_output_rows"
        return "unknown"
    if _module_path_matches(name, components.input_embedding):
        return "input_embedding"
    if _module_path_matches(name, components.output_head):
        return "output_head"
    if "lora_A" in name or "lora_B" in name:
        return "language_lora" if _matches_any(name, components.language) else "nonlanguage_lora"
    if _matches_any(name, components.aligner):
        return "aligner"
    if _matches_any(name, components.vision):
        return "vision"
    if _matches_any(name, components.language):
        return "base_language"
    return "unknown"


def audit_trainable_scope(
    model: torch.nn.Module,
    *,
    language_lora: bool,
    output_path: str | Path | None = None,
) -> JsonDict:
    """Fail closed unless only visual modules, LoRA, and two atlas deltas train."""

    components = get_model_components(model)
    setup = getattr(model, "_catan_semantic_token_setup", None)
    if setup is None or len(setup.token_ids) != 154:
        raise RuntimeError("model is missing the exact semantic token setup")

    groups = defaultdict_group_manifest()
    parameter_rows: JsonList = []
    trainable_parameters = 0
    errors: list[str] = []
    expected_delta_shape = (154, components.hidden_size)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = _parameter_category(name, components=components)
        count = parameter.numel()
        trainable_parameters += count
        groups[category]["tensors"] += 1
        groups[category]["parameters"] += count
        groups[category]["names"].append(name)
        parameter_rows.append(
            {
                "name": name,
                "category": category,
                "shape": json_list(parameter.shape),
                "dtype": str(parameter.dtype),
                "parameters": count,
            }
        )
        if category in ("atlas_input_rows", "atlas_output_rows") and tuple(
            parameter.shape
        ) != expected_delta_shape:
            errors.append(
                f"{category} delta must have shape {expected_delta_shape}: "
                f"{name} {tuple(parameter.shape)}"
            )

    required = ("vision", "aligner", "atlas_input_rows", "atlas_output_rows")
    for category in required:
        if groups[category]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {category}")
    for category in ("atlas_input_rows", "atlas_output_rows"):
        if groups[category]["tensors"] != 1:
            errors.append(f"exactly one {category} delta tensor is required")
    if language_lora and groups["language_lora"]["parameters"] == 0:
        errors.append("language LoRA profile has no trainable LoRA parameters")
    if not language_lora and groups["language_lora"]["parameters"]:
        errors.append("vision-only profile unexpectedly trains language LoRA")
    for category in (
        "input_embedding",
        "output_head",
        "base_language",
        "nonlanguage_lora",
        "unknown",
    ):
        if groups[category]["parameters"]:
            errors.append(f"forbidden trainable group is non-empty: {category}")

    group_manifest: JsonDict = {
        category: {
            "tensors": group["tensors"],
            "parameters": group["parameters"],
            "names": json_list(group["names"]),
        }
        for category, group in groups.items()
    }
    manifest: JsonDict = {
        "schema": SCOPE_SCHEMA,
        "ms_swift_version": MS_SWIFT_VERSION,
        "peft_version": PEFT_VERSION,
        "language_lora": language_lora,
        "model_components": components.as_dict(),
        "semantic_token_setup": setup.as_dict(),
        "groups": group_manifest,
        "parameters": parameter_rows,
        "trainable_tensors": len(parameter_rows),
        "trainable_parameters": trainable_parameters,
        "errors": json_list(errors),
    }
    if output_path is not None:
        write_json_atomic(Path(output_path), manifest)
    if errors:
        raise RuntimeError("invalid ms-swift trainable scope: " + "; ".join(errors))
    return manifest
