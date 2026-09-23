"""Trainable scope audit and optimizer construction."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import TypedDict

import torch

from sft.json_types import JsonList, json_dict, json_list
from sft.scripts.train.train_trl_catan_vision._common import JsonDict, write_json_atomic
from sft.scripts.train.train_trl_catan_vision._config import (
    ModelComponents,
    TokenSetup,
    TrainConfig,
)
from sft.scripts.train.train_trl_catan_vision._structure import (
    _matches_path,
    resolve_wrapped_module,
)


class ScopeGroup(TypedDict):
    tensors: int
    parameters: int
    names: list[str]


def parameter_category(name: str, components: ModelComponents) -> str:
    if "trainable_tokens_delta" in name:
        if _matches_path(name, components.input_embedding):
            return "atlas_input_rows"
        if _matches_path(name, components.output_head):
            return "atlas_output_rows"
    if "lora_A" in name or "lora_B" in name:
        if _matches_path(name, components.language):
            return "language_lora"
        if _matches_path(name, components.vision):
            return "vision_lora"
        return "forbidden"
    if _matches_path(name, components.merger):
        return "merger"
    if _matches_path(name, components.vision):
        return "vision"
    return "forbidden"


def audit_trainable_scope(
    model: torch.nn.Module,
    components: ModelComponents,
    setup: TokenSetup,
    config: TrainConfig,
    output_path: Path,
) -> JsonDict:
    groups: dict[str, ScopeGroup] = {
        name: {"tensors": 0, "parameters": 0, "names": []}
        for name in (
            "vision",
            "merger",
            "atlas_input_rows",
            "atlas_output_rows",
            "language_lora",
            "vision_lora",
            "forbidden",
        )
    }
    parameters: JsonList = []
    errors: list[str] = []
    dtypes: dict[str, Counter[str]] = defaultdict(Counter)
    expected_delta = (154, components.hidden_size)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name, components)
        groups[category]["tensors"] += 1
        groups[category]["parameters"] += parameter.numel()
        groups[category]["names"].append(name)
        dtypes[category][str(parameter.dtype)] += 1
        parameters.append(
            {"name": name, "category": category, "shape": json_list(parameter.shape)}
        )
        if category.startswith("atlas_") and tuple(parameter.shape) != expected_delta:
            errors.append(f"{name} must have shape {expected_delta}")
        if config.text_only and category == "language_lora":
            rank_axis = 0 if ".lora_A." in name else 1
            if parameter.ndim != 2 or parameter.shape[rank_axis] != config.lora_rank:
                errors.append(f"{name} must have configured LoRA rank {config.lora_rank}")
    required_groups: tuple[str, ...]
    if config.text_only:
        required_groups = ("atlas_input_rows", "atlas_output_rows", "language_lora")
    elif config.olora:
        required_groups = ("atlas_input_rows", "atlas_output_rows", "language_lora", "vision_lora")
    else:
        required_groups = ("vision", "merger", "atlas_input_rows", "atlas_output_rows")
    for required in required_groups:
        if groups[required]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {required}")
    if config.text_only:
        for frozen in ("vision", "merger", "vision_lora"):
            if groups[frozen]["parameters"]:
                errors.append(f"{frozen} weights must stay frozen in text mode")
        visual = resolve_wrapped_module(model, components.vision)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in visual.state_dict().values()):
            errors.append("dormant visual state must stay FP32 in text mode")
    elif config.olora:
        for frozen in ("vision", "merger"):
            if groups[frozen]["parameters"]:
                errors.append(f"{frozen} weights must stay frozen under the olora profile")
    elif groups["vision_lora"]["parameters"]:
        errors.append("vision LoRA parameters need the olora profile")
    for category in ("atlas_input_rows", "atlas_output_rows"):
        if groups[category]["tensors"] != 1:
            errors.append(f"exactly one {category} tensor is required")
    if config.language_lora != bool(groups["language_lora"]["parameters"]):
        errors.append("language LoRA parameters disagree with the selected profile")
    if groups["forbidden"]["parameters"]:
        errors.append("base language or unknown parameters are trainable")
    for category in ("vision", "merger", "vision_lora"):
        if set(dtypes[category]) - {"torch.float32"}:
            errors.append(f"{category} trainable parameters must hold fp32 master weights")
    report: JsonDict = {
        "schema": "catan_trl_trainable_scope/v2",
        "profile": config.profile,
        "input_mode": config.input_mode,
        "components": components.as_dict(),
        "semantic_tokens": setup.as_dict(),
        "groups": {
            name: {"tensors": group["tensors"], "parameters": group["parameters"],
                   "names": json_list(group["names"])}
            for name, group in groups.items()
        },
        "dtypes": {name: json_dict(counter) for name, counter in dtypes.items()},
        "initialization": getattr(model, "_catan_initialization", None),
        "parameters": parameters,
        "errors": json_list(errors),
    }
    if config.text_only:
        report["lora"] = {
            "rank": config.lora_rank, "alpha": config.lora_alpha,
            "scaling": config.lora_alpha / config.lora_rank, "use_rslora": False,
        }
    write_json_atomic(output_path, report)
    if errors:
        raise RuntimeError("invalid trainable scope: " + "; ".join(errors))
    return report


def build_optimizer(
    model: torch.nn.Module,
    components: ModelComponents,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    grouped: dict[str, list[torch.nn.Parameter]] = {
        "vision": [],
        "merger": [],
        "token_rows": [],
        "language_lora": [],
        "vision_lora": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        category = parameter_category(name, components)
        if config.text_only and category not in {"atlas_input_rows", "atlas_output_rows", "language_lora"}:
            raise RuntimeError(f"text optimizer cannot contain {category}: {name}")
        if category == "vision":
            grouped["vision"].append(parameter)
        elif category == "merger":
            grouped["merger"].append(parameter)
        elif category in {"atlas_input_rows", "atlas_output_rows"}:
            grouped["token_rows"].append(parameter)
        elif category == "language_lora":
            grouped["language_lora"].append(parameter)
        elif category == "vision_lora":
            grouped["vision_lora"].append(parameter)
        else:
            raise RuntimeError(f"cannot route trainable parameter to optimizer: {name}")
    required: tuple[str, ...]
    if config.text_only:
        required = ("token_rows", "language_lora")
    elif config.olora:
        required = ("token_rows", "language_lora", "vision_lora")
    else:
        required = ("vision", "merger", "token_rows")
    if any(not grouped[name] for name in required):
        raise RuntimeError({name: len(values) for name, values in grouped.items()})
    if config.language_lora != bool(grouped["language_lora"]):
        raise RuntimeError("language LoRA optimizer group disagrees with the selected profile")
    optimizer_groups = [
        {
            "params": grouped["vision"],
            "lr": config.vision_learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "vision",
        },
        {
            "params": grouped["merger"],
            "lr": config.merger_learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "merger",
        },
        {
            "params": grouped["token_rows"],
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
            "catan_name": "token_rows",
        },
    ]
    optimizer_groups = [group for group in optimizer_groups if group["params"]]
    if grouped["vision_lora"]:
        optimizer_groups.append(
            {
                "params": grouped["vision_lora"],
                "lr": config.vision_lora_learning_rate,
                "weight_decay": config.weight_decay,
                "catan_name": "vision_lora",
            }
        )
    if grouped["language_lora"]:
        optimizer_groups.append(
            {
                "params": grouped["language_lora"],
                "lr": config.language_lora_learning_rate,
                "weight_decay": config.weight_decay,
                "catan_name": "language_lora",
            }
        )
    return torch.optim.AdamW(optimizer_groups)


def audit_optimizer_coverage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    output_path: Path,
) -> JsonDict:
    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    occurrences: dict[int, int] = {}
    groups: JsonList = []
    grouped_names: list[str] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            occurrences[parameter_id] = occurrences.get(parameter_id, 0) + 1
            names.append(trainable.get(parameter_id, f"<unknown:{parameter_id}>"))
        groups.append(
            {
                "name": group.get("catan_name"),
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameters": json_list(names),
            }
        )
        grouped_names.extend(names)
    missing = sorted(name for key, name in trainable.items() if key not in occurrences)
    duplicate = sorted(
        trainable.get(key, f"<unknown:{key}>")
        for key, count in occurrences.items()
        if count != 1
    )
    unknown = sorted(name for name in grouped_names if name.startswith("<unknown:"))
    errors: list[str] = []
    if missing:
        errors.append(f"missing trainable tensors: {missing[:8]}")
    if duplicate:
        errors.append(f"duplicate tensors: {duplicate[:8]}")
    if unknown:
        errors.append(f"unknown tensors: {unknown[:8]}")
    report: JsonDict = {
        "schema": "catan_trl_optimizer_coverage/v1",
        "trainable_tensors": len(trainable),
        "covered_tensors": len(occurrences),
        "groups": groups,
        "errors": json_list(errors),
    }
    write_json_atomic(output_path, report)
    if errors:
        raise RuntimeError("invalid optimizer coverage: " + "; ".join(errors))
    return report
