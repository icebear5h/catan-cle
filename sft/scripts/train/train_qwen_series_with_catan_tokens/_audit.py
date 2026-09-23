"""audit."""

from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypedDict

import torch

from sft.qwen_series_vision_sft import (
    VISION_LANGUAGE_LORA,
    VISION_ONLY,
)


class TrainableGroup(TypedDict):
    parameters: int
    tensors: int
    names: list[str]


class TrainableScopeSummary(TypedDict):
    schema: str
    profile: str
    total_parameters: int
    total_trainable_parameters: int
    trainable_fraction: float
    tie_word_embeddings: bool
    trainable_token_count: int
    trainable_tokens: list[str]
    groups: dict[str, TrainableGroup]
    errors: list[str]


class TrainerInit(Protocol):
    """An upstream trainer `__init__` looked up on the class (unbound)."""

    def __call__(self, trainer: object, /, *args: object, **kwargs: object) -> None: ...


def _empty_group() -> TrainableGroup:
    return {"parameters": 0, "tensors": 0, "names": []}


def _trainable_parameter_group(name: str) -> str:
    lowered = name.lower()
    if "token_adapter" in lowered or "trainable_tokens" in lowered:
        return "catan_token_rows"
    if "lora_" in lowered:
        if "visual" in lowered:
            return "vision_lora"
        return "language_lora"
    if "visual" in lowered and "merger" in lowered:
        return "merger"
    if "visual" in lowered:
        return "vision_tower"
    if any(part in lowered for part in ("language_model", "lm_head", "embed_tokens")):
        return "language_base"
    return "other"


def _audit_trainable_parameters(
    model: torch.nn.Module,
    profile: str,
    output_dir: Path,
    catan_tokens: list[str] | None = None,
) -> TrainableScopeSummary:
    groups: dict[str, TrainableGroup] = {}
    total_parameters = 0
    total_trainable = 0
    for name, parameter in model.named_parameters():
        count = parameter.numel()
        total_parameters += count
        if not parameter.requires_grad:
            continue
        total_trainable += count
        group_name = _trainable_parameter_group(name)
        group = groups.setdefault(group_name, _empty_group())
        group["parameters"] += count
        group["tensors"] += 1
        if len(group["names"]) < 20:
            group["names"].append(name)

    for group_name in (
        "vision_tower",
        "merger",
        "catan_token_rows",
        "language_lora",
        "vision_lora",
        "language_base",
        "other",
    ):
        groups.setdefault(group_name, _empty_group())

    errors: list[str] = []
    for required_group in ("vision_tower", "merger", "catan_token_rows"):
        if groups[required_group]["parameters"] == 0:
            errors.append(f"required trainable group is empty: {required_group}")
    if groups["vision_lora"]["parameters"]:
        errors.append("vision LoRA is not part of the full-tower profiles")
    if groups["language_base"]["parameters"]:
        errors.append("base language parameters must remain frozen")
    if groups["other"]["parameters"]:
        errors.append("unclassified trainable parameters are present")
    if profile == VISION_ONLY and groups["language_lora"]["parameters"]:
        errors.append("vision_only must not contain language LoRA parameters")
    if profile == VISION_LANGUAGE_LORA and not groups["language_lora"]["parameters"]:
        errors.append("vision_language_lora requires language LoRA parameters")

    token_names = groups["catan_token_rows"]["names"]
    if not getattr(model.config, "tie_word_embeddings", False):
        if not any("lm_head" in name for name in token_names):
            errors.append("untied language output token rows are not trainable")

    summary: TrainableScopeSummary = {
        "schema": "catan_qwen_trainable_parameters/v1",
        "profile": profile,
        "total_parameters": total_parameters,
        "total_trainable_parameters": total_trainable,
        "trainable_fraction": total_trainable / total_parameters if total_parameters else 0.0,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "trainable_token_count": len(catan_tokens or []),
        "trainable_tokens": list(catan_tokens or []),
        "groups": groups,
        "errors": errors,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "trainable_parameters.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if errors:
        raise RuntimeError("Unsafe trainable parameter scope: " + "; ".join(errors))
    return summary


def _patch_trainer_scope_audit(
    upstream: ModuleType,
    profile: str,
    output_dir: Path,
    catan_tokens: list[str],
) -> None:
    original_trainer: type = upstream.QwenSFTTrainer

    # The upstream trainer class is only known at runtime, so the subclass is
    # built with `type()`. The next `__init__` in its MRO is the upstream one,
    # which is what `super().__init__` resolved to in a class statement.
    original_init: TrainerInit = getattr(original_trainer, "__init__")

    def __init__(self: object, *args: object, **kwargs: object) -> None:
        model = kwargs.get("model")
        if model is None and args:
            model = args[0]
        if model is None:
            raise RuntimeError("Could not inspect trainer model before optimization")
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"Trainer model is not a torch module: {type(model).__name__}")
        _audit_trainable_parameters(model, profile, output_dir, catan_tokens)
        original_init(self, *args, **kwargs)

    audited_trainer = type(
        "AuditedQwenSFTTrainer",
        (original_trainer,),
        {"__init__": __init__},
    )
    setattr(upstream, "QwenSFTTrainer", audited_trainer)
