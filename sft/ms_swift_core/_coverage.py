"""Optimizer-coverage audit and atomic receipt writes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import torch

from sft.json_types import JsonLike, JsonList, json_list

from ._base import OPTIMIZER_COVERAGE_SCHEMA, JsonDict


class ScopeGroup(TypedDict):
    """Running trainable-tensor totals for one parameter category."""

    tensors: int
    parameters: int
    names: list[str]


def defaultdict_group_manifest() -> dict[str, ScopeGroup]:
    categories = (
        "vision",
        "aligner",
        "atlas_input_rows",
        "atlas_output_rows",
        "language_lora",
        "input_embedding",
        "output_head",
        "base_language",
        "nonlanguage_lora",
        "unknown",
    )
    return {
        category: {"tensors": 0, "parameters": 0, "names": []}
        for category in categories
    }


def audit_optimizer_coverage(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    output_path: str | Path | None = None,
) -> JsonDict:
    trainable = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    occurrences: dict[int, int] = {}
    group_rows: JsonList = []
    grouped_names: list[str] = []
    for index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            occurrences[parameter_id] = occurrences.get(parameter_id, 0) + 1
            names.append(trainable.get(parameter_id, f"<unknown:{parameter_id}>"))
        group_rows.append(
            {
                "index": index,
                "learning_rate": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_tensors": len(group["params"]),
                "names": json_list(names),
            }
        )
        grouped_names.extend(names)
    missing = sorted(name for parameter_id, name in trainable.items() if parameter_id not in occurrences)
    duplicate = sorted(
        trainable.get(parameter_id, f"<unknown:{parameter_id}>")
        for parameter_id, count in occurrences.items()
        if count != 1
    )
    unknown = sorted(name for name in grouped_names if name.startswith("<unknown:"))
    errors: list[str] = []
    if missing:
        errors.append(f"optimizer omitted trainable parameters: {missing[:8]}")
    if duplicate:
        errors.append(f"optimizer duplicated trainable parameters: {duplicate[:8]}")
    if unknown:
        errors.append(f"optimizer contains frozen or unknown parameters: {unknown[:8]}")
    report: JsonDict = {
        "schema": OPTIMIZER_COVERAGE_SCHEMA,
        "trainable_tensors": len(trainable),
        "covered_tensors": len(occurrences),
        "groups": group_rows,
        "missing": json_list(missing),
        "duplicate": json_list(duplicate),
        "unknown": json_list(unknown),
        "errors": json_list(errors),
    }
    if output_path is not None:
        write_json_atomic(Path(output_path), report)
    if errors:
        raise RuntimeError("invalid optimizer coverage: " + "; ".join(errors))
    return report


def write_json_atomic(path: Path, payload: Mapping[str, JsonLike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
