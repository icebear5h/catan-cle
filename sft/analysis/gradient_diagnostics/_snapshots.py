"""Raw gradient capture and the exact norm/cosine relationships it feeds."""

from __future__ import annotations

import math
from statistics import fmean, pstdev
from typing import Callable, Mapping, Sequence

import torch

from sft.json_types import json_path, opt_float

from ._types import PARAMETER_GROUPS, GradientSnapshot, JsonDict


def probe_parameter_group(training_category: str) -> str | None:
    if training_category in {"vision", "merger", "language_lora"}:
        return training_category
    if training_category in {"atlas_input_rows", "atlas_output_rows"}:
        return "token_rows"
    return None


def capture_gradient_snapshot(
    model: torch.nn.Module,
    category_for_name: Callable[[str], str],
) -> GradientSnapshot:
    """Copy one full raw gradient into CPU fp32 tensors by parameter group."""

    snapshot: GradientSnapshot = {name: {} for name in PARAMETER_GROUPS}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = probe_parameter_group(category_for_name(name))
        if group is None:
            raise RuntimeError(f"unroutable trainable parameter: {name}")
        if parameter.grad is not None:
            snapshot[group][name] = parameter.grad.detach().to(
                device="cpu",
                dtype=torch.float32,
                copy=True,
            )
    return snapshot


def mean_gradient_snapshot(snapshots: Sequence[GradientSnapshot]) -> GradientSnapshot:
    if not snapshots:
        raise ValueError("cannot average an empty snapshot sequence")
    result: GradientSnapshot = {name: {} for name in PARAMETER_GROUPS}
    scale = 1.0 / len(snapshots)
    for snapshot in snapshots:
        for group in PARAMETER_GROUPS:
            for name, tensor in snapshot[group].items():
                if name not in result[group]:
                    result[group][name] = tensor.clone().mul_(scale)
                else:
                    result[group][name].add_(tensor, alpha=scale)
    return result


def _group_dot(left: GradientSnapshot, right: GradientSnapshot, group: str) -> float:
    shared = left[group].keys() & right[group].keys()
    return sum(
        float(torch.vdot(left[group][name].reshape(-1), right[group][name].reshape(-1)))
        for name in shared
    )


def _group_norm_sq(snapshot: GradientSnapshot, group: str) -> float:
    return sum(float(torch.vdot(tensor.reshape(-1), tensor.reshape(-1))) for tensor in snapshot[group].values())


def _safe_cosine(dot: float, left_norm_sq: float, right_norm_sq: float) -> float | None:
    denominator = math.sqrt(left_norm_sq * right_norm_sq)
    return dot / denominator if denominator > 0 else None


def snapshot_norms(snapshot: GradientSnapshot, learning_rates: Mapping[str, float]) -> JsonDict:
    group_norm_sq = {group: _group_norm_sq(snapshot, group) for group in PARAMETER_GROUPS}
    result: JsonDict = {}
    for group, norm_sq in group_norm_sq.items():
        learning_rate = float(learning_rates[group])
        result[group] = {
            "raw": math.sqrt(norm_sq),
            "lr_scaled": learning_rate * math.sqrt(norm_sq),
            "learning_rate": learning_rate,
        }
    result["all"] = {
        "raw": math.sqrt(sum(group_norm_sq.values())),
        "lr_scaled": math.sqrt(
            sum(float(learning_rates[group]) ** 2 * value for group, value in group_norm_sq.items())
        ),
    }
    return result


def snapshot_relationship(
    left: GradientSnapshot,
    right: GradientSnapshot,
    learning_rates: Mapping[str, float],
) -> JsonDict:
    left_norm_sq = {group: _group_norm_sq(left, group) for group in PARAMETER_GROUPS}
    right_norm_sq = {group: _group_norm_sq(right, group) for group in PARAMETER_GROUPS}
    dots = {group: _group_dot(left, right, group) for group in PARAMETER_GROUPS}
    result: JsonDict = {}
    for group in PARAMETER_GROUPS:
        learning_rate = float(learning_rates[group])
        cosine = _safe_cosine(dots[group], left_norm_sq[group], right_norm_sq[group])
        result[group] = {
            "raw_dot": dots[group],
            "raw_cosine": cosine,
            "lr_scaled_dot": learning_rate**2 * dots[group],
            "lr_scaled_cosine": cosine,
        }

    raw_left = sum(left_norm_sq.values())
    raw_right = sum(right_norm_sq.values())
    raw_dot = sum(dots.values())
    scaled_left = sum(
        float(learning_rates[group]) ** 2 * left_norm_sq[group] for group in PARAMETER_GROUPS
    )
    scaled_right = sum(
        float(learning_rates[group]) ** 2 * right_norm_sq[group] for group in PARAMETER_GROUPS
    )
    scaled_dot = sum(
        float(learning_rates[group]) ** 2 * dots[group] for group in PARAMETER_GROUPS
    )
    result["all"] = {
        "raw_dot": raw_dot,
        "raw_cosine": _safe_cosine(raw_dot, raw_left, raw_right),
        "lr_scaled_dot": scaled_dot,
        "lr_scaled_cosine": _safe_cosine(scaled_dot, scaled_left, scaled_right),
    }
    return result


def _distribution(values: Sequence[float | None]) -> JsonDict:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "fraction_negative": None}
    return {
        "count": len(finite),
        "mean": fmean(finite),
        "std": pstdev(finite),
        "min": min(finite),
        "max": max(finite),
        "fraction_negative": sum(value < 0 for value in finite) / len(finite),
    }


def minibatch_relationships(
    left: Sequence[GradientSnapshot],
    right: Sequence[GradientSnapshot],
    learning_rates: Mapping[str, float],
) -> JsonDict:
    if len(left) != len(right):
        raise ValueError("paired minibatch sequences must have equal lengths")
    relationships = [
        snapshot_relationship(left_item, right_item, learning_rates)
        for left_item, right_item in zip(left, right, strict=True)
    ]
    result: JsonDict = {}
    for group in (*PARAMETER_GROUPS, "all"):
        result[group] = {
            field: _distribution([opt_float(json_path(item, group, field)) for item in relationships])
            for field in ("raw_dot", "raw_cosine", "lr_scaled_dot", "lr_scaled_cosine")
        }
    return result
