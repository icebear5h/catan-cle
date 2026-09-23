"""Bounded-memory CPU arithmetic: one base shard and one pair of factors at a time."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from sft.safetensor_types import TensorFile, TensorHeader, open_tensors

from ._contracts import ADAPTER, VISUAL, MergePlan, Rounding, require


def finite(tensor: torch.Tensor, name: str) -> None:
    flat = tensor.reshape(-1)
    for start in range(0, flat.numel(), 1 << 20):
        require(bool(torch.isfinite(flat[start:start + (1 << 20)]).all()),
                f"nonfinite tensor or dtype overflow: {name}")


def _rounding() -> Rounding:
    return {"elements": 0, "rounded_elements": 0, "max_abs_rounding_error": 0.0}


def _cast(values: torch.Tensor, stats: Rounding, name: str) -> torch.Tensor:
    finite(values, name)
    rounded = values.to(torch.bfloat16)
    finite(rounded, name)
    errors = (values - rounded.float()).abs()
    stats["elements"] += values.numel()
    stats["rounded_elements"] += int(torch.count_nonzero(errors).item())
    stats["max_abs_rounding_error"] = max(stats["max_abs_rounding_error"], float(errors.max().item()))
    return rounded


def merge_lora(weight: torch.Tensor, a: torch.Tensor, b: torch.Tensor,
               name: str) -> tuple[torch.Tensor, Rounding]:
    """Compute W + 2*(B@A) in FP32, then round once to BF16, independently per module."""
    finite(a, name + ".A")
    finite(b, name + ".B")
    stats = _rounding()
    output = torch.empty_like(weight, dtype=torch.bfloat16, device="cpu")
    factor_a = a.float()
    # Bound FP32 workspaces, including for unusually wide projections.
    chunk_rows = max(1, min(1024, (1 << 20) // weight.shape[1]))
    for start in range(0, weight.shape[0], chunk_rows):
        end = start + chunk_rows
        values = (b[start:end].float() @ factor_a).mul_(2.0)
        values.add_(weight[start:end].float())
        output[start:end] = _cast(values, stats, name)
    return output, stats


def replace_rows(weight: torch.Tensor, rows: torch.Tensor, ids: tuple[int, ...],
                 name: str) -> tuple[torch.Tensor, Rounding]:
    finite(weight, name)
    stats = _rounding()
    # Do not materialize the full vocabulary in FP32. These are replacement rows.
    output = weight.to(dtype=torch.bfloat16, copy=True)
    finite(output, name)
    output.index_copy_(0, torch.tensor(ids, dtype=torch.long, device="cpu"),
                       _cast(rows.float(), stats, name))
    return output, stats


def _transform(key: str, weight: torch.Tensor, adapter: TensorFile, visual: TensorFile,
               plan: MergePlan, rounding: dict[str, Rounding]) -> torch.Tensor:
    if key in plan.visual:
        tensor = visual.get_tensor(plan.visual[key])
        finite(tensor, key)
        return tensor
    if key in plan.lora:
        a, b = plan.lora[key]
        result, rounding[key] = merge_lora(weight, adapter.get_tensor(a), adapter.get_tensor(b), key)
        return result
    if key in plan.rows:
        result, rounding[key] = replace_rows(weight, adapter.get_tensor(plan.rows[key]), plan.token_ids, key)
        return result
    finite(weight, key)
    return weight


def stream_shards(base_dir: Path, adapter_dir: Path, output: Path, plan: MergePlan,
                  ) -> tuple[dict[str, str], dict[str, TensorHeader], dict[str, Rounding]]:
    weights: dict[str, str] = {}
    headers: dict[str, TensorHeader] = {}
    rounding: dict[str, Rounding] = {}
    count = len(plan.base.shards)
    with torch.no_grad(), torch.autocast(device_type="cpu", enabled=False), open_tensors(
        adapter_dir / ADAPTER,
    ) as adapter, open_tensors(adapter_dir / VISUAL) as visual:
        for index, (source, keys) in enumerate(plan.base.shards.items(), start=1):
            name = f"model-{index:05d}-of-{count:05d}.safetensors"
            with open_tensors(base_dir / source) as base:
                tensors: dict[str, torch.Tensor] = {}
                for key in keys:
                    tensors[key] = _transform(key, base.get_tensor(key), adapter, visual, plan, rounding)
                    weights[key] = name
                    dtype = "F32" if key in plan.visual else (
                        "BF16" if key in plan.lora or key in plan.rows else plan.base.headers[key]["dtype"])
                    headers[key] = {"shape": list(tensors[key].shape), "dtype": dtype}
                save_file(tensors, output / name, metadata={"format": "pt"})
                del tensors
            # Check the serialized visual bytes, including signed zeros; no BF16 round trip.
            with open_tensors(output / name) as saved:
                for key in keys:
                    if key in plan.visual:
                        actual = saved.get_tensor(key).reshape(-1).view(torch.uint8)
                        expected = visual.get_tensor(plan.visual[key]).reshape(-1).view(torch.uint8)
                        require(torch.equal(actual, expected), f"visual bytes changed during export: {key}")
                        del actual, expected
            # No base shard tensors are retained across iterations.
    require(set(weights) == set(plan.base.weight_map), "not all base tensors were exported")
    require(set(rounding) == set(plan.lora) | set(plan.rows), "not all adapter tensors were consumed")
    return weights, headers, rounding
