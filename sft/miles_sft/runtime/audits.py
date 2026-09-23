"""Audit the actual tensors; only native Bridge A/B weights may require gradients."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import torch

from sft.json_types import JsonLikeDict

from .contracts import TARGET_SUFFIXES, require

_TARGET = re.compile(
    r"(?:module\.)*language_model\.decoder\.layers\.[0-9]+\."
    r"(" + "|".join(re.escape(s) for s in TARGET_SUFFIXES) + r")"
)
_ADAPTER = re.compile(_TARGET.pattern + r"\.adapter\.linear_(in|out)\.weight")


def adapter_parameters(
    model: Sequence[torch.nn.Module], tensor_parallel_size: int = 1,
) -> tuple[dict[str, torch.nn.Parameter], JsonLikeDict]:
    """Check every requires_grad flag, all target coverage, and local rank dimensions."""
    require(len(model) == 1, "one non-pipelined model chunk is required")
    require(tensor_parallel_size > 0 and 16 % tensor_parallel_size == 0, "invalid LoRA TP size")
    params: dict[str, torch.nn.Parameter] = {}
    pairs: dict[str, set[str]] = {}
    kinds: set[str] = set()
    frozen = vision = tokens = 0
    for name, param in model[0].named_parameters(remove_duplicate=False):
        match = _ADAPTER.fullmatch(name)
        if param.requires_grad:
            require(match is not None, f"forbidden trainable parameter: {name}")
        if match is None:
            frozen += param.numel()
            vision += param.numel() if any(s in name.split(".") for s in
                                          ("vision_model", "vision_projection", "visual")) else 0
            tokens += param.numel() if any(s in name.split(".") for s in
                                          ("word_embeddings", "output_layer", "embed_tokens", "lm_head")) else 0
            continue
        require(param.requires_grad, f"adapter unexpectedly frozen: {name}")
        require(name not in params, f"duplicate adapter parameter: {name}")
        require(param.ndim == 2 and param.numel() > 0, f"invalid adapter shape: {name}")
        dimension = param.shape[0 if match[2] == "in" else 1]
        require(dimension in {16, 16 // tensor_parallel_size}, f"wrong rank16 adapter shape: {name}")
        params[name] = param
        base = name.split(".adapter.")[0]
        pairs.setdefault(base, set()).add(match[2])
        kinds.add(match[1])
    require(bool(params), "no trainable native Bridge adapters found")
    require(len({id(p) for p in params.values()}) == len(params), "aliased adapter tensors forbidden")
    require(kinds == set(TARGET_SUFFIXES), "missing language target families, including GDN")
    targets = {name for name, _ in model[0].named_modules() if _TARGET.fullmatch(name)}
    require(set(pairs) == targets and all(parts == {"in", "out"} for parts in pairs.values()),
            "every language target must have both A and B; missing or partial adapters")
    require(frozen > 0 and vision > 0 and tokens > 0, "full frozen base/vision/token matrices required")
    return params, {
        "trainable_shapes": {n: list(p.shape) for n, p in params.items()},
        "trainable_numel": sum(p.numel() for p in params.values()), "frozen_numel": frozen,
        "frozen_vision_numel": vision, "frozen_token_numel": tokens,
        "target_families": sorted(kinds), "adapter_pairs": len(pairs),
    }


def audit_fresh_adapter(params: Mapping[str, torch.Tensor]) -> None:
    for name, param in params.items():
        require(bool(torch.isfinite(param).all().item()), f"nonfinite initial adapter: {name}")
        nonzero = bool(torch.count_nonzero(param).item())
        require(nonzero == name.endswith(".linear_in.weight"),
                f"fresh LoRA must have nonzero A and zero B: {name}")


def gradient_witness(params: Mapping[str, torch.nn.Parameter]) -> JsonLikeDict:
    """Read Megatron main_grad before .grad; never treat missing gradients as zero."""
    require(bool(params), "cannot witness an empty adapter")
    nonzero = main_grad_count = 0
    norm_squared = 0.0
    for name, param in params.items():
        main_grad: object = getattr(param, "main_grad", None)
        grad = main_grad if main_grad is not None else param.grad
        require(isinstance(grad, torch.Tensor), f"missing adapter gradient: {name}")
        if not isinstance(grad, torch.Tensor):
            raise TypeError(f"invalid gradient: {name}")
        require(grad.shape == param.shape, f"gradient shape mismatch: {name}")
        require(bool(torch.isfinite(grad).all().item()), f"nonfinite adapter gradient: {name}")
        main_grad_count += main_grad is not None
        square = float(grad.detach().double().square().sum().item())
        norm_squared += square
        nonzero += square > 0
    require(nonzero > 0, "no nonzero adapter gradients")
    return {"gradient_tensors": len(params), "main_grad_tensors": main_grad_count,
            "nonzero_gradient_tensors": nonzero, "gradient_l2": norm_squared ** 0.5}


def update_witness(
    before: Mapping[str, torch.Tensor], params: Mapping[str, torch.nn.Parameter],
) -> JsonLikeDict:
    require(bool(params) and before.keys() == params.keys(), "adapter snapshot scope changed")
    changed: list[str] = []
    norm_squared = 0.0
    for name, param in params.items():
        require(bool(torch.isfinite(param).all().item()), f"nonfinite updated adapter: {name}")
        if not torch.equal(before[name], param.detach()):
            changed.append(name)
            norm_squared += float((param.detach().double() - before[name].double()).square().sum().item())
    return {"changed_tensors": changed, "update_l2": norm_squared ** 0.5,
            "parameters_changed": bool(changed)}
