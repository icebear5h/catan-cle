"""Merged visual features, spatial patch objectives and O-LoRA protected subspaces."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def _merged_patch_counts(image_grid_thw: torch.Tensor, merge_size: int) -> list[int]:
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError(f"image_grid_thw must have shape [batch, 3], got {image_grid_thw.shape}")
    counts = []
    for temporal, height, width in image_grid_thw.detach().cpu().tolist():
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"vision grid {(temporal, height, width)} is not divisible by merge size {merge_size}"
            )
        counts.append(int(temporal * (height // merge_size) * (width // merge_size)))
    return counts


def split_merged_visual_features(
    pooled_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    merge_size: int,
) -> list[torch.Tensor]:
    """Split Qwen's post-merger patch stream into one tensor per image."""

    counts = _merged_patch_counts(image_grid_thw, merge_size)
    if pooled_output.ndim == 2:
        if pooled_output.shape[0] != sum(counts):
            raise ValueError(
                f"pooled visual rows {pooled_output.shape[0]} do not match merged grids {counts}"
            )
        return list(torch.split(pooled_output, counts, dim=0))
    if pooled_output.ndim == 3 and pooled_output.shape[0] == len(counts):
        if any(count > pooled_output.shape[1] for count in counts):
            raise ValueError("batched pooled visual output is shorter than its grid")
        return [pooled_output[index, :count] for index, count in enumerate(counts)]
    raise ValueError(f"unsupported pooled visual output shape: {pooled_output.shape}")


def soft_patch_target(
    bbox: torch.Tensor,
    *,
    grid_height: int,
    grid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a bbox-aware soft distribution over the merged patch grid."""

    device = bbox.device
    ys = (torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5) / grid_height
    xs = (torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5) / grid_width
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x1, y1, x2, y2 = bbox.float()
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    sigma_x = torch.clamp((x2 - x1) / 2, min=1.0 / grid_width)
    sigma_y = torch.clamp((y2 - y1) / 2, min=1.0 / grid_height)
    gaussian = torch.exp(
        -0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2)
    )
    patch_x1, patch_x2 = xx - 0.5 / grid_width, xx + 0.5 / grid_width
    patch_y1, patch_y2 = yy - 0.5 / grid_height, yy + 0.5 / grid_height
    overlap_x = torch.clamp(torch.minimum(patch_x2, x2) - torch.maximum(patch_x1, x1), min=0)
    overlap_y = torch.clamp(torch.minimum(patch_y2, y2) - torch.maximum(patch_y1, y1), min=0)
    overlap = overlap_x * overlap_y
    weights = gaussian + (overlap > 0).float()
    target = (weights / weights.sum()).flatten()
    return target, xx.flatten(), yy.flatten()


def spatial_patch_loss(
    pooled_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    token_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    merge_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return normalized soft CE, tolerant top-1 accuracy, and target count."""

    features_by_image = split_merged_visual_features(
        pooled_output,
        image_grid_thw,
        merge_size=merge_size,
    )
    losses = []
    correct = []
    grids = image_grid_thw.detach().cpu().tolist()
    for index, (features, grid) in enumerate(zip(features_by_image, grids, strict=True)):
        if not bool(target_mask[index].item()):
            continue
        temporal, raw_height, raw_width = (int(value) for value in grid)
        if temporal != 1:
            raise ValueError("Catan patch localization supports one image frame per row")
        grid_height, grid_width = raw_height // merge_size, raw_width // merge_size
        feature_vectors = torch.nn.functional.normalize(features.float(), dim=-1)
        token_vector = torch.nn.functional.normalize(token_embeddings[index].float(), dim=-1)
        if feature_vectors.shape[-1] != token_vector.shape[-1]:
            raise ValueError(
                "post-merger visual features and semantic token embeddings must share LM space"
            )
        scores = feature_vectors @ token_vector / temperature
        target, xs, ys = soft_patch_target(
            bboxes[index],
            grid_height=grid_height,
            grid_width=grid_width,
        )
        target = target.to(scores.device)
        normalized = -(target * torch.log_softmax(scores, dim=0)).sum()
        normalized = normalized / max(math.log(scores.numel()), 1.0)
        losses.append(normalized)

        top = int(scores.detach().argmax().item())
        x1, y1, x2, y2 = bboxes[index].float()
        tolerance_x, tolerance_y = 1.0 / grid_width, 1.0 / grid_height
        correct.append(
            (xs[top] >= x1 - tolerance_x)
            & (xs[top] <= x2 + tolerance_x)
            & (ys[top] >= y1 - tolerance_y)
            & (ys[top] <= y2 + tolerance_y)
        )
    if not losses:
        zero = pooled_output.sum() * 0.0
        return zero, zero.detach(), 0
    return torch.stack(losses).mean(), torch.stack(correct).float().mean(), len(losses)


class VisionPoolerCapture:
    """Hold the differentiable post-merger output from the current forward pass."""

    def __init__(self, vision_module: torch.nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = vision_module.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: object,
                 output: object) -> None:
        pooled = getattr(output, "pooler_output", None)
        if pooled is None and isinstance(output, (tuple, list)) and len(output) > 1:
            pooled = output[1]
        if not isinstance(pooled, torch.Tensor):
            raise RuntimeError("Qwen vision forward did not expose a tensor pooler_output")
        self.output = pooled

    def take(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("vision pooler output was not captured for this batch")
        output = self.output
        self.output = None
        return output


def module_key(parameter_name: str) -> str:
    """The base-model module path an adapter or factor tensor belongs to.

    ``base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight``
    and ``model.visual.blocks.0.attn.qkv.lora_A.weight`` both map to their
    ``model....`` module path, so bases from a frozen adapter file and from the
    visual-delta factor file address the same modules as the live model.
    """

    name = parameter_name
    for prefix in ("base_model.model.", "base_model."):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for marker in (".lora_A.", ".lora_B."):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def orthonormal_rows(matrix: torch.Tensor) -> torch.Tensor:
    """An orthonormal basis (rows) of the row space of ``matrix``, in fp32."""

    rows = matrix.detach().float()
    q, r = torch.linalg.qr(rows.T)
    keep = torch.abs(torch.diagonal(r)) > 1e-6
    basis: torch.Tensor = q[:, keep].T.contiguous()
    return basis


def load_protected_bases(
    frozen_adapter_dir: Path,
    visual_delta_factors: Path | None,
) -> dict[str, torch.Tensor]:
    """Orthonormal input bases per module: the frozen adapter's ``lora_A`` rows, plus the visual delta's."""

    bases: dict[str, torch.Tensor] = {}
    adapter = load_file(frozen_adapter_dir / "adapter_model.safetensors")
    for name, tensor in adapter.items():
        if ".lora_A." in name:
            bases[module_key(name)] = orthonormal_rows(tensor)
    if visual_delta_factors is not None:
        factors = load_file(visual_delta_factors)
        for name, tensor in factors.items():
            if ".lora_A." in name:
                key = module_key(name)
                if key in bases:
                    raise RuntimeError(f"protected basis defined twice: {key}")
                bases[key] = orthonormal_rows(tensor)
    if not bases:
        raise RuntimeError("no protected bases were found")
    return bases


def orthogonal_penalty(
    model: torch.nn.Module,
    bases: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, float, int]:
    """O-LoRA's L_orth: the squared projection of every trainable ``lora_A`` onto its module's protected basis.

    Returns the penalty, the fraction of the adapters' total squared norm that
    lies inside the protected subspaces (a scale-free progress number), and the
    count of trainable adapters with no basis (which are unconstrained).
    """

    penalty = None
    inside = 0.0
    total = 0.0
    unprotected = 0
    for name, parameter in model.named_parameters():
        if ".lora_A." not in name or not parameter.requires_grad:
            continue
        key = module_key(name)
        basis = bases.get(key)
        if basis is None:
            unprotected += 1
            continue
        cached = cache.get(key)
        if cached is None or cached.device != parameter.device:
            cached = basis.to(parameter.device)
            cache[key] = cached
        rows = parameter.float()
        projection = rows @ cached.T
        term = (projection * projection).sum()
        penalty = term if penalty is None else penalty + term
        inside += float(term.detach())
        total += float((rows.detach() * rows.detach()).sum())
    if penalty is None:
        raise RuntimeError("no trainable lora_A parameter has a protected basis")
    return penalty, (inside / total if total else 0.0), unprotected


class LanguageHiddenCapture:
    """Hold the final language hidden state from the current forward pass."""

    def __init__(self, language_module: torch.nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = language_module.register_forward_hook(self._capture)

    def _capture(self, _module: torch.nn.Module, _inputs: object,
                 output: object) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("language forward did not expose a tensor last_hidden_state")
        self.output = hidden

    def take(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("language hidden state was not captured for this batch")
        output = self.output
        self.output = None
        return output
