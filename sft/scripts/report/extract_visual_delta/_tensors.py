"""tensors."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from ._base import RANKS, THRESHOLDS, DeltaStats, SvdSummary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_visual_key(name: str) -> str | None:
    if name.startswith("visual."):
        return "model." + name
    if ".visual." not in name:
        return None
    return "model.visual." + name.split(".visual.", 1)[1]


def visual_keys(names: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        canonical = canonical_visual_key(name)
        if canonical is None:
            continue
        if canonical in result:
            raise ValueError(f"duplicate canonical visual key: {canonical}")
        result[canonical] = name
    return result


def loaded_base(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    target = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    return tensor.to(dtype=target).to(dtype=torch.float32)


def tensor_delta(
    base: torch.Tensor, trained: torch.Tensor, dtype: str,
) -> tuple[torch.Tensor, DeltaStats]:
    if base.shape != trained.shape:
        raise ValueError(f"shape mismatch: base={tuple(base.shape)} trained={tuple(trained.shape)}")
    if not base.is_floating_point() or not trained.is_floating_point():
        raise ValueError("visual delta requires floating-point tensors")
    original = loaded_base(base, dtype)
    current = trained.float()
    if not bool(torch.isfinite(original).all() & torch.isfinite(current).all()):
        raise ValueError("nonfinite source weights")
    # FP32 subtraction can lose small trained values when base and delta cancel.
    # Keep the dense task vector in FP64; SVD below deliberately uses FP32.
    delta = (current.double() - original.double()).contiguous()
    reconstructed = (original.double() + delta).to(current.dtype)
    error = (reconstructed - current).abs()
    energy = float(torch.sum(delta.square(), dtype=torch.float64))
    base_energy = float(torch.sum(original.square(), dtype=torch.float64))
    norm = math.sqrt(energy)
    return delta, {
        "shape": list(delta.shape),
        "numel": delta.numel(),
        "base_storage_dtype": str(base.dtype),
        "trained_storage_dtype": str(trained.dtype),
        "delta_norm": norm,
        "delta_energy": energy,
        "relative_delta_norm": norm / math.sqrt(base_energy) if base_energy else None,
        "changed_fraction": float(torch.count_nonzero(delta)) / delta.numel(),
        "reconstruction_bit_exact": bool(torch.equal(reconstructed, current)),
        "reconstruction_max_abs_error": float(error.max()),
    }


def matrix_kind(name: str, tensor: torch.Tensor) -> str | None:
    if tensor.ndim == 2:
        return "embedding_matrix_not_linear_lora" if ".pos_embed." in name else "linear_weight"
    if tensor.ndim > 2:
        return "flattened_convolution_not_linear_lora"
    return None


def decompose_matrix(
    matrix: torch.Tensor, factor_rank: int,
) -> tuple[SvdSummary, dict[str, torch.Tensor]]:
    if matrix.ndim != 2 or factor_rank < 1:
        raise ValueError("requires a matrix and positive factor rank")
    u, singular, vh = torch.linalg.svd(matrix.float(), full_matrices=False)
    energies = singular.double().square()
    total = float(energies.sum())
    cumulative = energies.cumsum(0)
    k = min(factor_rank, singular.numel())
    root = singular[:k].sqrt()
    factor_b = (u[:, :k] * root).contiguous()
    factor_a = (root[:, None] * vh[:k, :]).contiguous()
    residual = matrix.float() - factor_b @ factor_a
    residual_energy = float(torch.sum(residual.square(), dtype=torch.float64))
    actual_energy = float(torch.sum(matrix.float().square(), dtype=torch.float64))
    numerical_tol = max(matrix.shape) * torch.finfo(torch.float32).eps * float(singular[0])
    rank_energy = {
        str(r): float(cumulative[min(r, singular.numel()) - 1]) / total if total else 1.0
        for r in RANKS
    }
    threshold_ranks = {
        str(t): int(torch.searchsorted(cumulative, t * total)) + 1 if total else 0
        for t in THRESHOLDS
    }
    result: SvdSummary = {
        "matrix_shape": list(matrix.shape),
        "maximum_rank": min(matrix.shape),
        "numerical_rank_fp32": int(torch.count_nonzero(singular > numerical_tol)),
        "singular_values": singular.tolist(),
        "rank_energy_fraction": rank_energy,
        "energy_threshold_ranks": threshold_ranks,
        "saved_factor_rank": k,
        "saved_factor_relative_frobenius_error": math.sqrt(residual_energy / actual_energy)
        if actual_energy else 0.0,
        "svd_energy_relative_error": abs(total - actual_energy) / actual_energy if actual_energy else 0.0,
        "output_basis_orthogonality_max_abs": float((u[:, :k].T @ u[:, :k] - torch.eye(k)).abs().max()),
        "input_basis_orthogonality_max_abs": float((vh[:k] @ vh[:k].T - torch.eye(k)).abs().max()),
    }
    factors = {
        "lora_A": factor_a,
        "lora_B": factor_b,
        "output_basis": u[:, :k].contiguous(),
        "input_basis": vh[:k, :].T.contiguous(),
        "singular_values": singular.contiguous(),
    }
    return result, factors
