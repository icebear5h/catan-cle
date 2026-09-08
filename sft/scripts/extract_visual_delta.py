"""Extract a dense visual task vector and diagnostic per-matrix SVD factors.

This does not load a model, train, merge adapters, or replace checkpoint weights.
The base tensors are cast through the original loader dtype before subtraction.
Saved factors use delta ~= lora_B @ lora_A with multiplier 1; they are NOT a
drop-in PEFT adapter. Non-linear-layer tensors remain in the exact dense delta.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file


RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
THRESHOLDS = (0.5, 0.9, 0.95, 0.99)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
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


def tensor_delta(base: torch.Tensor, trained: torch.Tensor, dtype: str) -> tuple[torch.Tensor, dict]:
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


def decompose_matrix(matrix: torch.Tensor, factor_rank: int) -> tuple[dict, dict[str, torch.Tensor]]:
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
    result = {
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


def summarize(rows: list[dict]) -> dict:
    result: dict[str, Any] = {}
    groups = {"all": rows, "linear_weights": [r for r in rows if r.get("matrix_kind") == "linear_weight"]}
    for group in ("tower", "merger"):
        groups[group] = [r for r in rows if r["component"] == group]
    for name, selected in groups.items():
        matrices = [r for r in selected if "svd" in r]
        energy = sum(r["delta_energy"] for r in matrices)
        result[name] = {
            "tensor_count": len(selected),
            "numel": sum(r["numel"] for r in selected),
            "matrix_count": len(matrices),
            "delta_norm": math.sqrt(sum(r["delta_energy"] for r in selected)),
            "matrix_delta_energy": energy,
            "nonmatrix_delta_energy": sum(r["delta_energy"] for r in selected if "svd" not in r),
            "matrix_energy_weighted_rank_capture": {
                str(rank): sum(r["delta_energy"] * r["svd"]["rank_energy_fraction"][str(rank)] for r in matrices) / energy
                if energy else 1.0 for rank in RANKS
            },
        }
    return result


def write_summary_tables(output_dir: Path, report: dict) -> None:
    """Small review artifacts; detailed spectra and factors remain in their files."""
    fields = ["name", "component", "matrix_kind", "shape", "delta_norm", "relative_delta_norm",
              "rank8_energy", "rank16_energy", "rank32_energy", "rank64_energy",
              "rank128_energy", "rank256_energy", "rank90", "rank95", "rank99"]
    with (output_dir / "matrix_spectra.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in report["tensors"]:
            if "svd" not in row:
                continue
            record = {key: row[key] for key in fields[:6]}
            for rank in (8, 16, 32, 64, 128, 256):
                record[f"rank{rank}_energy"] = row["svd"]["rank_energy_fraction"][str(rank)]
            for percent in (90, 95, 99):
                record[f"rank{percent}"] = row["svd"]["energy_threshold_ranks"][str(percent / 100)]
            writer.writerow(record)
    lines = ["# Visual delta extraction", "", f"Base: `{report['model_id']}@{report['hf_revision']}`.",
             "", "Dense delta = FP32 checkpoint minus original HF tensors cast through "
             f"`{report['base_load_dtype']}` (the trainer's base-load dtype).", "",
             "Dense deltas are FP64. Adding to the FP64-promoted loaded base and casting to FP32 "
             "reconstructs every saved checkpoint tensor bit-for-bit. SVD uses FP32 matrices.", "",
             "| Scope | Matrices | Rank 8 energy | Rank 16 | Rank 64 | Rank 256 |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, row in report["summary"].items():
        capture = row["matrix_energy_weighted_rank_capture"]
        values = " | ".join(f"{100*capture[str(r)]:.2f}%" for r in (8, 16, 64, 256))
        lines.append(f"| {group} | {row['matrix_count']} | {values} |")
    lines += ["", "Energy means squared Frobenius norm of weight changes, not task accuracy or retained knowledge.",
              "", "## Files", "", "- `visual_delta.safetensors`: dense delta, all visual tensors including biases/norms.",
              "- `visual_delta_rank16_linear.safetensors`: compact linear A/B factors, capped at rank 16 or the requested saved rank.",
              "- `svd_factors/*.safetensors`: top-k input/output bases, complete singular spectra, and low-rank A/B factors.",
              "- `matrix_spectra.csv`: one row per decomposed tensor; per-layer energy and threshold ranks.",
              "- `report.json`: source hashes, key mapping, reconstruction checks, and complete analysis.",
              "", "Factor convention: `delta_approx = lora_B @ lora_A`, scaling multiplier 1. "
              "For rank r <= saved rank, slice `lora_B[:, :r]` and `lora_A[:r, :]`.",
              "", "These are diagnostic factors, not a directly loadable PEFT adapter. "
              "Position embeddings and flattened convolution kernels are explicitly marked separately from linear weights. "
              "Norms and biases are not silently dropped from the dense delta.",
              "", "Keep the original v2 checkpoint intact. Do not replace it with the truncated factors. "
              "The factors are candidate protected directions; this run performs no training or behavioral evaluation.", ""]
    (output_dir / "README.md").write_text("\n".join(lines))


def extract(
    *, trained_path: Path, base_index: Path, base_dir: Path,
    hf_info: Path, output_dir: Path, revision: str,
    model_id: str, factor_rank: int = 256, base_load_dtype: str = "bfloat16",
    trained_sha256: str | None = None,
) -> dict:
    if factor_rank < 1:
        raise ValueError("factor_rank must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    info = json.loads(hf_info.read_text())
    if info.get("sha") != revision or info.get("id") != model_id:
        raise ValueError("Hugging Face metadata does not match requested model/revision")
    index = json.loads(base_index.read_text())["weight_map"]
    indexed = visual_keys(list(index))
    files = sorted({index[source] for source in indexed.values()})
    metadata = {row["rfilename"]: row for row in info["siblings"]}
    file_provenance = []
    for filename in files:
        path = base_dir / filename
        expected = metadata[filename]["lfs"]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"HF shard SHA256 mismatch: {filename}")
        file_provenance.append({"file": filename, "sha256": actual, "bytes": path.stat().st_size})
    actual_trained_sha = sha256_file(trained_path)
    if trained_sha256 is not None and actual_trained_sha != trained_sha256:
        raise ValueError("trained visual SHA256 mismatch")
    output_dir.mkdir(parents=True)
    report: dict[str, Any] = {
        "schema": "catan_visual_delta_svd/v1",
        "status": "extracting",
        "model_id": model_id, "hf_revision": revision,
        "base_load_dtype": base_load_dtype, "factor_rank": factor_rank,
        "dense_delta_dtype": "float64", "svd_dtype": "float32",
        "base_shards": file_provenance,
        "base_index_sha256": sha256_file(base_index),
        "trained_path": str(trained_path.resolve()), "trained_sha256": actual_trained_sha,
        "meaning": "trained visual weights minus HF original weights cast through base_load_dtype",
        "scope": "full visual tower and merger; excludes language LoRA and atlas rows",
        "factor_convention": "delta ~= lora_B @ lora_A; multiplier=1; diagnostic factors, not a PEFT bundle",
        "protection_warning": "large weight-change directions are not proven retention-sensitive directions",
        "tensors": [],
    }
    write_json(output_dir / "report.json", report)
    deltas: dict[str, torch.Tensor] = {}
    start = time.monotonic()
    with safe_open(str(trained_path), framework="pt", device="cpu") as trained:
        trained_keys = visual_keys(list(trained.keys()))
        if set(trained_keys) != set(indexed) or len(trained_keys) != len(trained.keys()):
            raise ValueError(f"visual key sets differ: missing={sorted(set(indexed)-set(trained_keys))}, extra={sorted(set(trained_keys)-set(indexed))}")
        for filename in files:
            with safe_open(str(base_dir / filename), framework="pt", device="cpu") as base:
                for name in sorted(indexed):
                    if index[indexed[name]] != filename:
                        continue
                    tensor = trained.get_tensor(trained_keys[name])
                    if tensor.dtype != torch.float32:
                        raise ValueError(f"expected FP32 intermediate visual checkpoint: {name} is {tensor.dtype}")
                    delta, row = tensor_delta(base.get_tensor(indexed[name]), tensor, base_load_dtype)
                    if not row["reconstruction_bit_exact"]:
                        raise ValueError(f"dense delta did not reconstruct checkpoint exactly: {name}")
                    row.update({"name": name, "trained_key": trained_keys[name], "base_key": indexed[name],
                                "component": "merger" if ".merger." in name else "tower",
                                "matrix_kind": matrix_kind(name, delta)})
                    report["tensors"].append(row)
                    deltas[name] = delta
    dense_path = output_dir / "visual_delta.safetensors"
    save_file(deltas, dense_path, metadata={"schema": report["schema"], "hf_revision": revision,
                                         "operation": "trained_minus_base_loaded", "base_load_dtype": base_load_dtype})
    del deltas
    report["dense_delta_sha256"] = sha256_file(dense_path)
    report["status"] = "decomposing"
    report["source_dtypes"] = dict(Counter(r["trained_storage_dtype"] for r in report["tensors"]))
    write_json(output_dir / "report.json", report)
    factor_dir = output_dir / "svd_factors"
    factor_dir.mkdir()
    rank16_linear: dict[str, torch.Tensor] = {}
    matrices = [r for r in report["tensors"] if r["matrix_kind"] is not None]
    with safe_open(str(dense_path), framework="pt", device="cpu") as dense:
        for i, row in enumerate(matrices, 1):
            tensor = dense.get_tensor(row["name"])
            matrix = tensor.reshape(tensor.shape[0], -1)
            before = time.monotonic()
            svd, factors = decompose_matrix(matrix, factor_rank)
            row["svd"] = svd
            filename = row["name"] + ".safetensors"
            save_file(factors, factor_dir / filename,
                      metadata={"source_tensor": row["name"], "matrix_kind": row["matrix_kind"],
                                "original_shape": json.dumps(row["shape"]), "multiplier": "1"})
            row["factor_file"] = "svd_factors/" + filename
            row["factor_sha256"] = sha256_file(factor_dir / filename)
            if row["matrix_kind"] == "linear_weight":
                k16 = min(16, svd["saved_factor_rank"])
                module = row["name"].removesuffix(".weight")
                rank16_linear[module + ".lora_A.weight"] = factors["lora_A"][:k16].clone().contiguous()
                rank16_linear[module + ".lora_B.weight"] = factors["lora_B"][:, :k16].clone().contiguous()
            write_json(output_dir / "report.json", report)
            print(json.dumps({"matrix": i, "of": len(matrices), "name": row["name"],
                              "seconds": round(time.monotonic()-before, 2),
                              "rank16_energy": svd["rank_energy_fraction"]["16"],
                              "rank90": svd["energy_threshold_ranks"]["0.9"]}), flush=True)
    report["summary"] = summarize(report["tensors"])
    rank16_path = output_dir / "visual_delta_rank16_linear.safetensors"
    save_file(rank16_linear, rank16_path,
              metadata={"schema": report["schema"], "multiplier": "1",
                        "scope": "linear visual weights only; excludes embedding convolution bias norm",
                        "note": "diagnostic SVD factors; NOT a complete PEFT adapter"})
    report["rank16_linear_sha256"] = sha256_file(rank16_path)
    report["elapsed_seconds"] = time.monotonic() - start
    report["status"] = "complete"
    write_json(output_dir / "report.json", report)
    write_summary_tables(output_dir, report)
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "summary": report["summary"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--trained-sha256")
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--hf-info", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-load-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--factor-rank", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    with torch.inference_mode():
        extract(trained_path=args.trained, trained_sha256=args.trained_sha256,
                base_index=args.base_index, base_dir=args.base_dir, hf_info=args.hf_info,
                output_dir=args.output_dir, revision=args.revision, model_id=args.model_id,
                factor_rank=args.factor_rank, base_load_dtype=args.base_load_dtype)


if __name__ == "__main__":
    main()
