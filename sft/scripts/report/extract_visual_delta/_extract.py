"""Extract a dense visual task vector and diagnostic per-matrix SVD factors.

This does not load a model, train, merge adapters, or replace checkpoint weights.
The base tensors are cast through the original loader dtype before subtraction.
Saved factors use delta ~= lora_B @ lora_A with multiplier 1; they are NOT a
drop-in PEFT adapter. Non-linear-layer tensors remain in the exact dense delta.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import save_file

from sft.json_types import as_dict, as_list, as_str, json_path, load_json_dict
from sft.safetensor_types import open_tensors

from ._base import DeltaReport, DeltaStats, ShardProvenance, TensorRow
from ._summaries import summarize, write_summary_tables
from ._tensors import (
    decompose_matrix,
    matrix_kind,
    sha256_file,
    tensor_delta,
    visual_keys,
    write_json,
)


def _tensor_row(
    stats: DeltaStats, *, name: str, trained_key: str, base_key: str, matrix_kind: str | None,
) -> TensorRow:
    return {
        "shape": stats["shape"],
        "numel": stats["numel"],
        "base_storage_dtype": stats["base_storage_dtype"],
        "trained_storage_dtype": stats["trained_storage_dtype"],
        "delta_norm": stats["delta_norm"],
        "delta_energy": stats["delta_energy"],
        "relative_delta_norm": stats["relative_delta_norm"],
        "changed_fraction": stats["changed_fraction"],
        "reconstruction_bit_exact": stats["reconstruction_bit_exact"],
        "reconstruction_max_abs_error": stats["reconstruction_max_abs_error"],
        "name": name,
        "trained_key": trained_key,
        "base_key": base_key,
        "component": "merger" if ".merger." in name else "tower",
        "matrix_kind": matrix_kind,
    }


def extract(
    *, trained_path: Path, base_index: Path, base_dir: Path,
    hf_info: Path, output_dir: Path, revision: str,
    model_id: str, factor_rank: int = 256, base_load_dtype: str = "bfloat16",
    trained_sha256: str | None = None,
) -> DeltaReport:
    if factor_rank < 1:
        raise ValueError("factor_rank must be positive")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    info = load_json_dict(hf_info)
    if info.get("sha") != revision or info.get("id") != model_id:
        raise ValueError("Hugging Face metadata does not match requested model/revision")
    index = {
        name: as_str(filename)
        for name, filename in as_dict(load_json_dict(base_index)["weight_map"]).items()
    }
    indexed = visual_keys(list(index))
    files = sorted({index[source] for source in indexed.values()})
    siblings = [as_dict(row) for row in as_list(info["siblings"])]
    metadata = {as_str(row["rfilename"]): row for row in siblings}
    file_provenance: list[ShardProvenance] = []
    for filename in files:
        path = base_dir / filename
        expected = json_path(metadata[filename], "lfs", "sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"HF shard SHA256 mismatch: {filename}")
        file_provenance.append({"file": filename, "sha256": actual, "bytes": path.stat().st_size})
    actual_trained_sha = sha256_file(trained_path)
    if trained_sha256 is not None and actual_trained_sha != trained_sha256:
        raise ValueError("trained visual SHA256 mismatch")
    output_dir.mkdir(parents=True)
    tensors: list[TensorRow] = []
    report: DeltaReport = {
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
        "tensors": tensors,
    }
    write_json(output_dir / "report.json", report)
    deltas: dict[str, torch.Tensor] = {}
    start = time.monotonic()
    with open_tensors(str(trained_path), framework="pt", device="cpu") as trained:
        trained_keys = visual_keys(list(trained.keys()))
        if set(trained_keys) != set(indexed) or len(trained_keys) != len(trained.keys()):
            raise ValueError(f"visual key sets differ: missing={sorted(set(indexed)-set(trained_keys))}, extra={sorted(set(trained_keys)-set(indexed))}")
        for filename in files:
            with open_tensors(str(base_dir / filename), framework="pt", device="cpu") as base:
                for name in sorted(indexed):
                    if index[indexed[name]] != filename:
                        continue
                    tensor = trained.get_tensor(trained_keys[name])
                    if tensor.dtype != torch.float32:
                        raise ValueError(f"expected FP32 intermediate visual checkpoint: {name} is {tensor.dtype}")
                    delta, stats = tensor_delta(base.get_tensor(indexed[name]), tensor, base_load_dtype)
                    if not stats["reconstruction_bit_exact"]:
                        raise ValueError(f"dense delta did not reconstruct checkpoint exactly: {name}")
                    tensors.append(_tensor_row(
                        stats, name=name, trained_key=trained_keys[name], base_key=indexed[name],
                        matrix_kind=matrix_kind(name, delta),
                    ))
                    deltas[name] = delta
    dense_path = output_dir / "visual_delta.safetensors"
    save_file(deltas, dense_path, metadata={"schema": report["schema"], "hf_revision": revision,
                                         "operation": "trained_minus_base_loaded", "base_load_dtype": base_load_dtype})
    del deltas
    report["dense_delta_sha256"] = sha256_file(dense_path)
    report["status"] = "decomposing"
    report["source_dtypes"] = dict(Counter(r["trained_storage_dtype"] for r in tensors))
    write_json(output_dir / "report.json", report)
    factor_dir = output_dir / "svd_factors"
    factor_dir.mkdir()
    rank16_linear: dict[str, torch.Tensor] = {}
    matrices = [r for r in tensors if r["matrix_kind"] is not None]
    with open_tensors(str(dense_path), framework="pt", device="cpu") as dense:
        for i, row in enumerate(matrices, 1):
            tensor = dense.get_tensor(row["name"])
            matrix = tensor.reshape(tensor.shape[0], -1)
            before = time.monotonic()
            svd, factors = decompose_matrix(matrix, factor_rank)
            row["svd"] = svd
            filename = row["name"] + ".safetensors"
            save_file(factors, factor_dir / filename,
                      metadata={"source_tensor": row["name"], "matrix_kind": str(row["matrix_kind"]),
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
    report["summary"] = summarize(tensors)
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
