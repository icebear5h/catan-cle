"""summaries."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from ._base import RANKS, DeltaReport, GroupSummary, TensorRow


def summarize(rows: list[TensorRow]) -> dict[str, GroupSummary]:
    result: dict[str, GroupSummary] = {}
    groups = {"all": rows, "linear_weights": [r for r in rows if r.get("matrix_kind") == "linear_weight"]}
    for group in ("tower", "merger"):
        groups[group] = [r for r in rows if r["component"] == group]
    for name, selected in groups.items():
        matrices = [(r, r["svd"]) for r in selected if "svd" in r]
        energy = sum(r["delta_energy"] for r, _ in matrices)
        result[name] = {
            "tensor_count": len(selected),
            "numel": sum(r["numel"] for r in selected),
            "matrix_count": len(matrices),
            "delta_norm": math.sqrt(sum(r["delta_energy"] for r in selected)),
            "matrix_delta_energy": energy,
            "nonmatrix_delta_energy": sum(r["delta_energy"] for r in selected if "svd" not in r),
            "matrix_energy_weighted_rank_capture": {
                str(rank): sum(r["delta_energy"] * svd["rank_energy_fraction"][str(rank)] for r, svd in matrices) / energy
                if energy else 1.0 for rank in RANKS
            },
        }
    return result


def write_summary_tables(output_dir: Path, report: DeltaReport) -> None:
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
            svd = row["svd"]
            record: dict[str, object] = {
                "name": row["name"], "component": row["component"],
                "matrix_kind": row["matrix_kind"], "shape": row["shape"],
                "delta_norm": row["delta_norm"], "relative_delta_norm": row["relative_delta_norm"],
            }
            for rank in (8, 16, 32, 64, 128, 256):
                record[f"rank{rank}_energy"] = svd["rank_energy_fraction"][str(rank)]
            for percent in (90, 95, 99):
                record[f"rank{percent}"] = svd["energy_threshold_ranks"][str(percent / 100)]
            writer.writerow(record)
    lines = ["# Visual delta extraction", "", f"Base: `{report['model_id']}@{report['hf_revision']}`.",
             "", "Dense delta = FP32 checkpoint minus original HF tensors cast through "
             f"`{report['base_load_dtype']}` (the trainer's base-load dtype).", "",
             "Dense deltas are FP64. Adding to the FP64-promoted loaded base and casting to FP32 "
             "reconstructs every saved checkpoint tensor bit-for-bit. SVD uses FP32 matrices.", "",
             "| Scope | Matrices | Rank 8 energy | Rank 16 | Rank 64 | Rank 256 |",
             "|---|---:|---:|---:|---:|---:|"]
    for group, summary in report["summary"].items():
        capture = summary["matrix_energy_weighted_rank_capture"]
        values = " | ".join(f"{100*capture[str(r)]:.2f}%" for r in (8, 16, 64, 256))
        lines.append(f"| {group} | {summary['matrix_count']} | {values} |")
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
