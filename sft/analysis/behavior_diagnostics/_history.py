"""Checkpoint retention matrices built from per-set evaluation receipts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

from sft.json_types import JsonList, JsonValue, as_dict, as_float, load_json_dict, loads_json

from ._summaries import records_fingerprint, summarize_behaviors
from ._types import JsonDict


def infer_eval_set_id(summary_path: Path, summary: JsonDict) -> str:
    """Recover a stable set label, including for historical panel outputs."""

    explicit = summary.get("eval_set_id")
    if explicit:
        return str(explicit)
    parent = summary_path.parent
    if parent.name in {"original", "blank", "shuffle", "target_occlusion", "control_occlusion"}:
        parent = parent.parent
    name = parent.name
    for prefix in ("regression-panel-", "pairs-v1-", "pairs-v2-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.removesuffix("-eval") or "unknown_eval_set"


def load_checkpoint_results(label: str, roots: Sequence[Path]) -> JsonDict:
    """Load all per-set summaries and records for one checkpoint label."""

    cells: JsonDict = {}
    adapter_dirs: set[str] = set()
    sources: JsonList = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for summary_path in sorted(root.rglob("summary.json")):
            records_path = summary_path.with_name("records.jsonl")
            if not records_path.is_file():
                raise FileNotFoundError(records_path)
            summary = load_json_dict(summary_path)
            if summary.get("adapter_dir"):
                adapter_dirs.add(str(summary["adapter_dir"]))
            variant = str(summary.get("image_variant") or summary_path.parent.name)
            if variant != "original":
                continue
            records = [
                as_dict(loads_json(line))
                for line in records_path.read_text().splitlines()
                if line.strip()
            ]
            eval_set = infer_eval_set_id(summary_path, summary)
            eval_set_fingerprint = records_fingerprint(records)
            sources.append(str(summary_path))
            for behavior, metrics in summarize_behaviors(records).items():
                key = f"{eval_set}/{behavior}"
                if key in cells:
                    raise ValueError(f"duplicate behavior cell for {label}: {key}")
                cells[key] = {
                    **metrics,
                    "eval_set_id": eval_set,
                    "behavior": behavior,
                    "eval_set_fingerprint": eval_set_fingerprint,
                    "sample_fingerprint": records_fingerprint(records, behavior),
                    "summary_path": str(summary_path),
                }
    if len(adapter_dirs) > 1:
        raise ValueError(f"checkpoint {label!r} mixes adapters: {sorted(adapter_dirs)}")
    return {
        "label": label,
        "adapter_dir": next(iter(adapter_dirs), None),
        "sources": sources,
        "cells": cells,
    }


def build_behavior_history(checkpoints: Sequence[JsonDict]) -> JsonDict:
    """Build accuracy and best-so-far forgetting cells in checkpoint order."""

    labels = [str(item["label"]) for item in checkpoints]
    if len(labels) != len(set(labels)):
        raise ValueError("checkpoint labels must be unique")
    # Historical ad-hoc eval roots did not persist a stable eval_set_id. Align
    # those roots to a prior set when both the behavior and exact scored-row
    # fingerprint agree. Reusing the same set label with different rows still
    # remains a fail-closed comparison error below.
    identity_aliases: dict[tuple[str, str], str] = {}
    key_fingerprints: dict[str, str] = {}
    normalized_cells: list[dict[str, JsonDict]] = []
    alias_report: JsonList = []
    for checkpoint in checkpoints:
        normalized: dict[str, JsonDict] = {}
        for source_key, raw_cell in as_dict(checkpoint["cells"]).items():
            cell = as_dict(raw_cell)
            behavior = str(cell.get("behavior") or source_key.rsplit("/", 1)[-1])
            identity = (
                behavior,
                str(cell.get("eval_set_fingerprint") or cell["sample_fingerprint"]),
            )
            canonical_key = identity_aliases.get(identity)
            if canonical_key is None:
                canonical_key = source_key
                if source_key not in key_fingerprints:
                    identity_aliases[identity] = canonical_key
                    key_fingerprints[source_key] = identity[1]
            if canonical_key in normalized:
                raise ValueError(
                    f"checkpoint {checkpoint['label']!r} has duplicate comparable cells: "
                    f"{canonical_key}"
                )
            normalized[canonical_key] = cell
            if canonical_key != source_key:
                alias_report.append(
                    {
                        "checkpoint": checkpoint["label"],
                        "source": source_key,
                        "canonical": canonical_key,
                    }
                )
        normalized_cells.append(normalized)

    behavior_keys = sorted({key for cells in normalized_cells for key in cells})
    behavior_rows: JsonDict = {}
    errors: JsonList = []

    for key in behavior_keys:
        canonical_fingerprint: str | None = None
        best = -math.inf
        row_cells: list[JsonValue] = []
        for checkpoint, cells in zip(checkpoints, normalized_cells, strict=True):
            source = cells.get(key)
            if source is None:
                row_cells.append({"status": "missing"})
                continue
            fingerprint = str(source["sample_fingerprint"])
            if canonical_fingerprint is None:
                canonical_fingerprint = fingerprint
            if fingerprint != canonical_fingerprint:
                message = (
                    f"{key} uses non-comparable rows at {checkpoint['label']}: "
                    f"{fingerprint[:12]} != {canonical_fingerprint[:12]}"
                )
                errors.append(message)
                row_cells.append({**source, "status": "incomparable", "reason": message})
                continue
            accuracy = as_float(source["exact_accuracy"])
            best = max(best, accuracy)
            row_cells.append(
                {
                    **source,
                    "status": "ok",
                    "best_so_far": best,
                    "forgetting": best - accuracy,
                }
            )
        behavior_rows[key] = {
            "description": next(
                (
                    found["description"]
                    for cells in normalized_cells
                    if (found := cells.get(key)) is not None
                ),
                "",
            ),
            "sample_fingerprint": canonical_fingerprint,
            "checkpoints": dict(zip(labels, row_cells, strict=True)),
        }

    checkpoint_order: JsonList = list(labels)
    checkpoint_sources: JsonList = [
        {
            "label": item["label"],
            "adapter_dir": item.get("adapter_dir"),
            "sources": item.get("sources", []),
        }
        for item in checkpoints
    ]
    return {
        "schema": "catan_behavior_checkpoint_matrix/v1",
        "checkpoint_order": checkpoint_order,
        "checkpoints": checkpoint_sources,
        "behaviors": behavior_rows,
        "eval_set_aliases": alias_report,
        "errors": errors,
    }
