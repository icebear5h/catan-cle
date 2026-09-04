"""Behavior-level evaluation summaries and checkpoint retention matrices."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class Behavior:
    """One stable, human-readable capability slice."""

    name: str
    description: str


BEHAVIORS = (
    Behavior("pair.positive", "occupied target on an adjacent-pair image"),
    Behavior("pair.road_positive", "occupied road target on an adjacent-pair image"),
    Behavior("pair.building_positive", "occupied building target on an adjacent-pair image"),
    Behavior("pair.empty", "all empty-location queries on an adjacent-pair image"),
    Behavior("pair.adjacent_empty", "empty location touching a pair member"),
    Behavior("pair.far_empty", "empty location far from both pair members"),
    Behavior("pair.localization", "piece-to-atlas localization on an adjacent-pair image"),
    Behavior("pair.edge_edge_positive", "occupied target on an edge-edge pair"),
    Behavior("single.positive", "occupied target on a single-piece image"),
    Behavior("single.road_positive", "occupied road target on a single-piece image"),
    Behavior("single.building_positive", "occupied building target on a single-piece image"),
    Behavior("single.empty", "all empty-location queries on a single-piece image"),
    Behavior("single.adjacent_empty", "empty location touching a single piece"),
    Behavior("single.far_empty", "empty location far from a single piece"),
    Behavior("single.localization", "piece-to-atlas localization on a single-piece image"),
    Behavior("tile.resource", "tile resource classification"),
    Behavior("tile.number", "tile number classification"),
    Behavior("tile.inverse", "tile description-to-atlas localization"),
    Behavior("full_board.node_occupancy", "node occupancy on production board renders"),
    Behavior("full_board.node_occupied", "occupied-node recall on production board renders"),
    Behavior("full_board.node_empty", "empty-node accuracy on production board renders"),
    Behavior("full_board.edge_owner", "edge owner on production board renders"),
    Behavior("full_board.road_occupied", "occupied-road recall on production board renders"),
    Behavior("full_board.edge_empty", "empty-edge accuracy on production board renders"),
    Behavior("full_board.tile_resource", "tile resource on production board renders"),
    Behavior("full_board.tile_number", "tile number on production board renders"),
    Behavior("full_board.tile_inverse", "inverse tile lookup on production board renders"),
    Behavior("full_board.tile_robber", "tile-level robber presence on production board renders"),
    Behavior("full_board.robber", "robber presence and localization on production board renders"),
    Behavior("full_board.port", "port classification on production board renders"),
    Behavior("full_board.inverse_node", "inverse node lookup on production board renders"),
    Behavior("full_board.inverse_edge", "inverse edge lookup on production board renders"),
    Behavior("full_board.inverse_port", "inverse port lookup on production board renders"),
    Behavior("full_board.spatial_grounding", "graph-relation queries on production board renders"),
)
BEHAVIOR_DESCRIPTIONS = {item.name: item.description for item in BEHAVIORS}


def _normalized_expected(record: JsonDict) -> str:
    score = record.get("score") or {}
    return str(score.get("expected_normalized") or record.get("expected") or "").strip().lower()


def behavior_names(record: JsonDict) -> tuple[str, ...]:
    """Return every curated behavior matched by an evaluation record."""

    metadata = record.get("metadata") or {}
    task_family = str(metadata.get("task_family") or "")
    task_type = str(metadata.get("task_type") or "")
    category = str(metadata.get("category") or "")
    piece = str(metadata.get("piece") or "").upper()
    pair_kind = str(metadata.get("pair_kind") or "")
    expected = _normalized_expected(record)
    names: list[str] = []
    suite = str(metadata.get("suite") or "")
    is_bidirectional = suite == "bidirectional"

    if task_family == "adjacent_pair_localization":
        if task_type == "occupancy_positive":
            names.append("pair.positive")
            names.append("pair.road_positive" if piece == "ROAD" else "pair.building_positive")
            if pair_kind == "edge_edge":
                names.append("pair.edge_edge_positive")
        elif task_type == "occupancy_negative_adjacent":
            names.append("pair.empty")
            names.append("pair.adjacent_empty")
        elif task_type == "occupancy_negative_far":
            names.append("pair.empty")
            names.append("pair.far_empty")
        elif task_type in {"colored_piece_to_token", "piece_to_token"}:
            names.append("pair.localization")

    if task_family == "single_piece_localization":
        if task_type == "occupancy_positive":
            names.append("single.positive")
            names.append("single.road_positive" if piece == "ROAD" else "single.building_positive")
        elif task_type in {"occupancy_negative", "occupancy_negative_adjacent"}:
            names.append("single.empty")
            if task_type == "occupancy_negative_adjacent":
                names.append("single.adjacent_empty")
        elif task_type == "occupancy_negative_far":
            names.append("single.empty")
            names.append("single.far_empty")
        elif task_type in {"colored_piece_to_token", "piece_to_token"}:
            names.append("single.localization")

    if not is_bidirectional and (category == "tile.resource" or task_type == "tile_resource"):
        names.append("tile.resource")
    if not is_bidirectional and (category == "tile.number" or task_type == "tile_number"):
        names.append("tile.number")
    if not is_bidirectional and (category == "inverse.tile" or task_type == "tile_to_token"):
        names.append("tile.inverse")

    # Production-board rows have no curriculum task family. Keeping this
    # condition explicit prevents synthetic single/pair rows from being folded
    # into the deployment-facing heads merely because they share a category.
    if not task_family and is_bidirectional:
        if category == "node.occupancy":
            names.append("full_board.node_occupancy")
            names.append("full_board.node_empty" if expected == "empty" else "full_board.node_occupied")
        elif category == "edge.owner":
            names.append("full_board.edge_owner")
            names.append("full_board.edge_empty" if expected == "empty" else "full_board.road_occupied")
        elif category == "tile.resource":
            names.append("full_board.tile_resource")
        elif category == "tile.number":
            names.append("full_board.tile_number")
        elif category == "inverse.tile":
            names.append("full_board.tile_inverse")
        elif category == "tile.robber":
            names.append("full_board.tile_robber")
        elif category == "port.port_type":
            names.append("full_board.port")
        elif category == "inverse.node":
            names.append("full_board.inverse_node")
        elif category == "inverse.edge":
            names.append("full_board.inverse_edge")
        elif category == "inverse.port":
            names.append("full_board.inverse_port")

    if suite == "spatial_robber":
        if category == "robber":
            names.append("full_board.robber")
        elif category == "spatial_grounding":
            names.append("full_board.spatial_grounding")

    return tuple(dict.fromkeys(names))


def summarize_behaviors(records: Iterable[JsonDict]) -> dict[str, JsonDict]:
    """Summarize exact and candidate accuracy for every curated behavior."""

    groups: dict[str, list[JsonDict]] = defaultdict(list)
    for record in records:
        if record.get("response") is None:
            continue
        for name in behavior_names(record):
            groups[name].append(record)

    result: dict[str, JsonDict] = {}
    for name, rows in sorted(groups.items()):
        correct = sum(bool((row.get("score") or {}).get("correct")) for row in rows)
        candidates = [row["candidate_score"] for row in rows if row.get("candidate_score")]
        entry: JsonDict = {
            "description": BEHAVIOR_DESCRIPTIONS[name],
            "total": len(rows),
            "correct": correct,
            "exact_accuracy": correct / len(rows),
        }
        if candidates:
            candidate_correct = sum(bool(item.get("correct")) for item in candidates)
            entry.update(
                candidate_total=len(candidates),
                candidate_correct=candidate_correct,
                candidate_exact_accuracy=candidate_correct / len(candidates),
            )
        result[name] = entry
    return result


def records_fingerprint(records: Sequence[JsonDict], behavior: str | None = None) -> str:
    """Fingerprint an eval set or the exact examples underlying one behavior."""

    rows = []
    for record in records:
        if behavior is not None and behavior not in behavior_names(record):
            continue
        metadata = record.get("metadata") or {}
        rows.append(
            {
                "id": record.get("id") or record.get("index"),
                "expected": _normalized_expected(record),
                "category": metadata.get("category"),
                "task_type": metadata.get("task_type"),
                "piece": metadata.get("piece"),
                "pair_kind": metadata.get("pair_kind"),
                "variant": metadata.get("eval_variant"),
            }
        )
    encoded = json.dumps(
        sorted(rows, key=lambda item: json.dumps(item, sort_keys=True)),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


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

    cells: dict[str, JsonDict] = {}
    adapter_dirs: set[str] = set()
    sources: list[str] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for summary_path in sorted(root.rglob("summary.json")):
            records_path = summary_path.with_name("records.jsonl")
            if not records_path.is_file():
                raise FileNotFoundError(records_path)
            summary = json.loads(summary_path.read_text())
            if summary.get("adapter_dir"):
                adapter_dirs.add(str(summary["adapter_dir"]))
            variant = str(summary.get("image_variant") or summary_path.parent.name)
            if variant != "original":
                continue
            records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
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
    alias_report: list[JsonDict] = []
    for checkpoint in checkpoints:
        normalized: dict[str, JsonDict] = {}
        for source_key, cell in checkpoint["cells"].items():
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
    behavior_rows: dict[str, JsonDict] = {}
    errors: list[str] = []

    for key in behavior_keys:
        canonical_fingerprint: str | None = None
        best = -math.inf
        row_cells: list[JsonDict] = []
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
            accuracy = float(source["exact_accuracy"])
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
                    cell["description"]
                    for cells in normalized_cells
                    if (cell := cells.get(key)) is not None
                ),
                "",
            ),
            "sample_fingerprint": canonical_fingerprint,
            "checkpoints": dict(zip(labels, row_cells, strict=True)),
        }

    return {
        "schema": "catan_behavior_checkpoint_matrix/v1",
        "checkpoint_order": labels,
        "checkpoints": [
            {
                "label": item["label"],
                "adapter_dir": item.get("adapter_dir"),
                "sources": item.get("sources", []),
            }
            for item in checkpoints
        ],
        "behaviors": behavior_rows,
        "eval_set_aliases": alias_report,
        "errors": errors,
    }


def _format_cell(cell: JsonDict, field: str) -> str:
    if cell.get("status") != "ok":
        return "N/A"
    value = float(cell[field])
    return f"{100.0 * value:.1f}%"


def history_markdown(history: JsonDict) -> str:
    """Render compact accuracy and forgetting tables."""

    labels = history["checkpoint_order"]
    lines = ["# Behavior checkpoint history", "", "## Exact accuracy", ""]
    header = "| Behavior | " + " | ".join(labels) + " |"
    rule = "|---|" + "---:|" * len(labels)
    lines.extend((header, rule))
    for key, row in history["behaviors"].items():
        cells = [_format_cell(row["checkpoints"][label], "exact_accuracy") for label in labels]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")

    lines.extend(("", "## Forgetting: best-so-far minus current", "", header, rule))
    for key, row in history["behaviors"].items():
        cells = [_format_cell(row["checkpoints"][label], "forgetting") for label in labels]
        lines.append(f"| `{key}` | " + " | ".join(cells) + " |")
    if history.get("errors"):
        lines.extend(("", "## Incomparable cells", ""))
        lines.extend(f"- {error}" for error in history["errors"])
    return "\n".join(lines) + "\n"
