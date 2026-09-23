"""Per-behavior accuracy summaries and their deterministic fingerprint."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Iterable, Sequence

from sft.json_types import JsonValue, as_dict, opt_dict

from ._types import BEHAVIOR_DESCRIPTIONS, JsonDict


def _section(record: JsonDict, key: str) -> JsonDict:
    return opt_dict(record.get(key) or None) or {}


def _normalized_expected(record: JsonDict) -> str:
    score = _section(record, "score")
    return str(score.get("expected_normalized") or record.get("expected") or "").strip().lower()


def behavior_names(record: JsonDict) -> tuple[str, ...]:
    """Return every curated behavior matched by an evaluation record."""

    metadata = _section(record, "metadata")
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
        correct = sum(bool(_section(row, "score").get("correct")) for row in rows)
        candidates = [as_dict(row["candidate_score"]) for row in rows if row.get("candidate_score")]
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

    rows: list[dict[str, JsonValue]] = []
    for record in records:
        if behavior is not None and behavior not in behavior_names(record):
            continue
        metadata = _section(record, "metadata")
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
