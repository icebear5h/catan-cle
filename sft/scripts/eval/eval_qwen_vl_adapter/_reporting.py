from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone

from sft.analysis.behavior_diagnostics import summarize_behaviors
from sft.board.board_fluency_scoring import SCHEMAS as SCHEMAS
from sft.board.board_fluency_scoring import validate_board_fluency_metadata
from sft.board.coordinate_comparison import SCHEMA as SCHEMA
from sft.board.coordinate_comparison import paired_summary, validate_comparison_metadata
from sft.board.spatial_tasks import TASK_TYPES as TASK_TYPES
from sft.board.symbolic_board_tasks import SYMBOLIC_TASKS, symbolic_task_role
from sft.board_state_readout import TASK as TASK
from sft.board_state_readout import summarize_board_states
from sft.json_types import JsonDict, as_bool, as_dict, as_float, as_int, as_str, json_dict

from ._summaries import summarize_dimension, summarize_neighbor_confusion

BOARD_FLUENCY_SCHEMAS = SCHEMAS
FULL_BOARD_TASK = TASK
COORDINATE_COMPARISON_SCHEMA = SCHEMA
SPATIAL_TASK_TYPES = TASK_TYPES

OCCUPANCY_CATEGORIES = ("node.occupancy", "edge.owner")


def occupancy_class(answer: str) -> str:
    """``empty``, or the piece word that ends an occupancy answer (``settlement``, ``city``, ``road``)."""

    words = answer.strip().lower().split()
    return "empty" if not words or words[-1] == "empty" else words[-1]


def _section(record: JsonDict, key: str) -> JsonDict:
    return as_dict(record.get(key, {}))


def summarize_occupancy_classes(records: list[JsonDict]) -> JsonDict:
    """Per-class recall and precision for the occupancy heads, and their balanced accuracy.

    Real boards are mostly empty, so exact accuracy on these heads rewards a
    model that answers ``empty`` everywhere; recall per piece class and the
    precision of ``empty`` are the numbers that mean something.
    """

    expected_counts: dict[str, int] = {}
    predicted_counts: dict[str, int] = {}
    correct_counts: dict[str, int] = {}
    for record in records:
        if _section(record, "metadata").get("category") not in OCCUPANCY_CATEGORIES:
            continue
        score = as_dict(record["score"])
        expected = occupancy_class(str(score.get("expected_normalized", "")))
        predicted = occupancy_class(str(score.get("response_normalized", "")))
        expected_counts[expected] = expected_counts.get(expected, 0) + 1
        predicted_counts[predicted] = predicted_counts.get(predicted, 0) + 1
        if score["correct"]:
            correct_counts[expected] = correct_counts.get(expected, 0) + 1
    names = sorted(set(expected_counts) | set(predicted_counts))
    recall = {name: correct_counts.get(name, 0) / expected_counts[name]
              for name in names if expected_counts.get(name, 0)}
    precision = {name: correct_counts.get(name, 0) / predicted_counts[name]
                 for name in names if predicted_counts.get(name, 0)}
    classes: JsonDict = {
        name: {
            "expected": expected_counts.get(name, 0),
            "predicted": predicted_counts.get(name, 0),
            "correct": correct_counts.get(name, 0),
            "recall": recall.get(name),
            "precision": precision.get(name),
        }
        for name in names
    }
    recalls = list(recall.values())
    occupied_expected = sum(expected_counts.get(name, 0) for name in names if name != "empty")
    occupied_correct = sum(correct_counts.get(name, 0) for name in names if name != "empty")
    return {
        "classes": classes,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "occupied_recall": occupied_correct / occupied_expected if occupied_expected else None,
        "empty_precision": precision.get("empty") if "empty" in names else None,
    }


def summarize_readout_items(records: list[JsonDict]) -> JsonDict:
    """Occupied and empty item rates per readout category."""

    totals: dict[str, dict[str, int]] = {}
    for record in records:
        score = as_dict(record["score"])
        if "occupied_items_total" not in score:
            continue
        category = str(_section(record, "metadata").get("category", "readout"))
        entry = totals.setdefault(category, {"readouts": 0, "exact": 0, "occupied_correct": 0, "occupied_total": 0, "empty_correct": 0, "empty_total": 0, "extra": 0})
        entry["readouts"] += 1
        entry["exact"] += int(as_bool(score["correct"]))
        entry["occupied_correct"] += as_int(score["occupied_items_correct"])
        entry["occupied_total"] += as_int(score["occupied_items_total"])
        entry["empty_correct"] += as_int(score["empty_items_correct"])
        entry["empty_total"] += as_int(score["empty_items_total"])
        entry["extra"] += as_int(score["items_extra"])
    result: JsonDict = {}
    for category, entry in totals.items():
        result[category] = {
            **entry,
            "occupied_item_recall": entry["occupied_correct"] / entry["occupied_total"] if entry["occupied_total"] else None,
            "empty_item_accuracy": entry["empty_correct"] / entry["empty_total"] if entry["empty_total"] else None,
        }
    return result


def _scoring(record: JsonDict) -> JsonDict:
    return as_dict(record["score"])


def summarize(records: list[JsonDict]) -> JsonDict:
    attempted = [record for record in records if record.get("response") is not None]

    correct_total = sum(as_bool(_scoring(record)["correct"]) for record in attempted)
    summary: JsonDict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(records),
        "attempted": len(attempted),
        "correct": correct_total,
        "exact_accuracy": correct_total / len(attempted) if attempted else 0.0,
    }
    scored = [as_dict(record["candidate_score"]) for record in attempted if record.get("candidate_score")]
    if scored:
        hits = sum(as_bool(item["correct"]) for item in scored)
        ranks = [as_float(item["expected_rank"]) for item in scored]
        summary.update(
            {
                "candidate_rows": len(scored),
                "candidate_correct": hits,
                "candidate_exact_accuracy": hits / len(scored),
                "candidate_expected_rank_mean": sum(ranks) / len(scored),
                "candidate_expected_rank_le3": sum(rank <= 3 for rank in ranks) / len(scored),
            }
        )
    for key in (
        "category",
        "suite",
        "density_bin",
        "row_kind",
        "curriculum_stage",
        "task_family",
        "task_type",
        "family",
        "operation",
        "entity_type",
        "relationship",
        "polarity",
        "marker_style",
        "probe_style",
        "piece",
        "color",
        "color_heldout",
        "grounding_stage",
        "eval_variant",
        "pair_kind",
        "negative_distance",
        "representation",
    ):
        summary[f"by_{key}"] = summarize_dimension(attempted, key)
    summary["categories"] = summary["by_category"]
    summary["occupancy_classes"] = summarize_occupancy_classes(attempted)
    summary["readout_items"] = summarize_readout_items(attempted)
    board_records = [r for r in attempted if _scoring(r).get("scoring") == FULL_BOARD_TASK]
    if board_records:
        summary["full_board"] = summarize_board_states(board_records)
    summary["neighbor_confusion"] = summarize_neighbor_confusion(attempted)
    summary["by_behavior"] = json_dict(summarize_behaviors(attempted))
    comparison = [r for r in attempted if _scoring(r).get("scoring") == COORDINATE_COMPARISON_SCHEMA]
    if comparison:
        summary["coordinate_comparison"] = as_dict(paired_summary(comparison))
    symbolic = [r for r in attempted if _scoring(r).get("scoring") in SYMBOLIC_TASKS]
    if symbolic:
        families: defaultdict[str, list[JsonDict]] = defaultdict(list)
        for record in symbolic:
            families[as_str(_scoring(record)["scoring"])].append(record)
        symbolic_families: JsonDict = {}
        summary["symbolic_families"] = symbolic_families
        for family, group in sorted(families.items()):
            weighted = ["macro_weight" in as_dict(r["metadata"]) for r in group]
            if any(weighted) and not all(weighted):
                raise ValueError("incomplete symbolic macro weights")
            raw_weights = [as_dict(r["metadata"])["macro_weight"] if all(weighted) else 1.0 for r in group]
            weights = [w for w in raw_weights if isinstance(w, (float, int)) and type(w) in (float, int)]
            if len(weights) != len(raw_weights) or any(not math.isfinite(w) or w <= 0 for w in weights):
                raise ValueError("invalid symbolic macro weights")
            outcomes = [as_bool(_scoring(r)["correct"]) for r in group]
            symbolic_families[family] = {
                "rows": len(group), "correct": sum(outcomes),
                "accuracy": sum(w * hit for w, hit in zip(weights, outcomes, strict=True)) / sum(weights),
                "weighting": "state_mode_polarity_macro" if all(weighted) else "row",
                "by_mode": summarize_dimension(group, "evaluation_mode"),
                "by_polarity": summarize_dimension(group, "polarity"),
            }
    return summary


def evaluation_metadata(row: JsonDict, *, image_variant: str) -> JsonDict:
    metadata = dict(as_dict(row.get("metadata", {})))
    schemas = [source["schema"] for source in (row, metadata) if "schema" in source]
    if COORDINATE_COMPARISON_SCHEMA in schemas:
        if any(schema != COORDINATE_COMPARISON_SCHEMA for schema in schemas):
            raise ValueError("conflicting coordinate-comparison schema declarations")
        validate_comparison_metadata(metadata)
    if (any(schema in BOARD_FLUENCY_SCHEMAS for schema in schemas)
            or any(str(schema).startswith("catan_board_fluency") for schema in schemas)
            or row.get("class") == "board_fluency" or metadata.get("class") == "board_fluency"):
        if not schemas or any(schema != schemas[0] for schema in schemas):
            raise ValueError("missing/conflicting board-fluency schema declarations")
        metadata["schema"] = schemas[0]
        for key in ("split", "task_role", "review_only", "admitted_for_training", "class",
                    "family", "operation", "target", "answer", "provenance"):
            if key in row:
                if key in metadata and (type(row[key]) is not type(metadata[key])
                                        or row[key] != metadata[key]):
                    raise ValueError(f"conflicting board-fluency {key}")
                metadata[key] = row[key]
        validate_board_fluency_metadata(metadata)
    declared = [source["task_type"] for source in (row, metadata) if "task_type" in source]
    spatial = [value for value in declared + [row.get("category"), metadata.get("category")]
               if value in SPATIAL_TASK_TYPES or value in SYMBOLIC_TASKS]
    if spatial and any(value != spatial[0] for value in declared + spatial):
        raise ValueError("conflicting spatial task declarations")
    if spatial and spatial[0] in SYMBOLIC_TASKS:
        for key in ("training_family", "task_role", "split"):
            if key in row and key in metadata and row[key] != metadata[key]:
                raise ValueError(f"conflicting symbolic {key}")
            if key in row:
                metadata[key] = row[key]
        if metadata.get("training_family", spatial[0]) != spatial[0]:
            raise ValueError("conflicting symbolic training family")
        if "task_role" in metadata or "split" in metadata:
            if metadata.get("task_role") != symbolic_task_role(spatial[0], metadata.get("split")):
                raise ValueError("conflicting symbolic task role")
    for key in (
        "category",
        "suite",
        "density_bin",
        "row_kind",
        "curriculum_stage",
        "grounding_stage",
        "task_family",
        "task_type",
        "entity_type",
        "relationship",
        "polarity",
        "marker_style",
        "probe_style",
        "piece",
        "color",
        "color_heldout",
        "target_token",
        "queried_token",
        "pair_kind",
        "partner_token",
        "partner_piece",
        "partner_color",
        "partner_distance",
        "same_color",
        "negative_distance",
        "negative_kind",
        "state_id",
        "layout_id",
        "piece_count",
        "item_count",
        "occupied_count",
        "eval_set_id",
        "eval_source_sha256",
    ):
        if key in row:
            metadata[key] = row[key]
    metadata["eval_variant"] = image_variant
    return metadata
