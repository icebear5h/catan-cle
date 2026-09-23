from __future__ import annotations

from collections import Counter, defaultdict

from sft.json_types import JsonDict, JsonValue, as_dict, as_int


def summarize_dimension(
    attempted: list[JsonDict],
    metadata_key: str,
) -> JsonDict:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for record in attempted:
        value = str(as_dict(record["metadata"]).get(metadata_key) or "unknown")
        grouped[value]["total"] += 1
        if as_dict(record["score"])["correct"]:
            grouped[value]["correct"] += 1
        candidate = record.get("candidate_score")
        if candidate is not None:
            candidate = as_dict(candidate)
            grouped[value]["candidate_total"] += 1
            grouped[value]["candidate_rank_sum"] += as_int(candidate["expected_rank"])
            if candidate["correct"]:
                grouped[value]["candidate_correct"] += 1

    result: JsonDict = {}
    for value, counts in sorted(grouped.items()):
        total = counts["total"]
        correct = counts["correct"]
        entry: JsonDict = {
            "total": total,
            "correct": correct,
            "exact_accuracy": correct / total if total else 0.0,
        }
        candidate_total = counts["candidate_total"]
        if candidate_total:
            entry.update(
                {
                    "candidate_total": candidate_total,
                    "candidate_correct": counts["candidate_correct"],
                    "candidate_exact_accuracy": counts["candidate_correct"] / candidate_total,
                    "candidate_expected_rank_mean": counts["candidate_rank_sum"] / candidate_total,
                }
            )
        result[value] = entry
    return result


NEIGHBOR_CONFUSION_CATEGORIES = ("node.occupancy", "edge.owner")
NEIGHBOR_CONFUSION_BUCKETS = ("names_target", "names_partner", "answers_empty", "other")


def piece_answer(color: JsonValue, piece: JsonValue) -> str | None:
    """Render metadata color/piece as the answer string occupancy rows expect."""

    if not color or not piece:
        return None
    return f"{str(color).replace('_', ' ')} {str(piece)}".lower()


def is_neighbor_confusion_record(record: JsonDict) -> bool:
    metadata = as_dict(record.get("metadata") or {})
    if metadata.get("category") in NEIGHBOR_CONFUSION_CATEGORIES:
        return True
    return str(metadata.get("task_type") or "").startswith("occupancy_")


def summarize_neighbor_confusion(records: list[JsonDict]) -> JsonDict:
    """Bucket wrong occupancy/owner answers by what the model named instead.

    Reads only row metadata (target and partner color/piece plus distances),
    so it needs no board graph at eval time. Rows without partner fields land
    in names_target / answers_empty / other; rows lacking any of the fields
    simply skip the buckets they cannot support.
    """

    groups: dict[str, Counter[str]] = defaultdict(Counter)
    by_distance: dict[str, dict[str, Counter[str]]] = {
        "names_target": defaultdict(Counter),
        "names_partner": defaultdict(Counter),
    }
    for record in records:
        if not is_neighbor_confusion_record(record):
            continue
        metadata = as_dict(record.get("metadata") or {})
        score = as_dict(record.get("score") or {})
        group = str(metadata.get("task_type") or metadata.get("category") or "unknown")
        groups[group]["total"] += 1
        if score.get("correct"):
            continue
        groups[group]["total_wrong"] += 1

        response = str(score.get("response_normalized") or "").strip().lower()
        expected = str(score.get("expected_normalized") or "").strip().lower()
        target = piece_answer(metadata.get("color"), metadata.get("piece"))
        partner = piece_answer(metadata.get("partner_color"), metadata.get("partner_piece"))
        if metadata.get("negative_distance") is not None:
            distance = metadata.get("negative_distance")
        else:
            distance = metadata.get("partner_distance")

        if target is not None and response == target:
            bucket = "names_target"
        elif partner is not None and response == partner:
            bucket = "names_partner"
        elif response == "empty" and expected != "empty":
            bucket = "answers_empty"
        else:
            bucket = "other"
        groups[group][bucket] += 1
        if bucket in by_distance:
            by_distance[bucket][group][str(distance)] += 1

    result: JsonDict = {}
    for group, counts in sorted(groups.items()):
        entry: JsonDict = {
            "total": counts["total"],
            "total_wrong": counts["total_wrong"],
        }
        for bucket in NEIGHBOR_CONFUSION_BUCKETS:
            entry[bucket] = counts[bucket]
        distances: JsonDict = {}
        for bucket in by_distance:
            tallies: JsonDict = dict(sorted(by_distance[bucket][group].items()))
            distances[bucket] = tallies
        entry["by_distance"] = distances
        result[group] = entry
    return result
