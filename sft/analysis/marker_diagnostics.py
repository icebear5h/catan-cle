"""Deterministic, paired marker sampling and CPU-only failure analysis."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from sft.json_types import (
    JsonDict,
    JsonList,
    as_dict,
    as_float,
    as_list,
    as_str,
    json_dict,
    json_list,
    loads_json,
)

DIRECTIONS = ("marker_to_token", "token_to_marker")
TOKEN = re.compile(r"^<[NETP][0-9_]+>$")


def read_rows(path: str | Path) -> list[JsonDict]:
    return [as_dict(loads_json(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def image_ref(row: JsonDict) -> str:
    return str(row.get("image") or as_list(row["images"])[0])


def select_rows(rows: list[JsonDict]) -> list[JsonDict]:
    """One paired image per token, assigned round-robin to sorted boards."""
    keys = [
        (as_str(r["state_id"]), as_str(r["target_token"]), as_str(r["task_type"])) for r in rows
    ]
    if len(set(keys)) != len(keys) or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source row or token/direction/board")
    if any(r["task_type"] not in DIRECTIONS for r in rows):
        raise ValueError("source must contain only marker tasks")
    boards = sorted({as_str(r["state_id"]) for r in rows})
    tokens = sorted({as_str(r["target_token"]) for r in rows})
    if not boards or not tokens:
        raise ValueError("empty marker source")
    lookup = dict(zip(keys, rows, strict=True))
    selected: list[JsonDict] = []
    for index, token in enumerate(tokens):
        board = boards[index % len(boards)]
        try:
            pair = [lookup[board, token, direction] for direction in DIRECTIONS]
        except KeyError as exc:
            raise ValueError(f"missing paired marker row: {exc}") from exc
        if image_ref(pair[0]) != image_ref(pair[1]) or pair[0]["marker"] != pair[1]["marker"]:
            raise ValueError(f"directions do not share a marked image: {token}")
        selected.extend(pair)
    return selected


def selection_manifest(source: list[JsonDict], selected: list[JsonDict]) -> JsonDict:
    ids = [as_str(r["row_id"]) for r in selected]
    return {
        "schema": "catan_marker_mini_selection/v1",
        "source_rows": len(source),
        "rows": len(selected),
        "tokens": len({as_str(r["target_token"]) for r in selected}),
        "boards": json_dict(dict(sorted(Counter(as_str(r["state_id"]) for r in selected).items()))),
        "entities": json_dict(
            dict(sorted(Counter(as_str(r["entity_type"]) for r in selected).items()))
        ),
        "directions": json_dict(Counter(as_str(r["task_type"]) for r in selected)),
        "row_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "row_ids": json_list(ids),
    }


def counted(value: float | None) -> float:
    """Read one accumulated bucket counter, which is never absent while counting."""
    if value is None:
        raise ValueError("bucket counter is missing")
    return value


def normalize(text: object) -> str:
    return str(text).split("<|im_end|>", 1)[0].strip()


def analyze(
    source: list[JsonDict],
    selected: list[JsonDict],
    records: list[JsonDict],
) -> JsonDict:
    """Never interpret occupancy-only counters as marker failure categories."""
    by_id = {as_str(r["row_id"]): r for r in selected}
    ids = [as_str(r["id"]) for r in records]
    if len(set(ids)) != len(ids) or set(ids) - set(by_id):
        raise ValueError("duplicate or unexpected prediction IDs")
    centers: dict[tuple[str, str], list[float]] = {}
    marker_maps: defaultdict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in source:
        state_id, target_token = as_str(row["state_id"]), as_str(row["target_token"])
        center = as_dict(as_list(row["spatial_targets"])[0])["center"]
        centers[state_id, target_token] = [as_float(value) for value in as_list(center)]
        marker_maps[state_id, image_ref(row)][as_str(row["marker"])] = target_token
    tokens = {as_str(r["target_token"]) for r in source}
    buckets: defaultdict[str, defaultdict[str, dict[str, float | None]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "errors": 0, "candidate_n": 0, "candidate_errors": 0})
    )
    failures: JsonList = []
    modes: Counter[str] = Counter()
    disagreement: list[str] = []
    for record in records:
        record_id = as_str(record["id"])
        row = by_id[record_id]
        expected = as_str(as_dict(as_list(row["messages"])[1])["content"]).strip()
        if normalize(record["expected"]) != expected:
            raise ValueError(f"prediction target mismatch: {record['id']}")
        response = normalize(record["response"])
        # Match the existing evaluator's normalized exact-match policy.
        response = as_str(as_dict(record.get("score", {})).get("response_normalized", response))
        correct = response == expected
        candidate_score = record.get("candidate_score")
        candidate = as_dict(candidate_score) if candidate_score else None
        token, board = as_str(row["target_token"]), as_str(row["state_id"])
        direction, entity = as_str(row["task_type"]), as_str(row["entity_type"])
        orientation = None
        if entity == "edge":
            a, b = token[2:-1].split("_")
            ax, ay = centers[board, f"<N{a}>"]
            bx, by = centers[board, f"<N{b}>"]
            orientation = "vertical" if math.degrees(math.atan2(abs(bx - ax), abs(by - ay))) <= 15 else "slanted"
        dimensions = {"overall": "all", "direction": direction, "entity": entity, "direction_entity": f"{direction}/{entity}", "token": token, "board": board}
        if orientation:
            dimensions.update(orientation=orientation, direction_orientation=f"{direction}/{orientation}")
        for dimension, value in dimensions.items():
            bucket = buckets[dimension][value]
            bucket["n"] = counted(bucket["n"]) + 1
            bucket["errors"] = counted(bucket["errors"]) + (not correct)
            if candidate:
                bucket["candidate_n"] = counted(bucket["candidate_n"]) + 1
                bucket["candidate_errors"] = (
                    counted(bucket["candidate_errors"]) + (not candidate["correct"])
                )
        if candidate and bool(candidate["correct"]) != correct:
            disagreement.append(record_id)
        if correct:
            continue
        predicted_token: str | None = None
        if direction == "token_to_marker":
            if TOKEN.fullmatch(response):
                mode = "wrong_answer_type"
            elif response in "ABCD" and len(response) == 1:
                mode = "wrong_marker_letter"
                predicted_token = marker_maps[board, image_ref(row)].get(response)
            else:
                mode = "malformed_output"
        elif response in "ABCD" and len(response) == 1:
            mode = "wrong_answer_type"
        elif not TOKEN.fullmatch(response):
            mode = "repeated_token" if len(re.findall(r"<[NETP][0-9_]+>", response)) > 1 else "malformed_output"
        elif response not in tokens:
            mode = "unknown_atlas_token"
        elif response[1] != token[1]:
            mode = "wrong_entity_type"
            predicted_token = response
        else:
            mode = "wrong_location"
            predicted_token = response
        modes[mode] += 1
        distance = None
        if predicted_token and (board, predicted_token) in centers:
            distance = math.dist(centers[board, token], centers[board, predicted_token]) * 1024
        failures.append({
            "row_id": record["id"], "task_type": direction, "entity_type": entity,
            "target_token": token, "orientation": orientation, "mode": mode,
            "expected": expected, "response": response, "raw_response": record["response"],
            "candidate_score": candidate_score, "predicted_location": predicted_token,
            "other_marked_location": (
                predicted_token in as_list(row["marker_group"]) if predicted_token else None
            ),
            "distance_px_1024": round(distance, 1) if distance is not None else None,
            "image": image_ref(row), "state_id": board,
        })
    for entries in buckets.values():
        for bucket in entries.values():
            bucket["accuracy"] = 1 - counted(bucket["errors"]) / counted(bucket["n"])
            candidate_total = counted(bucket["candidate_n"])
            bucket["candidate_accuracy"] = (
                1 - counted(bucket["candidate_errors"]) / candidate_total
                if candidate_total
                else None
            )
    by_bucket: JsonDict = {
        dimension: {value: json_dict(bucket) for value, bucket in entries.items()}
        for dimension, entries in buckets.items()
    }
    return {
        "schema": "catan_marker_mini_failures/v1", "complete": set(ids) == set(by_id),
        "expected_rows": len(selected), "scored_rows": len(records),
        "missing_ids": json_list(sorted(set(by_id) - set(ids))), "by": by_bucket,
        "failure_modes": json_dict(modes), "failures": failures,
        "free_candidate_disagreement_ids": json_list(disagreement),
        "limitations": "One image per token; marker localization only, not actual-piece transfer or causal shape/init attribution.",
    }
