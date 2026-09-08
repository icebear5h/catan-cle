"""Deterministic, paired marker sampling and CPU-only failure analysis."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DIRECTIONS = ("marker_to_token", "token_to_marker")
TOKEN = re.compile(r"^<[NETP][0-9_]+>$")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def image_ref(row):
    return row.get("image") or row["images"][0]


def select_rows(rows):
    """One paired image per token, assigned round-robin to sorted boards."""
    keys = [(r["state_id"], r["target_token"], r["task_type"]) for r in rows]
    if len(set(keys)) != len(keys) or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate source row or token/direction/board")
    if any(r["task_type"] not in DIRECTIONS for r in rows):
        raise ValueError("source must contain only marker tasks")
    boards = sorted({r["state_id"] for r in rows})
    tokens = sorted({r["target_token"] for r in rows})
    if not boards or not tokens:
        raise ValueError("empty marker source")
    lookup = dict(zip(keys, rows, strict=True))
    selected = []
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


def selection_manifest(source, selected):
    ids = [r["row_id"] for r in selected]
    return {
        "schema": "catan_marker_mini_selection/v1",
        "source_rows": len(source),
        "rows": len(selected),
        "tokens": len({r["target_token"] for r in selected}),
        "boards": dict(sorted(Counter(r["state_id"] for r in selected).items())),
        "entities": dict(sorted(Counter(r["entity_type"] for r in selected).items())),
        "directions": dict(Counter(r["task_type"] for r in selected)),
        "row_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "row_ids": ids,
    }


def normalize(text):
    return str(text).split("<|im_end|>", 1)[0].strip()


def analyze(source, selected, records):
    """Never interpret occupancy-only counters as marker failure categories."""
    by_id = {r["row_id"]: r for r in selected}
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) - set(by_id):
        raise ValueError("duplicate or unexpected prediction IDs")
    centers = {}
    marker_maps = defaultdict(dict)
    for row in source:
        centers[row["state_id"], row["target_token"]] = row["spatial_targets"][0]["center"]
        marker_maps[row["state_id"], image_ref(row)][row["marker"]] = row["target_token"]
    tokens = {r["target_token"] for r in source}
    buckets = defaultdict(lambda: defaultdict(lambda: {"n": 0, "errors": 0, "candidate_n": 0, "candidate_errors": 0}))
    failures = []
    modes = Counter()
    disagreement = []
    for record in records:
        row = by_id[record["id"]]
        expected = row["messages"][1]["content"].strip()
        if normalize(record["expected"]) != expected:
            raise ValueError(f"prediction target mismatch: {record['id']}")
        response = normalize(record["response"])
        # Match the existing evaluator's normalized exact-match policy.
        response = record.get("score", {}).get("response_normalized", response)
        correct = response == expected
        candidate = record.get("candidate_score")
        token, board, direction = row["target_token"], row["state_id"], row["task_type"]
        orientation = None
        if row["entity_type"] == "edge":
            a, b = token[2:-1].split("_")
            ax, ay = centers[board, f"<N{a}>"]
            bx, by = centers[board, f"<N{b}>"]
            orientation = "vertical" if math.degrees(math.atan2(abs(bx - ax), abs(by - ay))) <= 15 else "slanted"
        dimensions = {"overall": "all", "direction": direction, "entity": row["entity_type"], "direction_entity": f"{direction}/{row['entity_type']}", "token": token, "board": board}
        if orientation:
            dimensions.update(orientation=orientation, direction_orientation=f"{direction}/{orientation}")
        for dimension, value in dimensions.items():
            bucket = buckets[dimension][value]
            bucket["n"] += 1
            bucket["errors"] += not correct
            if candidate:
                bucket["candidate_n"] += 1
                bucket["candidate_errors"] += not candidate["correct"]
        if candidate and bool(candidate["correct"]) != correct:
            disagreement.append(record["id"])
        if correct:
            continue
        predicted_token = None
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
            "row_id": record["id"], "task_type": direction, "entity_type": row["entity_type"],
            "target_token": token, "orientation": orientation, "mode": mode,
            "expected": expected, "response": response, "raw_response": record["response"],
            "candidate_score": candidate, "predicted_location": predicted_token,
            "other_marked_location": predicted_token in row["marker_group"] if predicted_token else None,
            "distance_px_1024": round(distance, 1) if distance is not None else None,
            "image": image_ref(row), "state_id": board,
        })
    for entries in buckets.values():
        for bucket in entries.values():
            bucket["accuracy"] = 1 - bucket["errors"] / bucket["n"]
            bucket["candidate_accuracy"] = 1 - bucket["candidate_errors"] / bucket["candidate_n"] if bucket["candidate_n"] else None
    return {
        "schema": "catan_marker_mini_failures/v1", "complete": set(ids) == set(by_id),
        "expected_rows": len(selected), "scored_rows": len(records),
        "missing_ids": sorted(set(by_id) - set(ids)), "by": dict(buckets),
        "failure_modes": dict(modes), "failures": failures, "free_candidate_disagreement_ids": disagreement,
        "limitations": "One image per token; marker localization only, not actual-piece transfer or causal shape/init attribution.",
    }
