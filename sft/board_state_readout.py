"""Canonical full-board output schema and strict, occupied-aware scoring."""

from __future__ import annotations

import random
import re
from collections import Counter
from typing import TypedDict

from evals.catan_board_bench.tokens import atlas_tokens
from sft.json_types import JsonDict, as_bool, as_dict, as_int, as_str, json_dict, json_list

TASK = "full_board_readout"
PROMPT = (
    "Read the complete board state from the image. List all 19 tiles as token resource number, "
    "all 54 nodes as token building, all 72 edges as token road, and all 9 ports as token port. "
    "Use empty for unoccupied nodes and edges, colour settlement or colour city for buildings, "
    "colour road for roads, and none for the desert number. Order the groups tiles, nodes, "
    "edges, ports, with tokens sorted within each group. Finish with robber followed by its "
    "tile token. Separate all entries with semicolons. Output only the board state."
)


class ParsedState(TypedDict):
    items: list[tuple[str, str]]
    values: dict[str, str]
    duplicates: list[str]
    missing: list[str]
    extra: list[str]
    malformed: list[str]
    complete: bool
    ordered: bool


def board_keys() -> list[str]:
    tokens = atlas_tokens()
    return [token for family in "TNEP" for token in sorted(tokens) if token.startswith("<" + family)] + ["robber"]


def select_validation(rows: list[JsonDict], count: int = 16, seed: int = 43) -> list[JsonDict]:
    """Round-robin layouts and densities; shuffle intact examples within buckets."""
    if not 0 <= count <= len(rows):
        raise ValueError("sample count must be within available rows")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate row IDs")
    rng = random.Random(seed)
    layouts = sorted({as_str(r["layout_id"]) for r in rows})
    rng.shuffle(layouts)
    densities = ("dense", "sparse", "setup", "empty")
    buckets: dict[tuple[str, str], list[JsonDict]] = {}
    for row in sorted(rows, key=lambda r: as_str(r["row_id"])):
        if row["density_bin"] not in densities:
            raise ValueError("unknown density")
        key = (as_str(row["layout_id"]), as_str(row["density_bin"]))
        buckets.setdefault(key, []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[JsonDict] = []
    round_index = 0
    while len(selected) < count:
        for layout in layouts:
            for offset in range(len(densities)):
                bucket = buckets.get((layout, densities[(round_index + offset) % len(densities)]), [])
                if bucket:
                    selected.append(bucket.pop())
                    break
            if len(selected) == count:
                break
        round_index += 1
    return selected


def parse_state(text: str) -> ParsedState:
    items: list[tuple[str, str]] = []
    malformed: list[str] = []
    for entry in text.strip().split(";"):
        match = re.fullmatch(r"\s*(<[TNEP][0-9_]+>|robber)\s+(.+?)\s*", entry, flags=re.S)
        if match is None:
            malformed.append(entry)
        else:
            items.append((match[1], " ".join(match[2].split())))
    counts = Counter(key for key, _ in items)
    duplicates = sorted(key for key, n in counts.items() if n > 1)
    expected_keys = board_keys()
    # Ambiguous repeated addresses never earn item credit.
    values = {key: value for key, value in items if counts[key] == 1}
    missing = sorted(set(expected_keys) - set(counts))
    extra = sorted(set(counts) - set(expected_keys))
    return {"items": items, "values": values, "duplicates": duplicates,
            "missing": missing, "extra": extra, "malformed": malformed,
            "complete": not (duplicates or missing or extra or malformed),
            "ordered": [key for key, _ in items] == expected_keys}


def score_board_state(expected: str, response: str) -> JsonDict:
    target = parse_state(expected)
    if not target["complete"] or not target["ordered"]:
        raise ValueError("invalid canonical full-board target")
    predicted = parse_state(response)
    groups: dict[str, dict[str, int]] = {}

    def add(group: str, correct: bool) -> None:
        cell = groups.setdefault(group, {"correct": 0, "total": 0})
        cell["total"] += 1
        cell["correct"] += int(correct)

    matched = 0
    occupied_correct = occupied_total = empty_correct = empty_total = 0
    for key, value in target["items"]:
        got = predicted["values"].get(key)
        correct = got == value
        matched += correct
        if key.startswith("<T"):
            parts = value.split()
            actual = got.split() if got is not None else []
            add("tile_resource", len(actual) == 2 and actual[0] == parts[0])
            add("tile_number", len(actual) == 2 and actual[1] == parts[1])
        elif key.startswith(("<N", "<E")):
            family = "node" if key.startswith("<N") else "edge"
            add(family, correct)
            add(family + ("_empty" if value == "empty" else "_occupied"), correct)
            if value == "empty":
                empty_total += 1
                empty_correct += correct
            else:
                occupied_total += 1
                occupied_correct += correct
                color, piece = value.rsplit(" ", 1)
                add(piece, correct)
                add("color:" + color, correct)
        else:
            add("robber" if key == "robber" else "port", correct)
    semantic_exact = predicted["complete"] and matched == 155
    return {
        "scoring": TASK, "correct": semantic_exact and predicted["ordered"],
        "semantic_exact": semantic_exact, "ordered": predicted["ordered"],
        "coverage_complete": predicted["complete"], "items_correct": int(matched),
        "items_total": 155, "items_extra": len(predicted["extra"]),
        "missing": json_list(predicted["missing"]),
        "duplicates": json_list(predicted["duplicates"]),
        "malformed": json_list(predicted["malformed"]),
        "groups": {key: json_dict(cell) for key, cell in groups.items()},
        "occupied_items_correct": int(occupied_correct), "occupied_items_total": occupied_total,
        "empty_items_correct": int(empty_correct), "empty_items_total": empty_total,
    }


def summarize_board_states(records: list[JsonDict]) -> JsonDict:
    totals: dict[str, Counter[str]] = {}
    density: dict[str, Counter[str]] = {}
    layouts: dict[str, Counter[str]] = {}
    scores = [as_dict(record["score"]) for record in records]
    for record, score in zip(records, scores, strict=True):
        for key, cell in as_dict(score["groups"]).items():
            totals.setdefault(key, Counter()).update(
                {name: as_int(count) for name, count in as_dict(cell).items()}
            )
        counts = {"boards": 1, "exact": int(as_bool(score["correct"])),
            "occupied_correct": as_int(score["occupied_items_correct"]),
            "occupied_total": as_int(score["occupied_items_total"])}
        metadata = as_dict(record.get("metadata", {}))
        bucket = str(metadata.get("density_bin", "unknown"))
        density.setdefault(bucket, Counter()).update(counts)
        layout = str(metadata.get("layout_id", "unknown"))
        layouts.setdefault(layout, Counter()).update(counts)
    return {
        "boards": len(records), "board_exact": sum(as_bool(s["correct"]) for s in scores),
        "semantic_exact": sum(as_bool(s["semantic_exact"]) for s in scores),
        "complete_coverage": sum(as_bool(s["coverage_complete"]) for s in scores),
        "duplicate_boards": sum(bool(s["duplicates"]) for s in scores),
        "groups": {key: {**json_dict(cell), "accuracy": cell["correct"] / cell["total"]}
                   for key, cell in totals.items()},
        "by_density": {key: json_dict(cell) for key, cell in density.items()},
        "by_layout": {key: json_dict(cell) for key, cell in layouts.items()},
        "occupied_layout_macro_accuracy": (
            sum(cell["occupied_correct"] / cell["occupied_total"] for cell in layouts.values() if cell["occupied_total"])
            / sum(bool(cell["occupied_total"]) for cell in layouts.values())
            if any(cell["occupied_total"] for cell in layouts.values()) else None
        ),
    }
