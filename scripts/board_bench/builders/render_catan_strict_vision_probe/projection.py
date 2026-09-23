"""Alias inversion and canonical-ID projection of the strict questions."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

from cle.players.data import JsonValue
from evals.catan_board_bench.paths import resolve_benchmark_reference
from scripts.board_bench.builders.render_catan_strict_vision_probe.constants import (
    ENTITY_ID_PATTERN,
    json_digest,
)
from scripts.board_bench.shapes import JsonDict, obj, objs, read_json, text, values

__all__ = [
    "canonical_id_map",
    "canonicalize_question",
    "normalize_projected_answer",
    "translate_json_ids",
    "translate_text_ids",
    "validate_canonical_questions",
]


def canonical_id_map(aliases: JsonDict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for canonical, alias in obj(aliases["tiles"], "aliases tiles").items():
        mapping[text(alias, "tile alias")] = f"T{int(canonical):02d}"
    for canonical, alias in obj(aliases["nodes"], "aliases nodes").items():
        mapping[text(alias, "node alias")] = f"N{int(canonical):02d}"
    for canonical, alias in obj(aliases["edges"], "aliases edges").items():
        node_a, node_b = (int(value) for value in canonical.split(","))
        mapping[text(alias, "edge alias")] = f"E{min(node_a, node_b):02d}_{max(node_a, node_b):02d}"
    for canonical, alias in obj(aliases["ports"], "aliases ports").items():
        mapping[text(alias, "port alias")] = f"P{int(canonical):02d}"
    expected_count = 19 + 54 + 72 + 9
    if len(mapping) != expected_count or len(set(mapping.values())) != expected_count:
        raise ValueError("canonical ID map is incomplete or non-bijective")
    return mapping


def canonicalize_question(
    question: JsonDict,
    mapping: dict[str, str],
    *,
    manifest_row: JsonDict,
) -> JsonDict:
    projected_answer = normalize_projected_answer(
        text(question["category"], "question category"),
        translate_json_ids(question["answer"], mapping),
    )
    contract_path = resolve_benchmark_reference(
        text(manifest_row["source_contract"], "source_contract")
    )
    projected: JsonDict = {
        **question,
        "question": translate_text_ids(text(question["question"], "question"), mapping),
        "answer": projected_answer,
        "target": translate_json_ids(question["target"], mapping),
        "source_answer": question["answer"],
        "source_answer_text": question["answer_text"],
        "source_fact_digest": question["fact_digest"],
        "engine_state_sha256": json_digest(read_json(contract_path)),
        "identity_projection": "canonical_engine_ids",
    }
    projected["answer_text"] = json.dumps(
        projected["answer"],
        separators=(",", ":"),
        sort_keys=True,
    )
    if projected["category"] == "road_inventory":
        projected["output_schema"] = text(
            projected["output_schema"], "output_schema"
        ).replace("Exx", "Exx_yy")
    projected.pop("fact_digest", None)
    return projected


def normalize_projected_answer(category: str, answer: JsonValue) -> JsonValue:
    if not isinstance(answer, dict):
        return answer
    normalized = dict(answer)
    if category == "node_adjacent_tiles":
        normalized["tiles"] = list(sorted(values(normalized["tiles"], "answer tiles"), key=str))
    elif category == "road_inventory":
        normalized["edges"] = list(sorted(values(normalized["edges"], "answer edges"), key=str))
    elif category == "port_occupancy":
        occupants = sorted(
            objs(normalized["occupants"], "answer occupants"),
            key=lambda occupant: str(occupant["node"]),
        )
        normalized["occupants"] = list(occupants)
    return normalized


def translate_text_ids(value: str, mapping: dict[str, str]) -> str:
    return ENTITY_ID_PATTERN.sub(
        lambda match: mapping.get(match.group(0), match.group(0)), value
    )


def translate_json_ids(value: JsonValue, mapping: dict[str, str]) -> JsonValue:
    if isinstance(value, str):
        return translate_text_ids(value, mapping)
    if isinstance(value, list):
        return [translate_json_ids(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: translate_json_ids(item, mapping) for key, item in value.items()}
    return value


def validate_canonical_questions(questions: Sequence[JsonDict]) -> None:
    if len(questions) != 60 or len({text(row["id"], "question id") for row in questions}) != 60:
        raise ValueError("canonical QA must contain 60 unique questions")
    categories = Counter(text(row["category"], "question category") for row in questions)
    if len(categories) != 10 or set(categories.values()) != {6}:
        raise ValueError("canonical QA category balance changed")
    for row in questions:
        if row.get("identity_projection") != "canonical_engine_ids":
            raise ValueError(f"question projection marker is missing: {row['id']}")
        if row["answer_text"] != json.dumps(row["answer"], separators=(",", ":"), sort_keys=True):
            raise ValueError(f"canonical answer text mismatch: {row['id']}")
