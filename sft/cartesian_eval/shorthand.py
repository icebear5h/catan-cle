"""Public sparse-h evaluator API; gold remains bound to the dense Cartesian oracle."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from sft.board.board_fluency_scoring import _strip_transport

from .contracts import (
    DECLARATIONS,
    DIMENSIONS,
    compact,
    object_map,
    parse_json,
    render_prompt,
    require,
    sha256,
    text,
)
from .contracts import score_response as cartesian_score
from .contracts import validate_metadata as validate_cartesian_metadata
from .dataset import ROW_FIELDS
from .dataset import validate_rows as validate_cartesian_rows
from .geometry import SCHEMA as CARTESIAN_SCHEMA
from .shorthand_geometry import (
    SCHEMA,
    VERSION,
    mapping_sha256,
    project_text,
    root_response,
    sparse_prompt,
)

__all__ = ["SCHEMA", "VERSION", "score_response", "validate_rows"]

# The actual dense eval.jsonl used by the 143/200 stock run, not a new oracle panel.
PARENT_SHA256 = "ee2f3bb8824881dcf0740f8267b7cbfc577a11b57540e2755379345b51a86c64"


def derived_metadata(parent_id: str, parent: Mapping[str, object]) -> dict[str, object]:
    reference = parse_json(compact(parent))
    atlas = validate_cartesian_metadata(reference)
    require(parent_id == f"cartesian_eval_v2/{text(reference['source_id'])}", "parent identity mismatch")
    prompt, counts = sparse_prompt(render_prompt(atlas))
    return {
        "schema": SCHEMA, "version": VERSION, "representation": "cartesian_h",
        **{key: reference[key] for key in (*DIMENSIONS, "eval_position")},
        "source_sha256": reference["source_sha256"], "parent_sha256": PARENT_SHA256,
        "mapping_sha256": mapping_sha256(), "prompt_sha256": sha256(prompt.encode("utf-8")),
        "answer": project_text(text(reference["answer"])), "component_counts": counts,
        "admitted_for_training": False,
        "cartesian_reference": {"id": parent_id, "metadata": reference},
    }


def validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Independently validate the old oracle and rederive all outer cached declarations."""
    require(metadata.get("schema") == SCHEMA, "invalid shorthand schema")
    reference = object_map(metadata.get("cartesian_reference"))
    require(reference.keys() == {"id", "metadata"}, "invalid Cartesian reference fields")
    parent = object_map(reference.get("metadata"))
    expected = derived_metadata(text(reference.get("id")), parent)
    require(expected.keys() <= metadata.keys(), "missing shorthand metadata fields")
    require(compact({key: metadata[key] for key in expected}) == compact(expected),
            "shorthand metadata/gold projection mismatch")
    return parse_json(compact(parent))


def score_response(
    expected: str, response: str, metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Dispatch by schema, admit complete exact answers, then delegate actual scoring."""
    if metadata.get("schema") != SCHEMA:
        return None
    parent = validate_metadata(metadata)
    require(expected == project_text(text(parent["answer"])), "expected gold projection mismatch")
    normalized, root, error = response, "", None
    try:
        normalized = _strip_transport(text(response))
        root = root_response(normalized)
    except ValueError as exc:
        error = str(exc)
    score = object_map(cartesian_score(text(parent["answer"]), root, parent))
    return {
        **score, "scoring": SCHEMA, "expected": expected, "expected_normalized": expected,
        "response_raw": response, "raw_response_normalized": normalized,
        "response_normalized": (project_text(text(score["response_normalized"]))
                                if score["format_valid"] is True else normalized),
        "format_error": error if error is not None else score["format_error"],
    }


def parent_row(metadata: Mapping[str, object]) -> dict[str, object]:
    """Reconstruct the exact parent row without duplicating its prompt or atlas receipts."""
    parent = validate_metadata(metadata)
    atlas = validate_cartesian_metadata(parent)
    row_id = object_map(metadata["cartesian_reference"])["id"]
    return {
        "schema": CARTESIAN_SCHEMA, "id": row_id, "row_id": row_id,
        **{key: parent[key] for key in DECLARATIONS}, "metadata": parent,
        "messages": [{"role": "user", "content": render_prompt(atlas)},
                     {"role": "assistant", "content": parent["answer"]}],
    }


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Reuse the full old panel validator to bind all 200 cases and their original order."""
    require(len(rows) == 200, "panel requires exactly 200 rows")
    parents = [parent_row(object_map(row.get("metadata"))) for row in rows]
    parent_validation = validate_cartesian_rows(parents)
    require(sha256("".join(compact(row) + "\n" for row in parents).encode("utf-8")) == PARENT_SHA256,
            "reconstructed parent source hash mismatch")
    counts: Counter[str] = Counter()
    static = 0
    for row, parent in zip(rows, parents, strict=True):
        require(row.keys() == ROW_FIELDS, "invalid shorthand row fields")
        metadata = object_map(row["metadata"])
        expected = derived_metadata(text(parent["id"]), object_map(parent["metadata"]))
        require(metadata.keys() == expected.keys(), "unexpected dataset metadata")
        require(row["schema"] == SCHEMA and row["id"] == row["row_id"] == parent["id"],
                "shorthand case identity mismatch")
        require(all(row[key] == parent[key] for key in DECLARATIONS), "row declaration mismatch")
        atlas = validate_cartesian_metadata(object_map(parent["metadata"]))
        prompt, components = sparse_prompt(render_prompt(atlas))
        gold = text(expected["answer"])
        require(compact(row["messages"]) == compact([
            {"role": "user", "content": prompt}, {"role": "assistant", "content": gold},
        ]), "rendered prompt/gold mismatch")
        score = object_map(score_response(gold, gold, metadata))
        require(score["correct"] is True and score["format_valid"] is True, "gold roundtrip failed")
        counts.update(components)
        static += int(components["inventory_entities"] == 154)
    return {
        **parent_validation, "schema": SCHEMA, "version": VERSION,
        "mapping_sha256": mapping_sha256(), "parent_sha256": PARENT_SHA256,
        "by_representation": {"cartesian_h": 200}, "static_rows": static,
        "dynamic_rows": len(rows) - static, "component_counts": dict(counts),
        "parent_validation": parent_validation,
    }
