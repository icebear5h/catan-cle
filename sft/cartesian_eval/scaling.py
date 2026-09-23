"""Public factor-four evaluator API, bound to the historical 200 sparse-h cases."""

from __future__ import annotations

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
from .dataset import ROW_FIELDS
from .scaling_geometry import (
    SCHEMA,
    VARIANTS,
    VERSION,
    check_variant,
    mapping_sha256,
    parent_response,
    project_text,
    scaling_prompt,
)
from .shorthand import score_response as shorthand_score
from .shorthand import validate_metadata as validate_shorthand_metadata
from .shorthand import validate_rows as validate_shorthand_rows
from .shorthand_geometry import SCHEMA as SHORTHAND_SCHEMA
from .shorthand_geometry import sparse_prompt

__all__ = ["SCHEMA", "VARIANTS", "score_response", "validate_rows"]

PARENT_SHA256 = "93e115ffd90f902ff3962132141a09fd79163836964becae0f9912eea34092bd"


def _parent_prompt(parent: Mapping[str, object]) -> str:
    """Render from already validated references; the source artifact is not needed at scoring."""
    dense = object_map(object_map(parent["cartesian_reference"])["metadata"])
    atlas = object_map(object_map(dense["atlas_reference"])["metadata"])
    return sparse_prompt(render_prompt(atlas))[0]


def derived_metadata(
    parent_id: str, parent: Mapping[str, object], variant: str,
) -> dict[str, object]:
    check_variant(variant)
    reference = parse_json(compact(parent))
    validate_shorthand_metadata(reference)
    require(parent_id == object_map(reference["cartesian_reference"])["id"],
            "parent identity mismatch")
    prompt = scaling_prompt(_parent_prompt(reference), variant)
    return {
        "schema": SCHEMA, "version": VERSION, "variant": variant, "representation": variant,
        **{key: reference[key] for key in (*DIMENSIONS, "eval_position")},
        "source_sha256": reference["source_sha256"], "parent_sha256": PARENT_SHA256,
        "mapping_sha256": mapping_sha256(variant), "prompt_sha256": sha256(prompt.encode("utf-8")),
        "answer": project_text(text(reference["answer"]), variant),
        "component_counts": reference["component_counts"], "admitted_for_training": False,
        "shorthand_reference": {"id": parent_id, "metadata": reference},
    }


def validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Recompute all declarations and gold through the unchanged sparse-h oracle."""
    try:
        require(metadata.get("schema") == SCHEMA, "invalid scaling schema")
        reference = object_map(metadata.get("shorthand_reference"))
        require(reference.keys() == {"id", "metadata"}, "invalid shorthand reference fields")
        parent = object_map(reference.get("metadata"))
        expected = derived_metadata(text(reference.get("id")), parent, text(metadata.get("variant")))
        require(expected.keys() <= metadata.keys(), "missing scaling metadata fields")
        require(compact({key: metadata[key] for key in expected}) == compact(expected),
                "scaling metadata/gold projection mismatch")
        return parse_json(compact(parent))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed scaling metadata: {exc}") from exc


def score_response(
    expected: str, response: str, metadata: Mapping[str, object],
) -> dict[str, object] | None:
    if metadata.get("schema") != SCHEMA:
        return None
    parent = validate_metadata(metadata)
    variant = text(metadata["variant"])
    require(expected == metadata["answer"], "expected gold projection mismatch")
    normalized, restored, error = response, "", None
    try:
        normalized = _strip_transport(text(response))
        restored = parent_response(normalized, variant)
    except ValueError as exc:
        error = str(exc)
    score = object_map(shorthand_score(text(parent["answer"]), restored, parent))
    return {
        **score, "scoring": SCHEMA, "variant": variant,
        "expected": expected, "expected_normalized": expected,
        "response_raw": response, "raw_response_normalized": normalized,
        "response_normalized": (project_text(text(score["response_normalized"]), variant)
                                if score["format_valid"] is True else normalized),
        "format_error": error if error is not None else score["format_error"],
    }


def parent_row(metadata: Mapping[str, object]) -> dict[str, object]:
    parent = validate_metadata(metadata)
    row_id = object_map(metadata["shorthand_reference"])["id"]
    return {
        "schema": SHORTHAND_SCHEMA, "id": row_id, "row_id": row_id,
        **{key: parent[key] for key in DECLARATIONS}, "metadata": parent,
        "messages": [{"role": "user", "content": _parent_prompt(parent)},
                     {"role": "assistant", "content": parent["answer"]}],
    }


def validate_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Bind all 200 IDs/order/splits by reconstructing and validating the pinned parent."""
    require(len(rows) == 200, "panel requires exactly 200 rows")
    variants = {text(object_map(row.get("metadata")).get("variant")) for row in rows}
    require(len(variants) == 1, "mixed scaling variants within panel")
    variant = next(iter(variants))
    check_variant(variant)
    parents = [parent_row(object_map(row.get("metadata"))) for row in rows]
    parent_validation = validate_shorthand_rows(parents)
    require(sha256("".join(compact(row) + "\n" for row in parents).encode("utf-8")) == PARENT_SHA256,
            "reconstructed parent source hash mismatch")
    for row, parent in zip(rows, parents, strict=True):
        require(row.keys() == ROW_FIELDS, "invalid scaling row fields")
        metadata = object_map(row["metadata"])
        original = object_map(parent["metadata"])
        expected = derived_metadata(text(parent["id"]), original, variant)
        require(metadata.keys() == expected.keys(), "unexpected dataset metadata")
        require(row["schema"] == SCHEMA and row["id"] == row["row_id"] == parent["id"],
                "scaling case identity mismatch")
        require(all(row[key] == parent[key] for key in DECLARATIONS), "row declaration mismatch")
        prompt, gold = scaling_prompt(_parent_prompt(original), variant), text(expected["answer"])
        require(compact(row["messages"]) == compact([
            {"role": "user", "content": prompt}, {"role": "assistant", "content": gold},
        ]), "rendered prompt/gold mismatch")
        score = object_map(score_response(gold, gold, metadata))
        require(score["correct"] is True and score["format_valid"] is True, "gold roundtrip failed")
    return {
        **parent_validation, "schema": SCHEMA, "version": VERSION, "variant": variant,
        "mapping_sha256": mapping_sha256(variant), "parent_sha256": PARENT_SHA256,
        "by_representation": {variant: 200}, "parent_validation": parent_validation,
    }
