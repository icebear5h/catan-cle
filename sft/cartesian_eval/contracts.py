"""Hash-bound metadata, complete-output parsing, and the existing canonical oracle."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from functools import lru_cache
from typing import cast

from sft.board.board_fluency_scoring import _strip_transport
from sft.board.coordinate_comparison import (
    score_coordinate_comparison,
    validate_comparison_metadata,
)
from sft.board.symbolic_board_tasks import symbolic_prompt
from sft.json_types import as_dict

from .geometry import (
    LEGEND,
    SCHEMA,
    canonical_response,
    coordinate_mapping,
    mapping_artifact,
    project_text,
)
from .reference import ATLAS_METADATA_SHA256, SOURCE_SHA256

VERSION = "cartesian_eval_v2"
DECLARATIONS = ("task_type", "training_family", "task_role", "split")
DIMENSIONS = ("family", "area", "operation", *DECLARATIONS, "source_id", "source_split")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected text")
    return value


def object_map(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected object mapping")
    mapping = cast(Mapping[object, object], value)
    return {text(key): item for key, item in mapping.items()}


def object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("expected list")
    return list(cast(list[object], value))


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(value: object) -> str:
    return sha256(compact(value).encode("utf-8"))


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_number(value: str) -> None:
    raise ValueError(f"noninteger JSON number: {value}")


def parse_json(raw: str) -> dict[str, object]:
    return object_map(json.loads(raw, object_pairs_hook=_unique_object,
                                 parse_float=_bad_number, parse_constant=_bad_number))


@lru_cache(maxsize=1)
def mapping_sha256() -> str:
    return digest(mapping_artifact())


def cartesian_metadata(atlas: Mapping[str, object]) -> dict[str, object]:
    """One copy of original query/state/receipts, inside the pinned atlas reference."""
    reference = parse_json(compact(dict(atlas)))
    position = reference.get("comparison_position")
    require(type(position) is int, "invalid atlas reference position")
    return {
        "schema": SCHEMA, "representation": "cartesian",
        **{key: reference[key] for key in DIMENSIONS},
        "eval_position": cast(int, position) // 2,
        "mapping_sha256": mapping_sha256(), "source_sha256": SOURCE_SHA256,
        "answer": project_text(text(reference["canonical_answer"])),
        "admitted_for_training": False,
        "atlas_reference": {"id": text(reference["pair_id"]) + "/atlas", "metadata": reference},
    }


def validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return detached, validated atlas metadata; evaluator-only annotations may be extra.

    A cached Cartesian gold is never authority. The per-position source pin binds
    the actual immutable artifact; the old validator independently recomputes gold.
    """
    try:
        require(metadata.get("schema") == SCHEMA, "invalid Cartesian schema")
        reference = object_map(metadata.get("atlas_reference"))
        require(reference.keys() == {"id", "metadata"}, "invalid atlas reference fields")
        atlas = parse_json(compact(object_map(reference.get("metadata"))))
        position = atlas.get("comparison_position")
        if type(position) is not int or not 0 <= position < 400 or position % 2:
            raise ValueError("invalid atlas reference position")
        require(digest(atlas) == ATLAS_METADATA_SHA256[position // 2],
                "atlas reference differs from immutable source case")
        validate_comparison_metadata(as_dict(atlas))
        expected = cartesian_metadata(atlas)
        require(expected.keys() <= metadata.keys(), "missing Cartesian metadata fields")
        require(compact({key: metadata[key] for key in expected}) == compact(expected),
                "Cartesian metadata/gold projection mismatch")
        return atlas
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"malformed Cartesian metadata: {exc}") from exc


def render_prompt(atlas: Mapping[str, object]) -> str:
    """Original symbolic query and complete state, projected without answer expansion."""
    target = object_map(atlas["target"])
    prompt = symbolic_prompt(text(atlas["task_type"]), as_dict(target))
    prefix = "Use the fixed learned Catan atlas. "
    require(prompt.startswith(prefix), "unexpected original symbolic prompt")
    prompt = prompt.removeprefix(prefix)
    header = LEGEND
    if target["state"] is None:
        header += "\nEntity inventory: " + " ".join(coordinate_mapping().values())
    return header + "\n" + project_text(prompt)


def score_response(
    expected: str, response: str, metadata: Mapping[str, object],
) -> dict[str, object] | None:
    """Exact schema dispatch. Invalid contracts raise; invalid predictions get no credit."""
    if metadata.get("schema") != SCHEMA:
        return None
    atlas = validate_metadata(metadata)
    require(expected == metadata["answer"], "expected gold projection mismatch")
    normalized, canonical, error = response, None, None
    valid, correct = False, False
    try:
        normalized = _strip_transport(text(response))
        parsed = canonical_response(normalized)
        score = object_map(score_coordinate_comparison(text(atlas["answer"]), parsed, as_dict(atlas)))
        valid, correct = score["format_valid"] is True, score["correct"] is True
        canonical = text(score["canonical_response_normalized"]) if valid else None
        error = score["format_error"]
    except ValueError as exc:
        error = str(exc)
    return {
        "scoring": SCHEMA, "symbolic_scoring": atlas["task_type"],
        "correct": correct, "format_valid": valid, "format_error": error,
        "expected": expected, "expected_normalized": metadata["answer"],
        "canonical_expected_normalized": atlas["canonical_answer"],
        "response_raw": response, "raw_response_normalized": normalized,
        "response_normalized": project_text(canonical) if canonical is not None else normalized,
        "canonical_response_normalized": canonical,
    }
