"""Atlas-to-coordinate mapping, projection, and JSON/hash primitives."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from data_pipeline.board_recognition.sources import canonical_sha256
from sft.board.coordinate_comparison._constants import (
    ATLAS_ATOM,
    COORDINATE_ATOM,
    GEOMETRY_CONVENTION,
    REPRESENTATIONS,
    SCHEMA,
)
from sft.board.symbolic_board_tasks import atlas_geometry
from sft.board.symbolic_board_tasks._types import Atlas
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict, as_int, as_list, as_str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(actual: object, expected: object, label: str) -> None:
    _require(_compact(actual) == _compact(expected), f"{label} mismatch")


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> JsonDict:
    result: JsonDict = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_jsonl(path: Path) -> list[JsonDict]:
    with path.open() as handle:
        return [json.loads(line, object_pairs_hook=_unique_object) for line in handle if line.strip()]


@lru_cache(maxsize=1)
def _geometry() -> Atlas:
    return atlas_geometry()


@lru_cache(maxsize=1)
def _mapping() -> dict[str, str]:
    atlas = _geometry()
    base = atlas["positions"]
    points: dict[str, tuple[int, ...]] = {
        token: (2 * x, 2 * y) for token, (x, y) in base.items()}
    for edge, (a, b) in atlas["edges"].items():
        points[edge] = tuple(base[a][axis] + base[b][axis] for axis in (0, 1))
    for entry in as_list(as_dict(atlas["raw"])["ports"]):
        port = as_dict(entry)
        a, b = (f"<N{as_int(nid):02d}>" for nid in as_list(port["attached_nodes"]))
        points[as_str(port["token"])] = tuple(
            base[a][axis] + base[b][axis] for axis in (0, 1))
    # The inventory matches the complete dynamic board's entity order.
    order = [t for family in "TNEP" for t in sorted(atlas["tokens"]) if t[1] == family]
    mapping = {t: f"{t[1]}({points[t][0]},{points[t][1]})" for t in order}
    _require(len(mapping) == len(set(mapping.values())) == 154, "coordinate mapping is not bijective")
    _require(all(COORDINATE_ATOM.fullmatch(v) for v in mapping.values()), "noninteger coordinate mapping")
    return mapping


def coordinate_mapping() -> dict[str, str]:
    """Detached 154-entry atlas-to-coordinate mapping in model inventory order."""
    return dict(_mapping())


@lru_cache(maxsize=1)
def _inverse_mapping() -> dict[str, str]:
    return {point: token for token, point in _mapping().items()}


def mapping_artifact() -> JsonLikeDict:
    """Crosswalk for offline inspection, never included in a model prompt."""
    return {
        "schema": SCHEMA, "kind": "mapping", "scale": 2,
        "convention": GEOMETRY_CONVENTION, "inventory_order": list(_mapping()),
        "atlas_to_coordinates": coordinate_mapping(),
        "coordinates_to_atlas": dict(_inverse_mapping()),
        "added_tokenizer_tokens": [],
    }


@lru_cache(maxsize=1)
def mapping_sha256() -> str:
    # Hash the payload; never include this digest in its own input.
    return canonical_sha256(mapping_artifact())


@lru_cache(maxsize=1)
def _static_fact_sha256() -> str:
    atlas = _geometry()
    facts: dict[str, object] = {
        "graph": {t: sorted(v) for t, v in atlas["graph"].items()},
        "touching": {t: sorted(v) for t, v in atlas["touching"].items()},
        "tile_neighbors": {t: sorted(v) for t, v in atlas["tile_neighbors"].items()},
    }
    facts.update(positions=atlas["positions"], edges=atlas["edges"])
    return canonical_sha256(facts)


def project_text(text: str, representation: str) -> str:
    """Replace entity atoms only, retaining record order and all other facts."""
    _require(representation in REPRESENTATIONS and isinstance(text, str), "invalid text projection")
    if representation == "atlas":
        return text

    def replace(match: re.Match[str]) -> str:
        _require(match[0] in _mapping(), f"unknown atlas atom: {match[0]}")
        return _mapping()[match[0]]

    return ATLAS_ATOM.sub(replace, text)
