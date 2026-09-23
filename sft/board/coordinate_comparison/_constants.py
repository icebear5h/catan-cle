"""Dataset constants, geometry convention, and schema field sets."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypedDict

from sft.json_types import JsonDict, JsonLikeDict

SCHEMA = "catan_coordinate_comparison/v1"
VERSION = "coordinate_comparison_v1"
# One level deeper than the former sft/board/coordinate_comparison.py module.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "artifacts/generated/sft/symbolic_board_v2"
REPRESENTATIONS = ("atlas", "coordinates")
PAIR_QUOTAS = {
    "symbolic_direction": 32, "symbolic_direction_choice": 32,
    "symbolic_neighbors": 32, "symbolic_incidence": 16,
    "symbolic_piece_owner": 32, "symbolic_owned_nodes": 24,
    "symbolic_owned_roads": 16, "symbolic_owned_incident_roads": 16,
}
OPERATION_AREA = {
    "symbolic_direction": "direction", "symbolic_direction_choice": "direction",
    "symbolic_neighbors": "adjacency", "symbolic_incidence": "incidence",
    "symbolic_piece_owner": "ownership", "symbolic_owned_nodes": "ownership",
    "symbolic_owned_roads": "ownership", "symbolic_owned_incident_roads": "ownership",
}
INCIDENCE_RELATIONS = ("N->T", "N->E", "N->P", "T->N", "T->E", "E->N", "E->T", "P->N")
COMPLETE_TEST_TASKS = (
    "symbolic_direction", "symbolic_direction_choice", "symbolic_neighbors", "symbolic_piece_owner",
)
ATLAS_ATOM = re.compile(r"<[NTEP][0-9_]+>")
INTEGER = r"(?:0|-?[1-9][0-9]*)"
COORDINATE_ATOM = re.compile(rf"[NTEP]\({INTEGER},{INTEGER}\)")
GEOMETRY_CONVENTION = (
    "Coordinates use a scaled integer layout: x increases right and y increases down. "
    "N(x,y) is a node; T(x,y) is a land tile center; E(x,y) is a road-edge midpoint; "
    "P(x,y) is the midpoint of a port's two attached nodes. "
    "A tile's six corner offsets are (0,-4),(2,-2),(2,2),(0,4),(-2,2),(-2,-2). "
    "Road edges join consecutive tile corners; their coordinates are the arithmetic "
    "midpoints of their endpoint nodes. Ports attach to the two endpoints of the "
    "coastal edge at their midpoint. Entity types distinguish coincident points. "
    "Use decimal integers without leading zeros, plus signs, or negative zero."
)
ATOM_INSTRUCTION = "Return entity identifiers as whitespace-free typed atoms exactly as shown."
COMPARISON_FIELDS = frozenset({
    "schema", "representation", "pair_id", "area", "operation", "family", "source",
    "source_id", "source_split", "mapping_sha256", "canonical_query_sha256",
    "canonical_fact_sha256", "canonical_answer", "answer", "comparison_position",
})
SOURCE_ROW_FIELDS = frozenset({
    "schema", "id", "row_id", "task_type", "training_family", "task_role", "split", "messages", "metadata",
})


class SourceCase(TypedDict):
    """One verified source row with its raw-line receipt."""

    row: JsonDict
    receipt: JsonDict


class PairArm(TypedDict):
    """One representation arm of a scored comparison pair."""

    id: str
    metadata: JsonDict
    score: JsonLikeDict
