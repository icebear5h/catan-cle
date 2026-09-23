from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TypeAlias, TypedDict

from data_pipeline.board_recognition.sources import (
    canonical_sha256,
    file_sha256,
    repository_relative,
)
from sft.board.symbolic_board_tasks import TRAIN_TASKS, decode_state
from sft.board.symbolic_board_tasks._types import DecodedState
from sft.json_types import JsonDict, JsonLikeDict, as_dict, as_str, loads_json

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "artifacts/generated/sft/symbolic_board_v2/train.jsonl"
OUTPUT = ROOT / "artifacts/generated/sft/symbolic_board_fluency_review_v1"
SCHEMA = "catan_board_fluency_review/v1"
VERSION = "symbolic_board_fluency_review_v1"
MAX_SOURCE_ROWS = 400
SEED = 20260914
RESOURCES = ("brick", "ore", "sheep", "wheat", "wood")
FAMILIES = {
    "relations_joins": (
        "owned_buildings_touching_resource", "owned_incident_roads",
        "local_node_tiles", "port_access",
    ),
    "sets_coverage": (
        "coverage_union", "coverage_intersection", "coverage_difference", "coverage_missing",
    ),
    "aggregation_comparison": (
        "resource_pip_totals", "resource_pip_argmax", "node_pip_sum", "roll_production",
    ),
    "connectivity_structure": (
        "component_roads", "component_count", "shortest_distance", "reachable_nodes",
    ),
    "constraints_consequences": (
        "distance_rule_witnesses", "road_removal_connectivity",
        "settlement_upgrade_production", "robber_move_production",
    ),
}
OPERATION_FAMILY = {op: family for family, ops in FAMILIES.items() for op in ops}
# Every operation gets ten rows. These are semantic cells, not answer padding.
CELLS = {
    "owned_buildings_touching_resource": {"nonempty": 8, "empty": 2},
    "owned_incident_roads": {"nonempty": 8, "empty": 2},
    "local_node_tiles": {"resource_only": 8, "includes_desert": 2},
    "port_access": {"nonempty": 8, "empty": 2},
    "coverage_union": {"complementary": 10},
    "coverage_intersection": {"nonempty": 8, "empty": 2},
    "coverage_difference": {"nonempty": 8, "empty": 2},
    "coverage_missing": {"nonempty": 8, "empty": 2},
    "resource_pip_totals": {"totals": 10},
    "resource_pip_argmax": {"tied": 5, "unique": 5},
    "node_pip_sum": {"sum": 10},
    "roll_production": {"positive": 8, "zero": 2},
    "component_roads": {"blocked_boundary": 2, "ordinary": 8},
    "component_count": {"multiple": 5, "single": 5},
    "shortest_distance": {"reachable": 6, "blocked_unreachable": 2, "disconnected": 2},
    "reachable_nodes": {"enemy_start": 2, "ordinary_start": 8},
    "distance_rule_witnesses": {"nonempty": 8, "empty": 2},
    "road_removal_connectivity": {"disconnects": 5, "alternate_route": 5},
    "settlement_upgrade_production": {"gain": 8, "blocked_zero": 2},
    "robber_move_production": {"gain": 4, "loss": 4, "unchanged": 2},
}
SET_FORMAT = "Output only the sorted, space-separated set; NONE if empty. No explanation."
VECTOR_FORMAT = (
    'Output only compact JSON with all five lowercase resource keys '
    '"brick", "ore", "sheep", "wheat", "wood" and integer values.'
)
PIP_RULES = (
    "Pips are the number of two-dice outcomes: 2/12=1, 3/11=2, 4/10=3, "
    "5/9=4, 6/8=5; desert=0. Ignore the robber and all buildings; count each "
    "tile once. "
)
PRODUCTION_RULES = (
    "Count production from all of that player's existing buildings: a settlement "
    "receives 1 card and a city 2 per touching tile with the rolled number; "
    "the robber blocks its tile. Ignore bank supply. "
)
COVERAGE_RULES = (
    "A player's resource coverage is the set of resource types on tiles touching "
    "at least one of their settlements or cities. Exclude desert; ignore the "
    "robber and roads; include each resource type only once. "
)
COMPONENT_RULES = (
    "Partition the player's existing road edges into traversable components: "
    "two edges can join through a shared node unless an opponent building "
    "occupies that junction. Each edge belongs to exactly one component; "
    "blocked boundary nodes may occur in more than one component. "
)
CHANGE_FORMAT = (
    'Output only compact JSON with "before", "after", and "delta", each a '
    'five-resource integer object (brick, ore, sheep, wheat, wood). '
    'Delta means after minus before.'
)
OWN_FILES = frozenset({"review.jsonl", "metadata.json", "preview.json"})

# Every authored module whose bytes determine the emitted artifacts.
BUILD_DEPENDENCIES = (
    ROOT / "sft/scripts/builders/build_board_fluency_review/__init__.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_sources.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_facts.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_prompt.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_candidates.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_audit.py",
    ROOT / "sft/scripts/builders/build_board_fluency_review/_build.py",
    ROOT / "sft/board/symbolic_board_tasks/__init__.py",
    ROOT / "sft/board/symbolic_board_tasks/_constants.py",
    ROOT / "sft/board/symbolic_board_tasks/_geometry.py",
    ROOT / "sft/board/symbolic_board_tasks/_contracts.py",
    ROOT / "sft/board/symbolic_board_tasks/_solve.py",
    ROOT / "sft/board/symbolic_board_tasks/_scoring.py",
    ROOT / "sft/board/spatial_tasks/__init__.py",
    ROOT / "sft/board/spatial_tasks/_topology.py",
    ROOT / "sft/board/spatial_tasks/_contracts.py",
    ROOT / "sft/board/spatial_tasks/_scoring.py",
    ROOT / "evals/catan_board_bench/annotations/__init__.py",
    ROOT / "evals/catan_board_bench/annotations/__main__.py",
    ROOT / "evals/catan_board_bench/annotations/constants.py",
    ROOT / "evals/catan_board_bench/annotations/geometry.py",
    ROOT / "evals/catan_board_bench/annotations/payload.py",
    ROOT / "evals/catan_board_bench/annotations/render_state.py",
    ROOT / "evals/catan_board_bench/tokens/__init__.py",
    ROOT / "evals/catan_board_bench/tokens/atlas.py",
    ROOT / "evals/catan_board_bench/tokens/manifest.py",
    ROOT / "evals/catan_board_bench/tokens/vocabulary.py",
    ROOT / "data_pipeline/board_recognition/sources.py",
    ROOT / "data_pipeline/board_recognition/replay_dataset.py",
    ROOT / "data_pipeline/board_recognition/full_board_readout.py",
)

# Review query bags are flat keyword arguments: board tokens and colors are text,
# dice rolls are integers.
QueryDict: TypeAlias = dict[str, str | int]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def compact(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass
class Donor:
    line: int
    row: JsonDict
    state: JsonDict
    data: DecodedState
    provenance: JsonDict
    state_hash: str
    line_sha256: str


def load_prefix() -> tuple[list[Donor], JsonLikeDict]:
    """Read at most 400 physical lines, including static rows and blank lines."""
    manifest_path = SOURCE.with_name("manifest.json")
    manifest = as_dict(loads_json(manifest_path.read_text()))
    declaration = as_dict(as_dict(manifest["files"])["train"])
    require(Path(as_str(declaration["path"])).resolve() == SOURCE.resolve(),
            "manifest source path mismatch")
    require(declaration["rows"] == as_dict(manifest["counts"])["train"] == 3200,
            "unexpected declared training corpus size")
    digest, byte_count, physical_lines = hashlib.sha256(), 0, 0
    donors: list[Donor] = []
    skipped: Counter[str] = Counter()
    seen: set[object] = set()
    with SOURCE.open("rb") as handle:
        for line, raw in enumerate(islice(handle, MAX_SOURCE_ROWS), 1):
            physical_lines = line
            digest.update(raw)
            byte_count += len(raw)
            if not raw.strip():
                skipped["blank_physical_lines"] += 1
                continue
            row = as_dict(loads_json(raw))
            m = as_dict(row["metadata"])
            task = row["task_type"]
            require(row["schema"] == "catan_symbolic_board_row/v2", f"source schema: {line}")
            require(task in TRAIN_TASKS, f"non-training task: {line}")
            for key, value in (("split", "train"), ("task_role", "train"),
                               ("task_type", task), ("training_family", task)):
                require(row.get(key) == m.get(key) == value, f"source declaration {key}: {line}")
            require(row["id"] == row["row_id"] and row["row_id"] not in seen,
                    f"duplicate/mismatched donor ID: {line}")
            seen.add(row["row_id"])
            require(m["row_position"] == line - 1, f"source physical position: {line}")
            target_state = as_dict(m["target"])["state"]
            if target_state is None:
                skipped["static_rows_without_state"] += 1
                continue
            state = as_dict(target_state)
            p = as_dict(m["provenance"])
            source = as_dict(p["source"])
            require(p["split"] == "train", f"non-training donor provenance: {line}")
            require(source.get("split", "train") == "train",
                    f"non-training underlying source: {line}")
            require(bool(isinstance(source["kind"], str) and source["kind"]),
                    f"missing original source kind: {line}")
            for key in ("state_id", "density_bin", "board_map_sha256", "board_fact_sha256"):
                require(bool(isinstance(p[key], str) and p[key]),
                        f"missing provenance {key}: {line}")
            require(bool(as_dict(p["paths"])["contract"])
                    and len(as_str(as_dict(p["sha256"])["contract"])) == 64,
                    f"missing original contract/hash: {line}")
            donors.append(Donor(line, row, state, decode_state(state), p,
                                canonical_sha256(state), hashlib.sha256(raw).hexdigest()))
    require(physical_lines == MAX_SOURCE_ROWS, "source prefix shorter than required 400 lines")
    audit: JsonLikeDict = {
        "path": repository_relative(SOURCE),
        "max_physical_lines": MAX_SOURCE_ROWS,
        "physical_lines_read": physical_lines,
        "prefix_bytes": byte_count,
        "verified_prefix_sha256": digest.hexdigest(),
        "prefix_hash_scope": "Exact raw bytes of physical lines 1 through 400, including newlines",
        "manifest_path": repository_relative(manifest_path),
        "verified_manifest_file_sha256": file_sha256(manifest_path),
        "manifest_declared_full_rows": declaration["rows"],
        "manifest_declared_full_sha256": declaration["sha256"],
        "full_corpus_hash_verified": False,
        "skipped": dict(skipped),
        "state_bearing_donor_rows": len(donors),
        "distinct_donor_state_ids": len({d.provenance["state_id"] for d in donors}),
        "distinct_canonical_states": len({d.state_hash for d in donors}),
        "distinct_maps": len({d.provenance["board_map_sha256"] for d in donors}),
        "donor_rows_by_density": dict(Counter(as_str(d.provenance["density_bin"]) for d in donors)),
        "donor_rows_by_source_kind": dict(
            Counter(as_str(as_dict(d.provenance["source"])["kind"]) for d in donors)),
    }
    return donors, audit


class Message(TypedDict):
    """One chat turn in an emitted review row."""

    role: str
    content: str


class SourceRef(TypedDict):
    """Donor provenance carried on both the review row and its preview."""

    row_id: str
    line: int
    density: str
    map_id: str
    source_kind: str


class PreviewRow(TypedDict):
    """The flat shape the human review UI reads."""

    row_id: str
    state_id: str
    family: str
    operation: str
    question: str
    answer: str
    prompt: str
    source: SourceRef


class ReviewRow(TypedDict):
    """One emitted review.jsonl record."""

    schema: str
    id: str
    row_id: str
    messages: list[Message]
    metadata: JsonLikeDict
