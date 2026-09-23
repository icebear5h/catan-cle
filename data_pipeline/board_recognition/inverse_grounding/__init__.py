"""Inverse visual grounding and 2:1 forward/inverse ms-swift projection."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter as Counter
from collections import defaultdict as defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from jsonschema import Draft202012Validator as Draft202012Validator

from data_pipeline.board_recognition.inverse_grounding.contracts import _entities
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.tokens import atlas_tokens
from evals.catan_board_bench.tokens import (
    semantic_recognition_token_inventory as semantic_recognition_token_inventory,
)

from ..location_descriptions import (
    compact_node_signatures,
    inverse_location_descriptions,
)
from ..replay_dataset import ALL_SPLITS as ALL_SPLITS
from ..replay_dataset import PRIMARY_SPLITS as PRIMARY_SPLITS
from ..replay_dataset import read_jsonl as read_jsonl
from ..replay_dataset import validate_replay_v1_dataset as validate_replay_v1_dataset
from ..replay_ms_swift import DEFAULT_OUTPUT_NAME as _DEFAULT_FORWARD_PROJECTION
from ..replay_ms_swift import validate_ms_swift_row as _validate_forward_row
from ..replay_ms_swift import (
    validate_replay_v1_ms_swift_semantic as validate_replay_v1_ms_swift_semantic,
)
from ..sources import PROJECT_ROOT as PROJECT_ROOT
from ..sources import file_sha256 as file_sha256
from .descriptions import _edge_description as _edge_description
from .descriptions import _node_description as _node_description
from .export import (
    export_replay_v1_ms_swift_bidirectional as export_replay_v1_ms_swift_bidirectional,
)
from .rows import _forward_rows_by_state as _forward_rows_by_state
from .rows import _mixed_rows_for_state as _mixed_rows_for_state
from .rows import _validate_schema_rows as _validate_schema_rows
from .rows import validate_inverse_ms_swift_row as validate_inverse_ms_swift_row
from .validation import (
    validate_replay_v1_ms_swift_bidirectional as validate_replay_v1_ms_swift_bidirectional,
)

DEFAULT_FORWARD_PROJECTION = _DEFAULT_FORWARD_PROJECTION
validate_forward_row = _validate_forward_row
EXPORT_SCHEMA = "catan_board_recognition_ms_swift_bidirectional/v1"
AUDIT_SCHEMA = "catan_board_recognition_ms_swift_inverse_audit/v1"
CONTRACT_SCHEMA = "catan_board_recognition_inverse_grounding_contract/v1"
DEFAULT_OUTPUT_NAME = "ms_swift_bidirectional_v1"
INVERSE_ROWS_PER_STATE = 4
FORWARD_ROWS_PER_STATE = 8
MIXED_ROWS_PER_STATE = INVERSE_ROWS_PER_STATE + FORWARD_ROWS_PER_STATE
ENTITY_TYPES = ("tile", "node", "edge", "port")
ROW_KEYS = {"messages", "images"}
ATLAS_TOKENS = frozenset(atlas_tokens())
ATLAS_TOKEN_RE = re.compile(r"^<(?:T\d{2}|N\d{2}|E\d{2}_\d{2}|P\d{2})>$")


class InverseGroundingError(RuntimeError):
    """Raised when the inverse or mixed projection violates its contract."""


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _prepare_output_dir(path: Path, *, dataset_root: Path, overwrite: bool) -> None:
    protected = {
        dataset_root,
        dataset_root / "images",
        dataset_root / "contracts",
        dataset_root / "dense_labels",
        dataset_root / "qwen_sft",
        dataset_root / DEFAULT_FORWARD_PROJECTION,
    }
    if path in {candidate.resolve() for candidate in protected}:
        raise InverseGroundingError(f"refusing to replace protected source path: {path}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def inverse_grounding_contract() -> JsonDict:
    return {
        "schema": CONTRACT_SCHEMA,
        "task": "image plus board-local description to one canonical atlas token",
        "prompt_format": "<image>\\nWhere is <unambiguous board-local description>?",
        "answer_format": "exactly one of the 154 atlas tokens",
        "rows_per_state": INVERSE_ROWS_PER_STATE,
        "entity_types": list(ENTITY_TYPES),
        "resource_codes": {
            "WO": "wood",
            "B": "brick",
            "S": "sheep",
            "WH": "wheat",
            "O": "ore",
            "D": "desert",
        },
        "node_surface_forms": [
            "O10/WH5/S6",
            "ore 10/wheat 5/sheep 6",
            "O/WH/S when unique",
            "ore/wheat/sheep when unique",
        ],
        "ambiguity_policy": (
            "resource-only and resource-number descriptions are used only when unique "
            "within the current board; otherwise use a unique tile-anchored direction"
        ),
        "piece_policy": (
            "alternating complete node and edge atlas cycles prefer occupied targets "
            "and include visible color plus settlement, city, or road"
        ),
        "mixed_projection": {
            "forward_rows_per_state": FORWARD_ROWS_PER_STATE,
            "inverse_rows_per_state": INVERSE_ROWS_PER_STATE,
            "ratio": "2:1 forward:inverse",
            "order": "two forward rows followed by one inverse row, repeated four times",
        },
    }


def _words(value: str) -> str:
    return value.lower().replace("_", " ")


def _cyclic_eligible(rows: Sequence[JsonDict], eligible: set[str], index: int) -> JsonDict:
    if not eligible:
        raise InverseGroundingError("inverse description has no eligible targets")
    for offset in range(len(rows)):
        row = rows[(index + offset) % len(rows)]
        if row["token"] in eligible:
            return row
    raise InverseGroundingError("eligible inverse target is absent from canonical entity rows")


def _select_entity(
    rows: Sequence[JsonDict],
    eligible: set[str],
    *,
    state_index: int,
    occupied_field: str | None = None,
) -> JsonDict:
    prefer_occupied = (state_index // len(rows)) % 2 == 1
    if occupied_field is not None and prefer_occupied:
        occupied = [row for row in rows if row["token"] in eligible and row.get(occupied_field)]
        if occupied:
            return occupied[state_index % len(occupied)]
    return _cyclic_eligible(rows, eligible, state_index)


def inverse_queries_for_state(
    state: JsonDict,
    contract: JsonDict,
    *,
    state_index: int,
) -> list[JsonDict]:
    """Create one deterministic inverse query for each atlas entity type."""

    natural = inverse_location_descriptions(contract)
    signatures = compact_node_signatures(contract)
    compact_nodes = {
        token for token, row in signatures.items() if row["full_unique"]
    }
    node_eligible = compact_nodes | {token for token in natural if token.startswith("<N")}
    edge_endpoint_eligible = {
        as_str(edge["token"])
        for edge in map(as_dict, as_list(contract["edges"]))
        if all(
            signatures[as_str(token)]["full_unique"]
            for token in as_list(edge["node_tokens"])
        )
    }
    edge_eligible = edge_endpoint_eligible | {
        token for token in natural if token.startswith("<E")
    }
    eligible_by_entity = {
        "tile": {token for token in natural if token.startswith("<T")},
        "node": node_eligible,
        "edge": edge_eligible,
        "port": {token for token in natural if token.startswith("<P")},
    }

    selected = {
        "tile": _select_entity(
            _entities(contract, "tiles"), eligible_by_entity["tile"], state_index=state_index
        ),
        "node": _select_entity(
            _entities(contract, "nodes"),
            eligible_by_entity["node"],
            state_index=state_index,
            occupied_field="building",
        ),
        "edge": _select_entity(
            _entities(contract, "edges"),
            eligible_by_entity["edge"],
            state_index=state_index,
            occupied_field="road_color",
        ),
        "port": _select_entity(
            _entities(contract, "ports"), eligible_by_entity["port"], state_index=state_index
        ),
    }

    queries = []
    for entity_type in ENTITY_TYPES:
        target = selected[entity_type]
        token = target["token"]
        if entity_type == "node":
            description, style, qualifier = _node_description(
                target, signatures, natural, state_index=state_index
            )
        elif entity_type == "edge":
            description, style, qualifier = _edge_description(
                target, signatures, natural, state_index=state_index
            )
        else:
            description = natural[as_str(token)]
            style = f"{entity_type}_fact" if entity_type == "tile" else "port_directional_anchor"
            qualifier = None
        prompt = f"<image>\nWhere is {description}?"
        queries.append(
            {
                "query_id": f"{state['sample_id']}_inverse_{entity_type}",
                "entity_type": entity_type,
                "target_token": token,
                "description": description,
                "description_style": style,
                "visual_qualifier": qualifier,
                "prompt": prompt,
                "answer": token,
            }
        )
    return queries


def inverse_ms_swift_row(query: JsonDict, *, image_name: str) -> JsonDict:
    return {
        "messages": [
            {"role": "user", "content": query["prompt"]},
            {"role": "assistant", "content": query["answer"]},
        ],
        "images": [image_name],
    }


def inverse_audit_row(state: JsonDict, query: JsonDict, *, state_index: int) -> JsonDict:
    return {
        "schema": AUDIT_SCHEMA,
        "query_id": query["query_id"],
        "state_id": state["sample_id"],
        "state_index": state_index,
        "split": state["split"],
        "image_name": Path(as_str(state["image_path"])).name,
        "image_sha256": as_dict(state["sha256"])["image"],
        "contract_path": state["contract_path"],
        "contract_sha256": as_dict(state["sha256"])["contract"],
        "source_kind": as_dict(state["source"])["kind"],
        "game_id": as_dict(state["source"]).get("game_id"),
        "trajectory_id": as_dict(state["source"])["trajectory_id"],
        "density_bin": state["density_bin"],
        "building_count": state["building_count"],
        "road_count": state["road_count"],
        "piece_count": as_int(state["building_count"]) + as_int(state["road_count"]),
        "entity_type": query["entity_type"],
        "target_token": query["target_token"],
        "description": query["description"],
        "description_style": query["description_style"],
        "visual_qualifier": query["visual_qualifier"],
        "semantic_prompt": query["prompt"],
        "semantic_answer": query["answer"],
        "prompt_sha256": hashlib.sha256(as_str(query["prompt"]).encode()).hexdigest(),
        "answer_sha256": hashlib.sha256(as_str(query["answer"]).encode()).hexdigest(),
    }
