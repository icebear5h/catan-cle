"""Dense per-entity labels and their validation."""

from __future__ import annotations

from data_pipeline.board_recognition.replay_impl._config import (
    LABEL_SCHEMA,
    ReplayDatasetBuildError,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.tokens import (
    RECOGNITION_CLASS_VOCABULARIES,
    base_edges,
    edge_token,
    node_token,
    port_token,
    tile_token,
)


def normalize_class_name(attribute: str, value: object) -> str:
    if attribute == "number":
        return "NONE" if value is None else str(value)
    if attribute == "robber":
        return "PRESENT" if value else "ABSENT"
    return str(value)


def _rows(contract: JsonDict, key: str) -> list[JsonDict]:
    return [as_dict(row) for row in as_list(contract[key])]


def _edge_id(row: JsonDict) -> tuple[int, int]:
    left, right = (as_int(part) for part in as_list(row["id"]))
    return left, right


def dense_labels(contract: JsonDict, *, sample_id: str) -> JsonDict:
    labels: JsonDict = {
        "schema": LABEL_SCHEMA,
        "sample_id": sample_id,
        "entities": {
            "tiles": [
                {
                    "id": tile_token(as_int(tile["id"])),
                    "resource": "DESERT" if tile["resource"] is None else tile["resource"],
                    "number": tile["number"],
                    "robber": bool(tile["has_robber"]),
                }
                for tile in sorted(_rows(contract, "tiles"), key=lambda row: as_int(row["id"]))
            ],
            "nodes": [
                {
                    "id": node_token(as_int(node["id"])),
                    "occupancy": (
                        "EMPTY"
                        if node["color"] is None or node["building"] is None
                        else f"{node['color']}_{node['building']}"
                    ),
                }
                for node in sorted(_rows(contract, "nodes"), key=lambda row: as_int(row["id"]))
            ],
            "edges": [
                {
                    "id": edge_token(_edge_id(edge)),
                    "owner": edge["road_color"] or "EMPTY",
                }
                for edge in sorted(_rows(contract, "edges"), key=_edge_id)
            ],
            "ports": [
                {
                    "id": port_token(as_int(port["id"])),
                    "port_type": (
                        "THREE_TO_ONE"
                        if port["resource"] is None
                        else f"TWO_TO_ONE_{port['resource']}"
                    ),
                }
                for port in sorted(_rows(contract, "ports"), key=lambda row: as_int(row["id"]))
            ],
        },
    }
    validate_dense_labels(labels)
    return labels


def validate_dense_labels(labels: JsonDict) -> None:
    entities = as_dict(labels.get("entities", {}))
    expected_lengths = {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9}
    actual_lengths = {name: len(as_list(entities.get(name, []))) for name in expected_lengths}
    if actual_lengths != expected_lengths:
        raise ReplayDatasetBuildError(
            f"dense label topology mismatch: {actual_lengths} != {expected_lengths}"
        )
    expected_ids = {
        "tiles": {tile_token(index) for index in range(19)},
        "nodes": {node_token(index) for index in range(54)},
        "ports": {port_token(index) for index in range(9)},
        "edges": {edge_token(edge) for edge in base_edges()},
    }
    for collection, identifiers in expected_ids.items():
        if {as_str(row["id"]) for row in _rows(entities, collection)} != identifiers:
            raise ReplayDatasetBuildError(f"dense label {collection} IDs are incomplete")
    checks = (
        ("tiles", "resource", "tile.resource"),
        ("tiles", "number", "tile.number"),
        ("tiles", "robber", "tile.robber"),
        ("nodes", "occupancy", "node.occupancy"),
        ("edges", "owner", "edge.owner"),
        ("ports", "port_type", "port.port_type"),
    )
    for collection, attribute, head in checks:
        vocabulary = set(RECOGNITION_CLASS_VOCABULARIES[head])
        for row in _rows(entities, collection):
            class_name = normalize_class_name(attribute, row[attribute])
            if class_name not in vocabulary:
                raise ReplayDatasetBuildError(
                    f"unknown dense class {class_name!r} for {head} in {labels['sample_id']}"
                )
    tiles = _rows(entities, "tiles")
    if sum(1 for tile in tiles if tile["robber"]) != 1:
        raise ReplayDatasetBuildError("dense labels require exactly one robber")
    if any((tile["resource"] == "DESERT") != (tile["number"] is None) for tile in tiles):
        raise ReplayDatasetBuildError("dense tile resource/number labels disagree")
