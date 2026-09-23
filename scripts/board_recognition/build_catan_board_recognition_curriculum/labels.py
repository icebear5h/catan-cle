"""Dense symbolic labels and their validation."""

from __future__ import annotations

from cle.players.data import JsonValue
from evals.catan_board_bench.tokens import edge_token, node_token, port_token, tile_token
from scripts.board_recognition.build_catan_board_recognition_curriculum.contracts import (
    node_occupancy,
    port_type,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    LABEL_SCHEMA,
    NUMBER_CLASSES,
    PORT_CLASSES,
    RESOURCE_CLASSES,
    JsonDict,
    edge_pair,
    integer,
    obj,
    objs,
    text,
)

__all__ = [
    "dense_label_differences",
    "dense_labels",
    "expected_entity_ids",
    "flatten_entities",
    "target_label_path",
    "validate_dense_labels",
    "value_at_label_path",
]


def dense_labels(contract: JsonDict, *, stage_id: str) -> JsonDict:
    tiles = objs(contract["tiles"], "contract tiles")
    nodes = objs(contract["nodes"], "contract nodes")
    edges = objs(contract["edges"], "contract edges")
    ports = objs(contract["ports"], "contract ports")
    return {
        "schema": LABEL_SCHEMA,
        "sample_id": obj(contract["sample"], "contract sample")["id"],
        "stage": stage_id,
        "entities": {
            "tiles": [
                {
                    "id": tile_token(integer(tile["id"], "tile id")),
                    "resource": "DESERT" if tile["resource"] is None else tile["resource"],
                    "number": tile["number"],
                    "robber": bool(tile["has_robber"]),
                }
                for tile in sorted(tiles, key=lambda row: integer(row["id"], "tile id"))
            ],
            "nodes": [
                {
                    "id": node_token(integer(node["id"], "node id")),
                    "occupancy": node_occupancy(node),
                }
                for node in sorted(nodes, key=lambda row: integer(row["id"], "node id"))
            ],
            "edges": [
                {
                    "id": edge_token(edge_pair(edge["id"], "edge id")),
                    "owner": edge["road_color"] or "EMPTY",
                }
                for edge in sorted(edges, key=lambda row: edge_pair(row["id"], "edge id"))
            ],
            "ports": [
                {
                    "id": port_token(integer(port["id"], "port id")),
                    "port_type": port_type(port),
                }
                for port in sorted(ports, key=lambda row: integer(row["id"], "port id"))
            ],
        },
    }


def validate_dense_labels(
    labels: JsonDict,
    *,
    expected_ids: dict[str, set[str]],
    node_classes: set[str],
    edge_classes: set[str],
) -> None:
    if labels.get("schema") != LABEL_SCHEMA:
        raise ValueError(f"dense label schema mismatch: {labels.get('sample_id')}")
    entities = obj(labels.get("entities", {}), "label entities")
    for collection, expected in expected_ids.items():
        rows = entities.get(collection)
        if not isinstance(rows, list) or {
            row.get("id") if isinstance(row, dict) else None for row in rows
        } != expected:
            raise ValueError(
                f"dense label {collection} coverage mismatch: {labels.get('sample_id')}"
            )
    tiles = objs(entities["tiles"], "label tiles")
    if any(
        tile["resource"] not in RESOURCE_CLASSES
        or tile["number"] not in NUMBER_CLASSES
        or not isinstance(tile["robber"], bool)
        for tile in tiles
    ):
        raise ValueError(f"invalid tile class: {labels.get('sample_id')}")
    if any((tile["resource"] == "DESERT") != (tile["number"] is None) for tile in tiles):
        raise ValueError(f"tile resource/number mismatch: {labels.get('sample_id')}")
    if sum(1 for tile in tiles if tile["robber"]) != 1:
        raise ValueError(f"dense labels require exactly one robber: {labels.get('sample_id')}")
    if any(
        node["occupancy"] not in node_classes
        for node in objs(entities["nodes"], "label nodes")
    ):
        raise ValueError(f"invalid node class: {labels.get('sample_id')}")
    if any(
        edge["owner"] not in edge_classes
        for edge in objs(entities["edges"], "label edges")
    ):
        raise ValueError(f"invalid edge class: {labels.get('sample_id')}")
    if any(
        port["port_type"] not in PORT_CLASSES
        for port in objs(entities["ports"], "label ports")
    ):
        raise ValueError(f"invalid port class: {labels.get('sample_id')}")


def expected_entity_ids(atlas: JsonDict) -> dict[str, set[str]]:
    return {
        "tiles": {
            tile_token(integer(tile["id"], "atlas tile id"))
            for tile in objs(atlas["tiles"], "atlas tiles")
        },
        "nodes": {
            node_token(integer(node["id"], "atlas node id"))
            for node in objs(atlas["nodes"], "atlas nodes")
        },
        "edges": {
            edge_token(edge_pair(edge["id"], "atlas edge id"))
            for edge in objs(atlas["edges"], "atlas edges")
        },
        "ports": {
            port_token(integer(port["id"], "atlas port id"))
            for port in objs(atlas["ports"], "atlas ports")
        },
    }


def target_label_path(target: JsonDict) -> str:
    collection = {"tile": "tiles", "node": "nodes", "edge": "edges", "port": "ports"}[
        text(target["entity_type"], "target entity_type")
    ]
    return f"{collection}/{target['entity_id']}/{target['attribute']}"


def dense_label_differences(first: JsonDict, second: JsonDict) -> list[str]:
    first_flat = flatten_entities(first)
    second_flat = flatten_entities(second)
    if set(first_flat) != set(second_flat):
        raise ValueError("dense label keys changed across counterfactual pair")
    return sorted(path for path in first_flat if first_flat[path] != second_flat[path])


def flatten_entities(entities: JsonDict) -> dict[str, JsonValue]:
    flattened: dict[str, JsonValue] = {}
    for collection in ("tiles", "nodes", "edges", "ports"):
        for row in objs(entities[collection], f"label {collection}"):
            for attribute, value in row.items():
                if attribute != "id":
                    flattened[f"{collection}/{row['id']}/{attribute}"] = value
    return flattened


def value_at_label_path(entities: JsonDict, path: str) -> JsonValue:
    collection, entity_id, attribute = path.split("/")
    for row in objs(entities[collection], f"label {collection}"):
        if row["id"] == entity_id:
            return row[attribute]
    raise KeyError(path)
