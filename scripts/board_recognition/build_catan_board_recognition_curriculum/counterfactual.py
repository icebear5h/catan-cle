"""Counterfactual target selection and declared-value application."""

from __future__ import annotations

from collections.abc import Sequence

from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.players.data import JsonValue
from evals.catan_board_bench.tokens import (
    building_token,
    color_token,
    edge_token,
    node_token,
    port_token,
    resource_token,
    tile_token,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.contracts import (
    find_edge,
    find_token,
    node_occupancy,
    port_type,
    refresh_player_summaries,
)
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    PORT_CLASSES,
    RESOURCE_CLASSES,
    BoardIndices,
    JsonDict,
    edge_ids_from,
    integer,
    objs,
    text,
)

__all__ = ["apply_counterfactual", "apply_declared_value"]


def apply_counterfactual(
    contract: JsonDict,
    *,
    entity_type: str,
    indices: BoardIndices,
    colors: Sequence[str],
    selector: int,
) -> tuple[JsonDict, str, str]:
    if entity_type == "tile":
        candidates = [
            tile
            for tile in objs(contract["tiles"], "contract tiles")
            if tile["resource"] is not None
        ]
        tile = candidates[selector % len(candidates)]
        before = str(tile["resource"])
        resources = [
            resource for resource in RESOURCE_CLASSES if resource not in {"DESERT", before}
        ]
        after = resources[selector % len(resources)]
        return (
            {
                "entity_type": "tile",
                "entity_id": tile_token(integer(tile["id"], "tile id")),
                "attribute": "resource",
            },
            before,
            after,
        )
    if entity_type == "node":
        nodes = objs(contract["nodes"], "contract nodes")
        node = nodes[selector % len(nodes)]
        before = node_occupancy(node)
        choices = [
            "EMPTY",
            *[f"{color}_{building}" for color in colors for building in (SETTLEMENT, CITY)],
        ]
        after = choices[(choices.index(before) + 1 + selector) % len(choices)]
        if after == before:
            after = choices[(choices.index(before) + 1) % len(choices)]
        return (
            {
                "entity_type": "node",
                "entity_id": node_token(integer(node["id"], "node id")),
                "attribute": "occupancy",
            },
            before,
            after,
        )
    if entity_type == "edge":
        edge_ids = edge_ids_from(indices)
        edge_id = edge_ids[selector % len(edge_ids)]
        edge = find_edge(contract, edge_id)
        before = text(edge["road_color"], "edge road_color") if edge["road_color"] else "EMPTY"
        choices = ["EMPTY", *colors]
        after = choices[(choices.index(before) + 1 + selector) % len(choices)]
        if after == before:
            after = choices[(choices.index(before) + 1) % len(choices)]
        return (
            {
                "entity_type": "edge",
                "entity_id": edge_token(edge_id),
                "attribute": "owner",
            },
            before,
            after,
        )
    if entity_type == "port":
        ports = objs(contract["ports"], "contract ports")
        port = ports[selector % len(ports)]
        before = port_type(port)
        after = PORT_CLASSES[(PORT_CLASSES.index(before) + 1 + selector) % len(PORT_CLASSES)]
        if after == before:
            after = PORT_CLASSES[(PORT_CLASSES.index(before) + 1) % len(PORT_CLASSES)]
        return (
            {
                "entity_type": "port",
                "entity_id": port_token(integer(port["id"], "port id")),
                "attribute": "port_type",
            },
            before,
            after,
        )
    raise ValueError(f"unknown entity type: {entity_type}")


def apply_declared_value(
    contract: JsonDict,
    *,
    target: JsonDict,
    value: JsonValue,
    colors: Sequence[str],
) -> None:
    entity_type = target["entity_type"]
    entity_id = text(target["entity_id"], "target entity_id")
    if entity_type == "tile":
        tile = find_token(objs(contract["tiles"], "contract tiles"), entity_id)
        tile["resource"] = value
        tile["resource_token"] = resource_token(text(value, "declared resource"))
        return
    if entity_type == "node":
        node = find_token(objs(contract["nodes"], "contract nodes"), entity_id)
        if value == "EMPTY":
            node["color"] = None
            node["color_token"] = None
            node["building"] = None
            node["building_token"] = None
        else:
            color, building = text(value, "declared occupancy").split("_", maxsplit=1)
            node["color"] = color
            node["color_token"] = color_token(color)
            node["building"] = building
            node["building_token"] = building_token(building)
        refresh_player_summaries(contract, colors=colors)
        return
    if entity_type == "edge":
        edge = find_token(objs(contract["edges"], "contract edges"), entity_id)
        owner = None if value == "EMPTY" else value
        edge["road_color"] = owner
        edge["road_color_token"] = (
            color_token(text(owner, "declared owner")) if owner else None
        )
        refresh_player_summaries(contract, colors=colors)
        return
    if entity_type == "port":
        port = find_token(objs(contract["ports"], "contract ports"), entity_id)
        if value == "THREE_TO_ONE":
            port.update(
                {"kind": "generic", "ratio": "3:1", "resource": None, "resource_token": None}
            )
        else:
            resource = text(value, "declared port type").removeprefix("TWO_TO_ONE_")
            port.update(
                {
                    "kind": "resource",
                    "ratio": "2:1",
                    "resource": resource,
                    "resource_token": resource_token(resource),
                }
            )
        return
    raise ValueError(f"unknown entity type: {entity_type}")
