"""Stage contract construction and dynamic piece placement."""

from __future__ import annotations

import random
from collections.abc import Sequence

from evals.catan_board_bench.tokens import building_token, color_token
from scripts.board_recognition.build_catan_board_recognition_curriculum.shapes import (
    BoardIndices,
    JsonDict,
    edge_ids_from,
    edge_pair,
    integer,
    obj,
    objs,
    text,
)
from sft.scripts.builders.build_node_factor_dataset import (
    build_contract,
    choose_distractor_buildings,
    player_summaries,
)

__all__ = [
    "find_edge",
    "find_token",
    "make_stage_contract",
    "node_occupancy",
    "port_type",
    "refresh_player_summaries",
    "set_dynamic_pieces",
    "set_sample_identity",
]


def make_stage_contract(
    *,
    atlas: JsonDict,
    indices: BoardIndices,
    stage: JsonDict,
    sample_id: str,
    sample_index: int,
    target_node_id: int,
    colors: Sequence[str],
    seed: int,
    image_size: int,
) -> JsonDict:
    contract: JsonDict = build_contract(
        atlas=atlas,
        indices=indices,
        sample_id=sample_id,
        sample_index=sample_index,
        target_node_id=target_node_id,
        occupancy={"name": "empty", "color": None, "building": None},
        colors=list(colors),
        seed=seed,
        image_size=image_size,
        distractor_buildings=0,
        distractor_roads=0,
        assume_rendered_images=True,
    )
    rng = random.Random(seed + 37)
    target_occupancy = text(stage["target_occupancy"], "stage target_occupancy")
    buildings: dict[int, tuple[str, str]] = {}
    if target_occupancy != "EMPTY":
        color, building = target_occupancy.split("_", maxsplit=1)
        buildings[target_node_id] = (color, building)
    buildings.update(
        choose_distractor_buildings(
            indices=indices,
            rng=rng,
            target_node_id=target_node_id,
            colors=list(colors),
            count=integer(stage["distractor_buildings"], "stage distractor_buildings"),
            existing=buildings,
        )
    )
    edge_ids = edge_ids_from(indices)
    rng.shuffle(edge_ids)
    road_count = integer(stage["roads"], "stage roads")
    roads = {
        edge: colors[(edge[0] + edge[1] + index + sample_index) % len(colors)]
        for index, edge in enumerate(edge_ids[:road_count])
    }
    set_dynamic_pieces(contract, buildings=buildings, roads=roads, colors=colors)
    return contract


def set_dynamic_pieces(
    contract: JsonDict,
    *,
    buildings: dict[int, tuple[str, str]],
    roads: dict[tuple[int, int], str],
    colors: Sequence[str],
) -> None:
    for node in objs(contract["nodes"], "contract nodes"):
        building = buildings.get(integer(node["id"], "node id"))
        node["color"] = building[0] if building else None
        node["color_token"] = color_token(building[0]) if building else None
        node["building"] = building[1] if building else None
        node["building_token"] = building_token(building[1]) if building else None
    for edge in objs(contract["edges"], "contract edges"):
        edge_id = edge_pair(edge["id"], "edge id")
        owner = roads.get(edge_id)
        edge["road_color"] = owner
        edge["road_color_token"] = color_token(owner) if owner else None
    contract["players"] = list(
        player_summaries(colors=list(colors), buildings=buildings, roads=roads)
    )


def refresh_player_summaries(contract: JsonDict, *, colors: Sequence[str]) -> None:
    buildings = {
        integer(node["id"], "node id"): (
            text(node["color"], "node color"),
            text(node["building"], "node building"),
        )
        for node in objs(contract["nodes"], "contract nodes")
        if node["color"] and node["building"]
    }
    roads = {
        edge_pair(edge["id"], "edge id"): text(edge["road_color"], "edge road_color")
        for edge in objs(contract["edges"], "contract edges")
        if edge["road_color"]
    }
    contract["players"] = list(
        player_summaries(colors=list(colors), buildings=buildings, roads=roads)
    )


def set_sample_identity(contract: JsonDict, sample_id: str) -> None:
    obj(contract["sample"], "contract sample")["id"] = sample_id


def node_occupancy(node: JsonDict) -> str:
    if not node.get("color") or not node.get("building"):
        return "EMPTY"
    return f"{node['color']}_{node['building']}"


def port_type(port: JsonDict) -> str:
    if port.get("resource") is None:
        return "THREE_TO_ONE"
    return f"TWO_TO_ONE_{port['resource']}"


def find_token(rows: Sequence[JsonDict], token: str) -> JsonDict:
    for row in rows:
        candidate = row.get("token")
        if candidate == token:
            return row
    raise KeyError(token)


def find_edge(contract: JsonDict, edge_id: tuple[int, int]) -> JsonDict:
    for edge in objs(contract["edges"], "contract edges"):
        if edge_pair(edge["id"], "edge id") == edge_id:
            return edge
    raise KeyError(edge_id)
