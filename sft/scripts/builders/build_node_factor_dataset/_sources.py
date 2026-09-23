from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

from cle.game_engine.models.enums import CITY, SETTLEMENT
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens import canonical_edge
from sft.json_types import JsonDict, as_dict, as_int, as_list

DEFAULT_COLORS = ("RED", "BLUE", "ORANGE", "WHITE", "BLACK")
RESOURCE_DECK = (
    ["WOOD"] * 4
    + ["BRICK"] * 3
    + ["SHEEP"] * 4
    + ["WHEAT"] * 4
    + ["ORE"] * 3
    + [None]
)
NUMBER_DECK = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
PORT_DECK = [None, None, None, None, "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
PROMPT_PREFIX = "Answer exactly using Catan atlas tokens. Use compact JSON when requested. Do not explain."
DATASET_NAME = "post_atlas_node_visual_grounding"
DATASET_SCHEMA = "catan_post_atlas_node_visual_grounding/v1"
CURRICULUM_STAGE = "phase_1_post_atlas_visual_grounding"
REQUIRES_STAGE = "phase_0_text_atlas_topology"
DATASET_ROLE = "bind_stable_node_tokens_to_transient_visual_facts"
CATEGORY_GROUPS = {
    "board_atlas_bboxes": "full_board_atlas_bbox_map",
    "node_bbox": "atlas_coordinate_grounding",
    "node_occupancy": "transient_node_state_readout",
    "node_adjacent_tile_resource_numbers": "local_tile_readout",
    "node_incident_road_owners": "local_edge_readout",
    "node_local_state_json": "local_state_composition",
}


def category_group(category: str) -> str:
    return CATEGORY_GROUPS.get(category, "uncategorized")


class AtlasIndices(TypedDict):
    tiles: dict[int, JsonDict]
    nodes: dict[int, JsonDict]
    edges: dict[tuple[int, int], JsonDict]
    ports: dict[int, JsonDict]
    node_tiles: dict[int, set[int]]
    node_edges: dict[int, set[tuple[int, int]]]
    edge_tiles: dict[tuple[int, int], set[int]]
    node_ports: dict[int, set[int]]
    node_neighbors: dict[int, set[int]]


def edge_pair(value: object) -> tuple[int, int]:
    """Narrow a JSON `[a, b]` edge id to a node-id pair."""
    a, b = as_list(value)
    return as_int(a), as_int(b)


def json_objects(value: object) -> list[JsonDict]:
    """Narrow a JSON array of objects."""
    return [as_dict(item) for item in as_list(value)]


def build_indices(atlas: JsonDict) -> AtlasIndices:
    tiles = {as_int(tile["id"]): tile for tile in json_objects(atlas["tiles"])}
    nodes = {as_int(node["id"]): node for node in json_objects(atlas["nodes"])}
    edges = {edge_pair(edge["id"]): edge for edge in json_objects(atlas["edges"])}
    ports = {as_int(port["id"]): port for port in json_objects(atlas["ports"])}

    node_tiles: dict[int, set[int]] = defaultdict(set)
    node_edges: dict[int, set[tuple[int, int]]] = defaultdict(set)
    edge_tiles: dict[tuple[int, int], set[int]] = defaultdict(set)
    node_ports: dict[int, set[int]] = defaultdict(set)
    node_neighbors: dict[int, set[int]] = defaultdict(set)

    for tile_id, tile in tiles.items():
        for node_id in as_dict(tile["nodes"]).values():
            node_tiles[as_int(node_id)].add(tile_id)
        for edge in as_dict(tile["edges"]).values():
            edge_tuple = canonical_edge(edge_pair(edge))
            edge_tiles[edge_tuple].add(tile_id)
            a, b = edge_tuple
            node_edges[a].add(edge_tuple)
            node_edges[b].add(edge_tuple)
            node_neighbors[a].add(b)
            node_neighbors[b].add(a)

    for port_id, port in ports.items():
        for node_id in as_list(port["attached_nodes"]):
            node_ports[as_int(node_id)].add(port_id)

    return {
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges,
        "ports": ports,
        "node_tiles": node_tiles,
        "node_edges": node_edges,
        "edge_tiles": edge_tiles,
        "node_ports": node_ports,
        "node_neighbors": node_neighbors,
    }


def validate_colors(colors: list[str]) -> list[str]:
    valid = {color.value for color in Color}
    unknown = sorted(set(colors) - valid)
    if unknown:
        raise ValueError(f"unknown colors: {unknown}; valid={sorted(valid)}")
    return colors


def occupancy_cases(colors: list[str]) -> list[dict[str, str | None]]:
    cases: list[dict[str, str | None]] = [{"name": "empty", "color": None, "building": None}]
    for color in colors:
        cases.append({"name": f"{color.lower()}_settlement", "color": color, "building": SETTLEMENT})
        cases.append({"name": f"{color.lower()}_city", "color": color, "building": CITY})
    return cases
