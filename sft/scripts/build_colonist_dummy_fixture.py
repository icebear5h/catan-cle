"""Build a dense Colonist-like board fixture for renderer calibration.

The fixture is intentionally hand-authored against the canonical atlas ids. It
is not meant to be a legal replay state; it is a visual stress case with real
tile ids, node ids, edge ids, resources, numbers, ports, buildings, roads, and
robber placement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


from catan_board_bench.tokens import (
    atlas_metadata,
    canonical_edge,
    color_token,
    edge_token,
    node_token,
    port_token,
    resource_token,
    tile_token,
)
from sft.paths import RENDER_CONTRACT_FIXTURE_DIR


DEFAULT_OUTPUT = RENDER_CONTRACT_FIXTURE_DIR / "colonist_dummy_setup.json"


TILE_LAYOUT = {
    15: ("WOOD", 9),
    16: ("WHEAT", 12),
    17: ("BRICK", 11),
    14: ("SHEEP", 10),
    5: (None, None),
    6: ("ORE", 6),
    18: ("SHEEP", 4),
    13: ("WOOD", 8),
    4: ("WOOD", 5),
    0: ("BRICK", 11),
    1: ("ORE", 3),
    7: ("BRICK", 8),
    12: ("WHEAT", 3),
    3: ("WHEAT", 4),
    2: ("SHEEP", 9),
    8: ("ORE", 10),
    11: ("WOOD", 6),
    10: ("WHEAT", 2),
    9: ("SHEEP", 5),
}


PORT_RESOURCES = {
    0: "BRICK",
    1: None,
    2: None,
    3: "WOOD",
    4: "SHEEP",
    5: None,
    6: "WHEAT",
    7: "ORE",
    8: "BRICK",
}


BUILDINGS = {
    1: ("GREEN", "CITY"),
    10: ("BLUE", "CITY"),
    14: ("BLUE", "CITY"),
    17: ("GREEN", "SETTLEMENT"),
    22: ("RED", "SETTLEMENT"),
    25: ("BLUE", "SETTLEMENT"),
    27: ("BLUE", "SETTLEMENT"),
    33: ("RED", "SETTLEMENT"),
    42: ("GREEN", "SETTLEMENT"),
}


ROADS = {
    (0, 1): "GREEN",
    (0, 5): "GREEN",
    (0, 20): "GREEN",
    (1, 6): "GREEN",
    (5, 16): "GREEN",
    (16, 18): "GREEN",
    (22, 49): "RED",
    (48, 49): "RED",
    (49, 50): "RED",
    (50, 51): "RED",
    (51, 52): "RED",
    (10, 11): "RED",
    (11, 32): "RED",
    (32, 33): "RED",
    (24, 25): "BLUE",
    (24, 53): "BLUE",
    (25, 26): "BLUE",
    (26, 27): "BLUE",
    (27, 28): "BLUE",
    (10, 29): "BLUE",
    (14, 15): "BLUE",
    (15, 17): "BLUE",
    (4, 15): "BLUE",
}


def build_fixture() -> dict[str, Any]:
    atlas = atlas_metadata()
    tile_refs = {tile["id"]: tile for tile in atlas["tiles"]}
    port_refs = {port["id"]: port for port in atlas["ports"]}
    valid_edges = {tuple(edge["id"]) for edge in atlas["edges"]}

    missing_tiles = sorted(set(TILE_LAYOUT) - set(tile_refs))
    if missing_tiles:
        raise RuntimeError(f"unknown tile ids: {missing_tiles}")

    road_edges = {canonical_edge(edge): color for edge, color in ROADS.items()}
    invalid_roads = sorted(edge for edge in road_edges if edge not in valid_edges)
    if invalid_roads:
        raise RuntimeError(f"invalid road edges: {invalid_roads}")

    tiles = []
    for tile_id in sorted(tile_refs):
        resource, number = TILE_LAYOUT[tile_id]
        tile_ref = tile_refs[tile_id]
        tile: dict[str, Any] = {
            "id": tile_id,
            "token": tile_token(tile_id),
            "coord": tile_ref["coord"],
            "nodes": list(sorted(tile_ref["nodes"].values())),
            "node_tokens": [node_token(node_id) for node_id in sorted(tile_ref["nodes"].values())],
            "edges": [
                list(edge)
                for edge in sorted(
                    canonical_edge(tuple(edge)) for edge in tile_ref["edges"].values()
                )
            ],
            "edge_tokens": [
                edge_token(tuple(edge))
                for edge in sorted(
                    canonical_edge(tuple(edge)) for edge in tile_ref["edges"].values()
                )
            ],
            "resource": resource,
            "resource_token": resource_token(resource),
            "number": number,
            "has_robber": tile_id == 1,
        }
        tiles.append(tile)

    ports = []
    for port_id in sorted(port_refs):
        resource = PORT_RESOURCES[port_id]
        port_ref = port_refs[port_id]
        ports.append(
            {
                "id": port_id,
                "token": port_token(port_id),
                "coord": port_ref["coord"],
                "direction": port_ref["direction"],
                "resource": resource,
                "resource_token": resource_token(resource) if resource else None,
                "attached_nodes": port_ref["attached_nodes"],
                "attached_node_tokens": [
                    node_token(node_id) for node_id in port_ref["attached_nodes"]
                ],
            }
        )

    nodes = []
    for node_ref in atlas["nodes"]:
        node_id = node_ref["id"]
        color, building = BUILDINGS.get(node_id, (None, None))
        nodes.append(
            {
                "id": node_id,
                "token": node_token(node_id),
                "building": building,
                "building_token": f"<{building}>" if building else None,
                "color": color,
                "color_token": color_token(color) if color else None,
            }
        )

    edges = []
    for edge_ref in atlas["edges"]:
        edge = tuple(edge_ref["id"])
        road_color = road_edges.get(edge)
        edges.append(
            {
                "id": list(edge),
                "token": edge_token(edge),
                "nodes": list(edge),
                "node_tokens": [node_token(node_id) for node_id in edge],
                "road_color": road_color,
                "road_color_token": color_token(road_color) if road_color else None,
            }
        )

    return {
        "schema": "catan_public_board_contract/v0",
        "source": {
            "kind": "hand_authored_renderer_fixture",
            "name": "colonist_dummy_setup",
            "note": "Approximate visual recreation of a dense Colonist board screenshot for renderer calibration.",
        },
        "sample": {
            "id": "colonist_dummy_setup",
            "split": "fixture",
            "category": "renderer_calibration",
        },
        "players": [
            {"color": color, "color_token": color_token(color)}
            for color in ["RED", "BLUE", "GREEN"]
        ],
        "current": {
            "current_color": "GREEN",
            "current_color_token": "<GREEN>",
            "current_prompt": "RENDERER_CALIBRATION_FIXTURE",
            "is_initial_build_phase": False,
            "num_completed_turns": 42,
            "player_index": 2,
            "turn_index": 42,
        },
        "robber": {
            "tile_id": 1,
            "tile_token": tile_token(1),
            "coord": tile_refs[1]["coord"],
        },
        "achievements": {
            "longest_road": {"holder": None, "holder_token": None, "length": 0},
            "largest_army": {"holder": None, "holder_token": None, "size": 0},
        },
        "ports": ports,
        "nodes": nodes,
        "edges": edges,
        "tiles": tiles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    fixture = build_fixture()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
