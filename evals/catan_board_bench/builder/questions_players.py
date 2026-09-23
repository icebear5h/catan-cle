"""Player-count and board-topology question groups."""

from __future__ import annotations

from evals.catan_board_bench.builder.questions import AddQuestion
from evals.catan_board_bench.builder.selection import (
    _disconnected_node_pair,
    _pick,
    _road_edges_for_color,
    _tile_occupied_nodes_answer,
    _tile_occupied_nodes_target,
)
from evals.catan_board_bench.builder.shapes import json_list, rows, text, texts
from evals.catan_board_bench.tokens import node_token
from evals.json_types import JsonDict, JsonValue


def add_player_questions(contract: JsonDict, sample_index: int, add: AddQuestion) -> None:
    count_player = _pick(rows(contract, "players"), 1, sample_index + 5, stride=3)[0]
    add(
        "color_building_counts",
        f"How many settlements and cities does {count_player['color_token']} have on the board?",
        f"{count_player['color_token']} SETTLEMENTS {count_player['settlement_count']} CITIES {count_player['city_count']}",
        {
            "color": count_player["color"],
            "color_token": count_player["color_token"],
            "settlement_count": count_player["settlement_count"],
            "city_count": count_player["city_count"],
        },
    )
    add(
        "color_road_count",
        f"How many roads does {count_player['color_token']} have on the board?",
        f"{count_player['color_token']} ROADS {count_player['road_count']}",
        {
            "color": count_player["color"],
            "color_token": count_player["color_token"],
            "road_count": count_player["road_count"],
        },
    )
    road_edges = _road_edges_for_color(contract, text(count_player["color"], "player color"))
    road_edge_tokens = [text(edge["token"], "edge token") for edge in road_edges]
    add(
        "color_road_locations",
        f"Where are {count_player['color_token']}'s roads? List the edge tokens.",
        "NONE" if not road_edge_tokens else " ".join(road_edge_tokens),
        {
            "color": count_player["color"],
            "color_token": count_player["color_token"],
            "road_edges": [edge["id"] for edge in road_edges],
            "road_edge_tokens": json_list(road_edge_tokens),
        },
        scoring="set_exact",
    )

    occupied_tile = _pick(rows(contract, "tiles"), 1, sample_index + 6, stride=5)[0]
    add(
        "tile_occupied_nodes",
        f"Which buildings touch {occupied_tile['token']}?",
        _tile_occupied_nodes_answer(contract, occupied_tile),
        {
            "tile_id": occupied_tile["id"],
            "tile_token": occupied_tile["token"],
            "occupied_nodes": json_list(_tile_occupied_nodes_target(contract, occupied_tile)),
        },
    )


def add_topology_questions(contract: JsonDict, sample_index: int, add: AddQuestion) -> None:
    topology_node = _pick(rows(contract, "nodes"), 1, sample_index + 4, stride=13)[0]
    add(
        "node_adjacent_tiles",
        f"Which land tiles touch {topology_node['token']}?",
        " ".join(texts(topology_node["adjacent_tile_tokens"], "adjacent_tile_tokens")),
        {
            "node_id": topology_node["id"],
            "node_token": topology_node["token"],
            "adjacent_tiles": topology_node["adjacent_tiles"],
            "adjacent_tile_tokens": topology_node["adjacent_tile_tokens"],
        },
    )

    connected_edge = _pick(rows(contract, "edges"), 1, sample_index + 7, stride=17)[0]
    disconnected_pair = _disconnected_node_pair(contract, sample_index + 8)
    node_pair: JsonValue
    connecting_edge_token: JsonValue
    if sample_index % 2 == 0:
        node_pair = connected_edge["nodes"]
        node_pair_tokens = texts(connected_edge["node_tokens"], "edge node_tokens")
        connected = True
        connecting_edge_token = connected_edge["token"]
    else:
        node_pair = json_list(disconnected_pair)
        node_pair_tokens = [node_token(node_id) for node_id in disconnected_pair]
        connected = False
        connecting_edge_token = None
    add(
        "nodes_connected",
        f"Are {node_pair_tokens[0]} and {node_pair_tokens[1]} connected by a board edge?",
        "YES" if connected else "NO",
        {
            "nodes": node_pair,
            "node_tokens": json_list(node_pair_tokens),
            "connected": connected,
            "edge_token": connecting_edge_token,
        },
    )

    edge_probe = _pick(rows(contract, "edges"), 1, sample_index + 9, stride=19)[0]
    edge_node_pair: JsonValue
    if sample_index % 2 == 0:
        edge_node_pair = edge_probe["nodes"]
        edge_node_pair_tokens = texts(edge_probe["node_tokens"], "edge node_tokens")
        edge_connected = True
    else:
        probe_pair = _disconnected_node_pair(contract, sample_index + 10)
        edge_node_pair = json_list(probe_pair)
        edge_node_pair_tokens = [node_token(node_id) for node_id in probe_pair]
        edge_connected = False
    add(
        "edge_connects_nodes",
        f"Does {edge_probe['token']} connect {edge_node_pair_tokens[0]} and {edge_node_pair_tokens[1]}?",
        "YES" if edge_connected else "NO",
        {
            "edge": edge_probe["id"],
            "edge_token": edge_probe["token"],
            "nodes": edge_node_pair,
            "node_tokens": json_list(edge_node_pair_tokens),
            "connected": edge_connected,
        },
    )
