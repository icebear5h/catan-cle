"""Prompt assembly: atlas context, tile layout text, and sentinel hints."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Dict, List

from evals.catan_board_bench.scoring.categories import JsonDict
from evals.catan_board_bench.scoring.normalization import canonical_category, find_by_token
from evals.json_types import JsonValue, as_dict, as_dicts, as_int, as_list, as_str


def build_prompt(qa: Mapping[str, JsonValue], *, use_atlas: bool) -> str:
    lines = []
    if use_atlas and qa.get("contract"):
        lines.append("Fixed atlas context:")
        lines.append(tile_layout_text(as_dict(qa["contract"], "qa contract")))
        local = local_atlas_context(qa)
        if local:
            lines.append(local)
        lines.append("")

    sentinel = sentinel_hint(qa)
    if sentinel:
        lines.append(sentinel)

    lines.extend(
        [
            f"Question: {qa['question']}",
            "",
            "Return only the answer.",
        ]
    )
    return "\n".join(lines)


def _coord_at(row: JsonDict, index: int) -> int:
    return as_int(as_list(row["coord"], "tile coord")[index], "tile coord")


def _tokens(value: JsonValue, label: str) -> list[str]:
    return [as_str(item, label) for item in as_list(value, label)]


def _rows(contract: Mapping[str, JsonValue], key: str) -> list[JsonDict]:
    return as_dicts(contract[key], f"contract {key}")


def tile_layout_text(contract: Mapping[str, JsonValue]) -> str:
    rows: Dict[int, List[JsonDict]] = defaultdict(list)
    for tile in _rows(contract, "tiles"):
        z = _coord_at(tile, 2)
        rows[z].append(tile)
    parts = []
    for row_index, z in enumerate(sorted(rows)):
        tiles = sorted(rows[z], key=lambda t: _coord_at(t, 0))
        parts.append(
            f"row {row_index} left-to-right: "
            + " ".join(as_str(t["token"], "tile token") for t in tiles)
        )
    return "Tile rows top-to-bottom: " + "; ".join(parts) + "."


def local_atlas_context(qa: Mapping[str, JsonValue]) -> str:
    raw_contract = qa.get("contract")
    if not raw_contract:
        return ""
    contract = as_dict(raw_contract, "qa contract")
    target = as_dict(qa["target"], "qa target")
    category = canonical_category(as_str(qa["category"], "qa category"))

    if category in {"tile_resource_number", "tile_has_robber", "tile_occupied_nodes"}:
        tile = find_by_token(_rows(contract, "tiles"), target.get("tile_token"))
        if tile:
            return (
                f"Local atlas: {tile['token']} touches nodes "
                f"{' '.join(_tokens(tile['node_tokens'], 'node_tokens'))} and edges "
                f"{' '.join(_tokens(tile['edge_tokens'], 'edge_tokens'))}."
            )

    if category == "node_occupancy":
        node = find_by_token(_rows(contract, "nodes"), target.get("node_token"))
        if node:
            port_text = (
                f" ports {' '.join(_tokens(node['port_tokens'], 'port_tokens'))}"
                if node["port_tokens"]
                else " no ports"
            )
            return (
                f"Local atlas: {node['token']} is the intersection of tiles "
                f"{' '.join(_tokens(node['adjacent_tile_tokens'], 'adjacent_tile_tokens'))}; "
                "adjacent edges "
                f"{' '.join(_tokens(node['adjacent_edge_tokens'], 'adjacent_edge_tokens'))};"
                f"{port_text}."
            )

    if category == "edge_road_owner":
        edge = find_by_token(_rows(contract, "edges"), target.get("edge_token"))
        if edge:
            adjacent_tiles = []
            edge_nodes = set(as_list(edge["nodes"], "edge nodes"))
            for tile in _rows(contract, "tiles"):
                if edge_nodes.issubset(set(as_list(tile["nodes"], "tile nodes"))):
                    adjacent_tiles.append(as_str(tile["token"], "tile token"))
            return (
                f"Local atlas: {edge['token']} connects "
                f"{' and '.join(_tokens(edge['node_tokens'], 'node_tokens'))} "
                f"and borders tiles {' '.join(adjacent_tiles)}."
            )

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port = find_by_token(_rows(contract, "ports"), target.get("port_token"))
        if port:
            return (
                f"Local atlas: {port['token']} is at coord {port['coord']} "
                f"with direction {port['direction']} and touches nodes "
                f"{' '.join(_tokens(port['attached_node_tokens'], 'attached_node_tokens'))}."
            )

    if category == "node_adjacent_tiles":
        node = find_by_token(_rows(contract, "nodes"), target.get("node_token"))
        if node:
            return (
                f"Local atlas: {node['token']} touches land tiles "
                f"{' '.join(_tokens(node['adjacent_tile_tokens'], 'adjacent_tile_tokens'))}."
            )

    if category == "edge_connects_nodes":
        edge = find_by_token(_rows(contract, "edges"), target.get("edge_token"))
        if edge:
            return f"Local atlas: {edge['token']} is a board edge token."

    return ""


def sentinel_hint(qa: Mapping[str, JsonValue]) -> str:
    category = canonical_category(as_str(qa["category"], "qa category"))
    if category in {"longest_road_holder", "largest_army_holder"}:
        return "If no player holds the award, answer exactly NONE."
    if category in {"node_occupancy", "edge_road_owner"}:
        return "If the queried location is unoccupied, answer exactly EMPTY."
    if category == "color_road_locations":
        return "If the player has no roads, answer exactly NONE. Otherwise list only edge tokens."
    if category == "tile_resource_number":
        return "If the tile is desert, answer exactly <DESERT> NO_NUMBER."
    if category == "robber_resource_number":
        return "Answer as: <Txx> <RESOURCE> NUMBER. If the robber is on desert, answer <Txx> <DESERT> NO_NUMBER."
    if category == "tile_has_robber":
        return "Answer exactly YES or NO."
    if category in {"robber_adjacent_buildings", "tile_occupied_nodes"}:
        return "If no buildings touch the tile, answer exactly NONE. Otherwise list node, color, and building tokens."
    if category == "port_trade_type":
        return "If the port is a 3:1 port, answer GENERIC 3:1. Otherwise answer the exact resource token and 2:1."
    if category == "port_type_nodes":
        return "If the port is a 3:1 port, use GENERIC. Otherwise use the exact resource token."
    if category == "port_occupancy":
        return "If no building touches the port, answer exactly NONE. Otherwise list color, building, and node tokens."
    if category in {"nodes_connected", "edge_connects_nodes"}:
        return "Answer exactly YES or NO."
    if category == "robber_presence":
        return "Answer exactly YES or NO."
    return ""

