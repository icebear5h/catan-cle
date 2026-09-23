"""Contract inventory and QA-to-part targeting."""

from __future__ import annotations

from cle.players.data import JsonValue
from scripts.probes.probe_catan_board_coverage.model import (
    JsonDict,
    canonical_category,
    json_rows,
    normalize_token,
)

__all__ = ["build_expected_inventory", "question_to_targets", "robber_section"]


def robber_section(contract: JsonDict) -> JsonDict:
    robber = contract.get("robber", {}) or {}
    return robber if isinstance(robber, dict) else {}


def _target_of(qa: JsonDict) -> JsonDict:
    target = qa.get("target", {})
    return target if isinstance(target, dict) else {}


def question_to_targets(qa: JsonDict) -> list[tuple[str, str, str]]:
    """Return part probes targeted by one QA row.

    Returns tuples of (part_type, part_token, probe_kind).
    """

    category = canonical_category(str(qa.get("category", "")))
    target = _target_of(qa)

    out: list[tuple[str, str, str]] = []

    if category in {
        "tile_resource_number",
        "tile_has_robber",
        "tile_occupied_nodes",
    }:
        tile_token = normalize_token(target.get("tile_token"))
        if tile_token:
            out.append(("tile", tile_token, category))

    if category in {"robber_tile", "robber_presence", "robber_resource_number", "robber_adjacent_buildings"}:
        tile_token = normalize_token(target.get("tile_token"))
        if tile_token:
            out.append(("robber", tile_token, category))

    if category == "node_occupancy":
        node_token = normalize_token(target.get("node_token"))
        if node_token:
            out.append(("node", node_token, category))

    if category == "node_adjacent_tiles":
        node_token = normalize_token(target.get("node_token"))
        if node_token:
            out.append(("node", node_token, category))

    if category in {"edge_road_owner", "edge_connects_nodes"}:
        edge_token = normalize_token(target.get("edge_token"))
        if edge_token:
            out.append(("edge", edge_token, category))

    if category in {"port_trade_type", "port_type_nodes", "port_occupancy"}:
        port_token = normalize_token(target.get("port_token"))
        if port_token:
            out.append(("port", port_token, category))

    if category == "nodes_connected":
        node_tokens = target.get("node_tokens", []) or []
        if isinstance(node_tokens, list):
            for token in node_tokens:
                norm = normalize_token(token)
                if norm:
                    out.append(("node", norm, "nodes_connected"))

    return out


def _listed(payload: JsonDict, key: str) -> JsonValue:
    return payload.get(key, [])


def build_expected_inventory(contract: JsonDict) -> dict[str, dict[str, JsonDict]]:
    robber_tile_id = robber_section(contract).get("tile_id")

    tiles: dict[str, JsonDict] = {}
    for tile in json_rows(contract, "tiles"):
        token = normalize_token(tile.get("token"))
        if not token:
            continue
        tiles[token] = {
            "id": tile.get("id"),
            "token": token,
            "resource": tile.get("resource"),
            "number": tile.get("number"),
            "has_robber": tile.get("id") == robber_tile_id,
            "node_tokens": _listed(tile, "node_tokens"),
            "edge_tokens": _listed(tile, "edge_tokens"),
        }

    nodes: dict[str, JsonDict] = {}
    for node in json_rows(contract, "nodes"):
        token = normalize_token(node.get("token"))
        if not token:
            continue
        nodes[token] = {
            "id": node.get("id"),
            "token": token,
            "building": node.get("building"),
            "color": node.get("color"),
            "adjacent_tiles": _listed(node, "adjacent_tiles"),
            "adjacent_edges": _listed(node, "adjacent_edges"),
            "port_tokens": _listed(node, "port_tokens"),
        }

    edges: dict[str, JsonDict] = {}
    for edge in json_rows(contract, "edges"):
        token = normalize_token(edge.get("token"))
        if not token:
            continue
        edges[token] = {
            "id": edge.get("id"),
            "token": token,
            "nodes": _listed(edge, "nodes"),
            "road_color": edge.get("road_color"),
            "node_tokens": _listed(edge, "node_tokens"),
        }

    ports: dict[str, JsonDict] = {}
    for port in json_rows(contract, "ports"):
        token = normalize_token(port.get("token"))
        if not token:
            continue
        ports[token] = {
            "id": port.get("id"),
            "token": token,
            "resource": port.get("resource"),
            "ratio": port.get("ratio"),
            "nodes": _listed(port, "attached_nodes"),
            "node_tokens": _listed(port, "attached_node_tokens"),
            "coord": port.get("coord"),
        }

    robber_token = normalize_token(robber_section(contract).get("tile_token"))

    return {
        "tile": tiles,
        "node": nodes,
        "edge": edges,
        "port": ports,
        "robber": {robber_token: {"token": robber_token}} if robber_token else {},
    }
