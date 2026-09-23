"""Category-to-part mapping and atlas token patterns."""

from __future__ import annotations

import re

CANONICAL_CATEGORY_MAP: dict[str, str] = {
    "isolated_tile_resource_number": "tile",
    "local_patch_tile_resource_number": "tile",
    "isolated_road_owner": "edge",
    "local_patch_edge_road_owner": "edge",
    "isolated_node_occupancy": "node",
    "local_patch_node_occupancy": "node",
    "isolated_port_trade_type": "port",
    "local_patch_port_trade_type": "port",
    "isolated_robber_presence": "tile",
    "local_patch_robber_presence": "tile",
    "tile_resource_number": "tile",
    "tile_has_robber": "tile",
    "tile_occupied_nodes": "tile",
    "robber_tile": "tile",
    "robber_resource_number": "tile",
    "robber_adjacent_buildings": "tile",
    "node_occupancy": "node",
    "node_adjacent_tiles": "node",
    "edge_road_owner": "edge",
    "edge_connects_nodes": "edge",
    "port_trade_type": "port",
    "port_type_nodes": "port",
    "port_occupancy": "port",
}

PARTS = ("tile", "node", "edge", "port")

TOKEN_PATTERNS = {
    "tile": re.compile(r"<T(\d{1,2})>"),
    "node": re.compile(r"<N(\d{1,2})>"),
    "edge": re.compile(r"<E(\d{1,2})_(\d{1,2})>"),
    "port": re.compile(r"<P(\d{1,2})>"),
}

__all__ = ["CANONICAL_CATEGORY_MAP", "PARTS", "TOKEN_PATTERNS"]
