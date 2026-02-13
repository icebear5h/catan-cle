#!/usr/bin/env python3
"""
Colonist.io Standard Board Layout

Defines the standard 4-player Catan board layout as used by Colonist.io.
This provides visual positions for rendering replays directly.

The board uses pointy-top hexagons with:
- 19 land tiles (1 center + 6 inner ring + 12 outer ring)
- 54 nodes (corners where settlements go)
- 72 edges (where roads go)

Colonist numbers these in a specific order that we map to visual coordinates.
"""

import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Hex geometry constants for pointy-top hexagons
HEX_SIZE = 1.0  # Unit size, scale as needed for rendering


def hex_to_pixel(q: int, r: int, size: float = HEX_SIZE) -> Tuple[float, float]:
    """Convert axial hex coordinates to pixel position (pointy-top)."""
    x = size * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
    y = size * (3 / 2 * r)
    return (x, y)


# Standard Catan board tile positions in axial coordinates (q, r)
# Center tile is (0, 0), then rings expand outward
TILE_POSITIONS: Dict[int, Tuple[int, int]] = {
    # Center
    0: (0, 0),
    # Inner ring (clockwise from east)
    1: (1, 0),
    2: (0, 1),
    3: (-1, 1),
    4: (-1, 0),
    5: (0, -1),
    6: (1, -1),
    # Outer ring (clockwise from east-northeast)
    7: (2, -1),
    8: (2, 0),
    9: (1, 1),
    10: (0, 2),
    11: (-1, 2),
    12: (-2, 2),
    13: (-2, 1),
    14: (-2, 0),
    15: (-1, -1),
    16: (0, -2),
    17: (1, -2),
    18: (2, -2),
}


def get_tile_center(tile_id: int, hex_size: float = HEX_SIZE) -> Tuple[float, float]:
    """Get the center pixel position of a tile."""
    if tile_id not in TILE_POSITIONS:
        raise ValueError(f"Invalid tile ID: {tile_id}")
    q, r = TILE_POSITIONS[tile_id]
    return hex_to_pixel(q, r, hex_size)


# Node positions relative to hex center (pointy-top)
# Nodes are at the 6 corners of each hexagon
NODE_ANGLES = {
    'N': 90,    # Top
    'NE': 30,   # Top-right
    'SE': -30,  # Bottom-right
    'S': -90,   # Bottom
    'SW': -150, # Bottom-left
    'NW': 150,  # Top-left
}


def get_node_offset(direction: str, size: float = HEX_SIZE) -> Tuple[float, float]:
    """Get the offset from hex center to a node in the given direction."""
    angle_deg = NODE_ANGLES[direction]
    angle_rad = math.radians(angle_deg)
    x = size * math.cos(angle_rad)
    y = size * math.sin(angle_rad)
    return (x, y)


# Colonist node ID -> which tile and direction
# This defines the standard numbering used by Colonist.io
# Each node touches up to 3 tiles; we define it by one tile + direction

# Based on analysis of Colonist replays and standard Catan board topology
# Nodes are numbered in a specific pattern starting from the center hex corners
# and spiraling outward

# Tile's node directions (for each tile, which nodes it has at which corners)
@dataclass
class TileNodes:
    """Nodes at each corner of a tile."""
    n: int   # North node
    ne: int  # Northeast node
    se: int  # Southeast node
    s: int   # South node
    sw: int  # Southwest node
    nw: int  # Northwest node


# This mapping is derived from Colonist's board layout
# Tile ID -> TileNodes (node IDs at each corner)
TILE_TO_NODES: Dict[int, TileNodes] = {
    # Center tile (tile 0)
    0: TileNodes(n=0, ne=1, se=2, s=3, sw=4, nw=5),
    # Inner ring
    1: TileNodes(n=6, ne=7, se=8, s=2, sw=1, nw=0),
    2: TileNodes(n=1, ne=8, se=9, s=10, sw=3, nw=2),
    3: TileNodes(n=2, ne=9, se=11, s=12, sw=13, nw=3),
    4: TileNodes(n=5, ne=4, se=3, s=13, sw=14, nw=15),
    5: TileNodes(n=16, ne=0, se=5, s=15, sw=17, nw=18),
    6: TileNodes(n=19, ne=6, se=0, s=5, sw=16, nw=20),
    # Outer ring
    7: TileNodes(n=21, ne=22, se=23, s=7, sw=6, nw=19),
    8: TileNodes(n=22, ne=24, se=25, s=8, sw=7, nw=23),
    9: TileNodes(n=7, ne=25, se=26, s=9, sw=8, nw=2),  # Note: overlaps with center
    10: TileNodes(n=8, ne=26, se=27, s=28, sw=10, nw=9),
    11: TileNodes(n=9, ne=27, se=29, s=30, sw=11, nw=10),
    12: TileNodes(n=10, ne=28, se=30, s=31, sw=32, nw=12),
    13: TileNodes(n=3, ne=10, se=12, s=32, sw=33, nw=13),
    14: TileNodes(n=4, ne=3, se=13, s=33, sw=34, nw=14),
    15: TileNodes(n=15, ne=14, se=34, s=35, sw=36, nw=17),
    16: TileNodes(n=18, ne=17, se=36, s=37, sw=38, nw=39),
    17: TileNodes(n=20, ne=16, se=18, s=39, sw=40, nw=41),
    18: TileNodes(n=41, ne=19, se=21, s=42, sw=43, nw=40),
}


def compute_node_positions(hex_size: float = HEX_SIZE) -> Dict[int, Tuple[float, float]]:
    """
    Compute pixel positions for all 54 nodes.

    We compute each node position by averaging the positions from all tiles that share it.
    """
    node_positions: Dict[int, List[Tuple[float, float]]] = {}

    for tile_id, nodes in TILE_TO_NODES.items():
        tile_center = get_tile_center(tile_id, hex_size)

        for direction, node_id in [
            ('N', nodes.n), ('NE', nodes.ne), ('SE', nodes.se),
            ('S', nodes.s), ('SW', nodes.sw), ('NW', nodes.nw)
        ]:
            offset = get_node_offset(direction, hex_size)
            pos = (tile_center[0] + offset[0], tile_center[1] + offset[1])

            if node_id not in node_positions:
                node_positions[node_id] = []
            node_positions[node_id].append(pos)

    # Average positions (handles shared nodes)
    result = {}
    for node_id, positions in node_positions.items():
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        result[node_id] = (avg_x, avg_y)

    return result


# Edge definitions: which two nodes each edge connects
# Edge ID -> (node1, node2)
EDGE_TO_NODES: Dict[int, Tuple[int, int]] = {
    # Center tile edges
    0: (0, 1),
    1: (1, 2),
    2: (2, 3),
    3: (3, 4),
    4: (4, 5),
    5: (5, 0),
    # Inner ring edges (connecting to center and between inner tiles)
    6: (0, 6),
    7: (6, 7),
    8: (7, 8),
    9: (8, 2),
    10: (1, 8),
    11: (8, 9),
    12: (9, 10),
    13: (10, 3),
    14: (2, 9),
    15: (9, 11),
    16: (11, 12),
    17: (12, 13),
    18: (3, 13),
    19: (13, 14),
    20: (14, 15),
    21: (15, 5),
    22: (4, 14),
    23: (15, 17),
    24: (17, 18),
    25: (18, 16),
    26: (16, 0),
    27: (5, 16),
    28: (16, 20),
    29: (20, 19),
    30: (19, 6),
    31: (0, 19),
    # Outer ring edges
    32: (19, 21),
    33: (21, 22),
    34: (22, 23),
    35: (23, 7),
    36: (6, 22),
    37: (22, 24),
    38: (24, 25),
    39: (25, 8),
    40: (7, 25),
    41: (25, 26),
    42: (26, 9),
    43: (8, 26),
    44: (26, 27),
    45: (27, 28),
    46: (28, 10),
    47: (9, 27),
    48: (27, 29),
    49: (29, 30),
    50: (30, 11),
    51: (10, 29),
    52: (30, 31),
    53: (31, 32),
    54: (32, 12),
    55: (11, 31),
    56: (32, 33),
    57: (33, 13),
    58: (12, 33),
    59: (33, 34),
    60: (34, 14),
    61: (13, 34),
    62: (34, 35),
    63: (35, 36),
    64: (36, 17),
    65: (15, 35),
    66: (36, 37),
    67: (37, 38),
    68: (38, 39),
    69: (39, 18),
    70: (17, 37),
    71: (18, 40),
}


def get_edge_positions(
    node_positions: Dict[int, Tuple[float, float]]
) -> Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Get start and end positions for each edge."""
    result = {}
    for edge_id, (node1, node2) in EDGE_TO_NODES.items():
        if node1 in node_positions and node2 in node_positions:
            result[edge_id] = (node_positions[node1], node_positions[node2])
    return result


# Port definitions
# Port tiles are water tiles adjacent to land with trade capability
# Port ID -> {direction pointing to land, nodes that can use the port}
PORTS: Dict[int, Dict] = {
    0: {"resource": None, "nodes": [21, 22]},      # 3:1 port
    1: {"resource": "WHEAT", "nodes": [24, 25]},   # Wheat 2:1
    2: {"resource": None, "nodes": [27, 28]},      # 3:1 port
    3: {"resource": "SHEEP", "nodes": [30, 31]},   # Sheep 2:1
    4: {"resource": None, "nodes": [33, 34]},      # 3:1 port
    5: {"resource": "WOOD", "nodes": [35, 36]},    # Wood 2:1
    6: {"resource": None, "nodes": [38, 39]},      # 3:1 port
    7: {"resource": "BRICK", "nodes": [40, 41]},   # Brick 2:1
    8: {"resource": "ORE", "nodes": [42, 43]},     # Ore 2:1
}


def generate_frontend_layout() -> dict:
    """Generate layout data for the frontend."""
    hex_size = 50  # Pixels
    spacing = 3
    effective_size = hex_size + spacing

    node_positions = compute_node_positions(effective_size)
    edge_positions = get_edge_positions(node_positions)

    # Scale tile centers
    tile_centers = {}
    for tile_id in TILE_POSITIONS:
        tile_centers[tile_id] = get_tile_center(tile_id, effective_size)

    return {
        "hex_size": hex_size,
        "spacing": spacing,
        "tiles": tile_centers,
        "nodes": node_positions,
        "edges": {
            edge_id: {
                "nodes": list(EDGE_TO_NODES[edge_id]),
                "start": list(positions[0]),
                "end": list(positions[1])
            }
            for edge_id, positions in edge_positions.items()
        },
        "ports": PORTS,
    }


if __name__ == "__main__":
    import json

    layout = generate_frontend_layout()

    print("=== COLONIST BOARD LAYOUT ===")
    print(f"Tiles: {len(layout['tiles'])}")
    print(f"Nodes: {len(layout['nodes'])}")
    print(f"Edges: {len(layout['edges'])}")

    print("\n=== TILE CENTERS ===")
    for tile_id, pos in sorted(layout['tiles'].items()):
        print(f"Tile {tile_id}: ({pos[0]:.1f}, {pos[1]:.1f})")

    print("\n=== NODE POSITIONS (first 10) ===")
    for node_id, pos in sorted(layout['nodes'].items())[:10]:
        print(f"Node {node_id}: ({pos[0]:.1f}, {pos[1]:.1f})")

    # Save to JSON
    output_path = "data_pipeline/colonist_layout.json"
    with open(output_path, "w") as f:
        json.dump(layout, f, indent=2)
    print(f"\nLayout saved to {output_path}")
