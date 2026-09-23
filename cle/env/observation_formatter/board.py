"""Building production, ports, and historical integer cube node labels."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.map import CatanMap

if TYPE_CHECKING:
    from . import CatanObservationFormatter, Observation


def format_board_state(self: CatanObservationFormatter, obs: Observation) -> str:
    """Format board state with strategic context."""
    lines = ["YOUR BUILDINGS:"]
    if not obs.my_settlements and not obs.my_cities:
        lines.append("  No buildings yet (initial placement phase)")
        return "\n".join(lines)
    if obs.my_settlements:
        lines.append(f"  Settlements ({len(obs.my_settlements)}):")
        for node_id in obs.my_settlements:
            context = self._get_node_strategic_context(node_id, obs)
            lines.append(f"    - {self._format_node(node_id, self._node_coords)}: {context}")
    if obs.my_cities:
        lines.append(f"  Cities ({len(obs.my_cities)}):")
        for node_id in obs.my_cities:
            context = self._get_node_strategic_context(node_id, obs)
            lines.append(f"    - {self._format_node(node_id, self._node_coords)}: {context}")
    if obs.my_roads:
        lines.append(f"  Roads ({len(obs.my_roads)}): {len(obs.my_roads)} connections")
        if obs.my_longest_road_length > 0:
            lines.append(f"    Longest road length: {obs.my_longest_road_length}")
    return "\n".join(lines)


def get_node_strategic_context(
    self: CatanObservationFormatter, node_id: int, obs: Observation,
) -> str:
    """Get strategic context for a node (ports, tiles, resources)."""
    parts = []
    board_map: CatanMap = obs.board_map
    if node_id in board_map.adjacent_tiles:
        tiles = board_map.adjacent_tiles[node_id]
        resources_with_numbers = []
        total_pips = 0
        for tile in tiles:
            if tile.resource is not None and tile.number is not None:
                resource_name = str(tile.resource)
                pips = self._number_to_pips(tile.number)
                total_pips += pips
                resources_with_numbers.append(f"{resource_name}(dice={tile.number},pips={pips})")
        if resources_with_numbers:
            parts.append(", ".join(resources_with_numbers))
            parts.append(f"{total_pips} pips")
    port_type = None
    for resource, nodes in board_map.port_nodes.items():
        if node_id in nodes:
            if resource is None:
                port_type = "3:1 port"
            else:
                port_type = f"{resource} 2:1 port"
            break
    if port_type:
        parts.append(port_type)
    return " | ".join(parts) if parts else "no production"


def number_to_pips(number: int) -> int:
    """Convert a production number to its two-dice probability dots."""
    pips_map = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
    return pips_map.get(number, 0)


def describe_node(self: CatanObservationFormatter, node_id: int, obs: Observation) -> str:
    """Describe a node with both dice numbers and production pips."""
    board_map: CatanMap = obs.board_map
    if node_id not in board_map.adjacent_tiles:
        return f"node {node_id}"
    tiles = board_map.adjacent_tiles[node_id]
    tile_descs = []
    total_pips = 0
    for tile in tiles:
        if tile.resource is not None and tile.number is not None:
            pips = self._number_to_pips(tile.number)
            total_pips += pips
            tile_descs.append(f"{tile.resource} dice={tile.number}({pips}pip)")
    if not tile_descs:
        return f"node {node_id}"
    port_str = ""
    for resource, nodes in board_map.port_nodes.items():
        if node_id in nodes:
            port_str = " (3:1 port)" if resource is None else f" ({resource} 2:1 port)"
            break
    return f"{'/'.join(tile_descs)} [{total_pips}pips]{port_str}"


def build_node_coordinate_map(board_map: CatanMap) -> dict[int, str]:
    """Sum adjacent tile cube coordinates into the historical node labels."""
    node_coords_accum: dict[int, list[Coordinate]] = defaultdict(list)
    for coord, tile in board_map.tiles.items():
        if hasattr(tile, 'nodes'):
            for node_id in tile.nodes.values():
                node_coords_accum[node_id].append(coord)
    result = {}
    for node_id, coords in node_coords_accum.items():
        x = sum(c[0] for c in coords)
        y = sum(c[1] for c in coords)
        z = sum(c[2] for c in coords)
        result[node_id] = f"({x}, {y}, {z})"
    return result


def format_node(node_id: int, node_coords: dict[int, str]) -> str:
    """Format a node_id with its coordinate."""
    coord = node_coords.get(node_id, "")
    if coord:
        return f"node {node_id} {coord}"
    return f"node {node_id}"


def format_edge(
    self: CatanObservationFormatter, edge: tuple[int, int], node_coords: dict[int, str],
) -> str:
    """Format an edge (node_id, node_id) with coordinates."""
    if isinstance(edge, tuple) and len(edge) == 2:
        n1 = self._format_node(edge[0], node_coords)
        n2 = self._format_node(edge[1], node_coords)
        return f"edge ({n1} -- {n2})"
    return f"edge {edge}"
