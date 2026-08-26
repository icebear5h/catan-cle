"""Catan-specific added vocabulary tokens.

These are intended to be added as regular tokenizer tokens, not chat/control
special tokens. They give the trainable model atomic symbols for the fixed
Catan atlas: tiles, nodes, edges, and ports.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from game_engine.models.enums import ActionType, RESOURCES, CITY, ROAD, SETTLEMENT
from game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    NUM_EDGES,
    NUM_NODES,
    NUM_TILES,
    CatanMap,
    LandTile,
    Port,
    PORT_DIRECTION_TO_NODEREFS,
    initialize_tiles,
)
from game_engine.models.player import Color


NodeId = int
TileId = int
PortId = int
EdgeId = Tuple[int, int]
BOARD_OBJECTS = ["ROBBER"]


@dataclass(frozen=True)
class CatanTokenSpec:
    """One atomic token and the engine object it names."""

    token: str
    category: str
    value: Any


def node_token(node_id: NodeId) -> str:
    _check_range("node_id", node_id, NUM_NODES)
    return f"<N{node_id:02d}>"


def tile_token(tile_id: TileId) -> str:
    _check_range("tile_id", tile_id, NUM_TILES)
    return f"<T{tile_id:02d}>"


def port_token(port_id: PortId) -> str:
    _check_range("port_id", port_id, 9)
    return f"<P{port_id:02d}>"


def edge_token(edge: EdgeId) -> str:
    a, b = canonical_edge(edge)
    return f"<E{a:02d}_{b:02d}>"


def resource_token(resource: str | None) -> str:
    return "<DESERT>" if resource is None else f"<{resource}>"


def object_token(object_name: str) -> str:
    return f"<{object_name}>"


def color_token(color: Color | str) -> str:
    value = color.value if isinstance(color, Color) else str(color)
    return f"<{value}>"


def action_token(action_type: ActionType | str) -> str:
    value = action_type.value if isinstance(action_type, ActionType) else str(action_type)
    return f"<{value}>"


def building_token(building_type: str) -> str:
    return f"<{building_type}>"


def canonical_edge(edge: EdgeId) -> EdgeId:
    a, b = edge
    if a == b:
        raise ValueError(f"edge endpoints must differ: {edge}")
    _check_range("edge node", a, NUM_NODES)
    _check_range("edge node", b, NUM_NODES)
    return (a, b) if a < b else (b, a)


def base_catan_map() -> CatanMap:
    """Build a deterministic base topology map.

    The atlas ids are topology-derived and independent of resource shuffles, but
    this avoids touching global random state while generating vocab metadata.
    """

    tiles = initialize_tiles(
        BASE_MAP_TEMPLATE,
        shuffled_numbers_param=list(reversed(BASE_MAP_TEMPLATE.numbers)),
        shuffled_port_resources_param=list(reversed(BASE_MAP_TEMPLATE.port_resources)),
        shuffled_tile_resources_param=list(reversed(BASE_MAP_TEMPLATE.tile_resources)),
    )
    return CatanMap.from_tiles(tiles)


def base_edges() -> List[EdgeId]:
    """Return the 72 canonical playable land edges in deterministic order."""

    catan_map = base_catan_map()
    edges = {
        canonical_edge(edge)
        for tile in catan_map.land_tiles.values()
        for edge in tile.edges.values()
    }
    if len(edges) != NUM_EDGES:
        raise RuntimeError(f"expected {NUM_EDGES} base edges, found {len(edges)}")
    return sorted(edges)


def atlas_metadata() -> Dict[str, Any]:
    """Return engine-derived atlas metadata for token consumers."""

    catan_map = base_catan_map()

    tiles = []
    for tile_id, tile in sorted(catan_map.tiles_by_id.items()):
        coordinate = _coordinate_for_tile(catan_map, tile)
        tiles.append(
            {
                "id": tile_id,
                "token": tile_token(tile_id),
                "coord": list(coordinate),
                "nodes": {node_ref.value: node_id for node_ref, node_id in tile.nodes.items()},
                "edges": {
                    edge_ref.value: list(canonical_edge(edge))
                    for edge_ref, edge in tile.edges.items()
                },
            }
        )

    ports = []
    for port_id, port in sorted(catan_map.ports_by_id.items()):
        coordinate = _coordinate_for_tile(catan_map, port)
        node_refs = PORT_DIRECTION_TO_NODEREFS[port.direction]
        attached_nodes = [port.nodes[node_ref] for node_ref in node_refs]
        ports.append(
            {
                "id": port_id,
                "token": port_token(port_id),
                "coord": list(coordinate),
                "direction": port.direction.value,
                "attached_nodes": attached_nodes,
            }
        )

    edges = [{"id": list(edge), "token": edge_token(edge)} for edge in base_edges()]
    nodes = [{"id": node_id, "token": node_token(node_id)} for node_id in range(NUM_NODES)]

    return {
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges,
        "ports": ports,
    }


def token_specs() -> List[CatanTokenSpec]:
    """Return all Catan added-token specs in stable order."""

    specs: List[CatanTokenSpec] = []

    specs.extend(
        CatanTokenSpec(node_token(node_id), "node", node_id) for node_id in range(NUM_NODES)
    )
    specs.extend(CatanTokenSpec(edge_token(edge), "edge", list(edge)) for edge in base_edges())
    specs.extend(
        CatanTokenSpec(tile_token(tile_id), "tile", tile_id) for tile_id in range(NUM_TILES)
    )
    specs.extend(CatanTokenSpec(port_token(port_id), "port", port_id) for port_id in range(9))
    specs.extend(
        CatanTokenSpec(resource_token(resource), "resource", resource)
        for resource in [*RESOURCES, None]
    )
    specs.extend(
        CatanTokenSpec(object_token(object_name), "object", object_name)
        for object_name in BOARD_OBJECTS
    )
    specs.extend(CatanTokenSpec(color_token(color), "color", color.value) for color in Color)
    specs.extend(
        CatanTokenSpec(building_token(building), "building", building)
        for building in [SETTLEMENT, CITY, ROAD]
    )
    specs.extend(
        CatanTokenSpec(action_token(action_type), "action", action_type.value)
        for action_type in ActionType
    )

    tokens = [spec.token for spec in specs]
    if len(tokens) != len(set(tokens)):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise RuntimeError(f"duplicate Catan tokens: {duplicates}")

    return specs


def added_tokens() -> List[str]:
    """Return just the regular added vocabulary tokens."""

    return [spec.token for spec in token_specs()]


def token_manifest() -> Dict[str, Any]:
    specs = token_specs()
    categories: Dict[str, List[str]] = {}
    for spec in specs:
        categories.setdefault(spec.category, []).append(spec.token)

    return {
        "token_type": "regular_added_tokens",
        "note": "Use tokenizer.add_tokens(...), not tokenizer.add_special_tokens(...).",
        "counts": {category: len(tokens) for category, tokens in categories.items()},
        "total": len(specs),
        "tokens": [spec.token for spec in specs],
        "specs": [asdict(spec) for spec in specs],
        "atlas": atlas_metadata(),
    }


def write_token_manifest(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(token_manifest(), indent=2) + "\n")
    return output_path


def add_tokens_to_tokenizer(tokenizer: Any) -> int:
    """Add Catan tokens to a Hugging Face tokenizer as regular tokens."""

    return tokenizer.add_tokens(added_tokens())


def _coordinate_for_tile(catan_map: CatanMap, target: LandTile | Port) -> Tuple[int, int, int]:
    for coordinate, tile in catan_map.tiles.items():
        if tile is target:
            return coordinate
    raise ValueError(f"tile not found: {target}")


def _check_range(name: str, value: int, size: int) -> None:
    if value < 0 or value >= size:
        raise ValueError(f"{name} must be in [0, {size - 1}], got {value}")
