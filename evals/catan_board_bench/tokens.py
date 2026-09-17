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

from cle.game_engine.board_tokens import (
    _check_range as _check_range,
    canonical_edge as canonical_edge,
    edge_token as edge_token,
    node_token as node_token,
    port_token as port_token,
    tile_token as tile_token,
)
from cle.game_engine.models.enums import ActionType, RESOURCES, CITY, ROAD, SETTLEMENT
from cle.game_engine.models.map import (
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
from cle.game_engine.models.player import Color


NodeId = int
TileId = int
PortId = int
EdgeId = Tuple[int, int]
BOARD_OBJECTS = ["ROBBER"]
RECOGNITION_HEADS = (
    "tile.resource",
    "tile.number",
    "tile.robber",
    "node.occupancy",
    "edge.owner",
    "port.port_type",
)
RECOGNITION_CLASS_VOCABULARIES = {
    "tile.resource": ("DESERT", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"),
    "tile.number": ("NONE", "2", "3", "4", "5", "6", "8", "9", "10", "11", "12"),
    "tile.robber": ("ABSENT", "PRESENT"),
    "node.occupancy": (
        "EMPTY",
        *(f"{color.value}_{building}" for color in Color for building in (SETTLEMENT, CITY)),
    ),
    "edge.owner": ("EMPTY", *(color.value for color in Color)),
    "port.port_type": (
        "THREE_TO_ONE",
        "TWO_TO_ONE_WOOD",
        "TWO_TO_ONE_BRICK",
        "TWO_TO_ONE_SHEEP",
        "TWO_TO_ONE_WHEAT",
        "TWO_TO_ONE_ORE",
    ),
}


@dataclass(frozen=True)
class CatanTokenSpec:
    """One atomic token and the engine object it names."""

    token: str
    category: str
    value: Any


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


def recognition_query_token(head: str) -> str:
    if head not in RECOGNITION_CLASS_VOCABULARIES:
        raise ValueError(f"unknown board-recognition head: {head}")
    return f"<Q_{head.replace('.', '_').upper()}>"


def recognition_answer_token(head: str, class_name: str) -> str:
    vocabulary = RECOGNITION_CLASS_VOCABULARIES.get(head)
    if vocabulary is None:
        raise ValueError(f"unknown board-recognition head: {head}")
    if class_name not in vocabulary:
        raise ValueError(f"unknown class {class_name!r} for {head}")
    return f"<A_{head.replace('.', '_').upper()}_{class_name}>"


def recognition_query_tokens() -> List[str]:
    return [recognition_query_token(head) for head in RECOGNITION_HEADS]


def recognition_answer_tokens() -> List[str]:
    return [
        recognition_answer_token(head, class_name)
        for head in RECOGNITION_HEADS
        for class_name in RECOGNITION_CLASS_VOCABULARIES[head]
    ]


def building_token(building_type: str) -> str:
    return f"<{building_type}>"


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
    specs.extend(
        CatanTokenSpec(recognition_query_token(head), "recognition_query", head)
        for head in RECOGNITION_HEADS
    )
    specs.extend(
        CatanTokenSpec(
            recognition_answer_token(head, class_name),
            "recognition_answer",
            {"head": head, "class_name": class_name},
        )
        for head in RECOGNITION_HEADS
        for class_name in RECOGNITION_CLASS_VOCABULARIES[head]
    )

    tokens = [spec.token for spec in specs]
    if len(tokens) != len(set(tokens)):
        duplicates = sorted({token for token in tokens if tokens.count(token) > 1})
        raise RuntimeError(f"duplicate Catan tokens: {duplicates}")

    return specs


def added_tokens() -> List[str]:
    """Return just the regular added vocabulary tokens."""

    return [spec.token for spec in token_specs()]


def atlas_tokens() -> List[str]:
    """Return the 154 stable board-location tokens in canonical order."""

    categories = {"node", "edge", "tile", "port"}
    return [spec.token for spec in token_specs() if spec.category in categories]


def recognition_trainable_tokens() -> List[str]:
    """Return the historical atlas/query/answer rows used by Qwen SFT."""

    return [*atlas_tokens(), *recognition_query_tokens(), *recognition_answer_tokens()]


def recognition_token_inventory() -> Dict[str, Any]:
    tokens = recognition_trainable_tokens()
    return {
        "schema": "catan_board_recognition_token_inventory/v1",
        "atlas_tokens": atlas_tokens(),
        "query_tokens": recognition_query_tokens(),
        "answer_tokens": recognition_answer_tokens(),
        "tokens": tokens,
        "counts": {
            "atlas": 154,
            "query": len(recognition_query_tokens()),
            "answer": len(recognition_answer_tokens()),
            "total": len(tokens),
        },
    }


def semantic_recognition_token_inventory() -> Dict[str, Any]:
    """Return the semantic projection's bidirectional atlas-row inventory."""

    tokens = atlas_tokens()
    return {
        "schema": "catan_board_recognition_token_inventory/v3",
        "token_type": "regular_added_tokens",
        "trainable_side": "input_and_output_rows",
        "atlas_tokens": tokens,
        "tokens": tokens,
        "counts": {
            "node": NUM_NODES,
            "edge": NUM_EDGES,
            "tile": NUM_TILES,
            "port": 9,
            "atlas": len(tokens),
            "total": len(tokens),
        },
    }


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
