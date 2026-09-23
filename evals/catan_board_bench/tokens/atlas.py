"""The fixed base topology and the derived tile/node/edge/port atlas."""

from __future__ import annotations

from typing import List, Tuple, TypedDict

from cle.game_engine.board_tokens import (
    canonical_edge,
    edge_token,
    node_token,
    port_token,
    tile_token,
)
from cle.game_engine.models.map import (
    BASE_MAP_TEMPLATE,
    NUM_EDGES,
    NUM_NODES,
    PORT_DIRECTION_TO_NODEREFS,
    CatanMap,
    LandTile,
    Port,
    initialize_tiles,
)
from evals.catan_board_bench.tokens.vocabulary import EdgeId
from evals.json_types import JsonDict


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


class AtlasTile(TypedDict):
    id: int
    token: str
    coord: List[int]
    nodes: dict[str, int]
    edges: dict[str, List[int]]


class AtlasPort(TypedDict):
    id: int
    token: str
    coord: List[int]
    direction: str
    attached_nodes: List[int]


class AtlasEdge(TypedDict):
    id: List[int]
    token: str


class AtlasNode(TypedDict):
    id: int
    token: str


class AtlasMetadata(TypedDict):
    tiles: List[AtlasTile]
    nodes: List[AtlasNode]
    edges: List[AtlasEdge]
    ports: List[AtlasPort]


def atlas_metadata() -> AtlasMetadata:
    """Return engine-derived atlas metadata for token consumers."""

    catan_map = base_catan_map()

    tiles: List[AtlasTile] = []
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

    ports: List[AtlasPort] = []
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

    edges: List[AtlasEdge] = [{"id": list(edge), "token": edge_token(edge)} for edge in base_edges()]
    nodes: List[AtlasNode] = [{"id": node_id, "token": node_token(node_id)} for node_id in range(NUM_NODES)]

    return {
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges,
        "ports": ports,
    }



def atlas_metadata_json() -> JsonDict:
    """Return ``atlas_metadata()`` as a plain JSON object with the same key order."""

    atlas = atlas_metadata()
    return {
        "tiles": [
            {
                "id": tile["id"],
                "token": tile["token"],
                "coord": list(tile["coord"]),
                "nodes": dict(tile["nodes"]),
                "edges": {ref: list(edge) for ref, edge in tile["edges"].items()},
            }
            for tile in atlas["tiles"]
        ],
        "nodes": [{"id": node["id"], "token": node["token"]} for node in atlas["nodes"]],
        "edges": [{"id": list(edge["id"]), "token": edge["token"]} for edge in atlas["edges"]],
        "ports": [
            {
                "id": port["id"],
                "token": port["token"],
                "coord": list(port["coord"]),
                "direction": port["direction"],
                "attached_nodes": list(port["attached_nodes"]),
            }
            for port in atlas["ports"]
        ],
    }


def _coordinate_for_tile(catan_map: CatanMap, target: LandTile | Port) -> Tuple[int, int, int]:
    for coordinate, tile in catan_map.tiles.items():
        if tile is target:
            return coordinate
    raise ValueError(f"tile not found: {target}")
