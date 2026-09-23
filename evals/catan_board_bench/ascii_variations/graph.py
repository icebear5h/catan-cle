"""Canonical public graph construction, opaque aliases, and graph lookups."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

from evals.catan_board_bench.ascii_variations.codec import _json_digest
from evals.catan_board_bench.ascii_variations.facts import (
    AliasMap,
    FullEdge,
    FullFacts,
    FullNode,
    FullPort,
    FullTile,
)
from evals.catan_board_bench.ascii_variations.schema import (
    CORNER_ORDER,
    FACT_SCHEMA,
    SIDE_ORDER,
)
from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge

Contract = Mapping[str, object]


def full_public_graph_facts(
    contract: Contract,
    *,
    sample_id: str,
) -> tuple[FullFacts, AliasMap]:
    """Normalize a complete target-neutral public graph with opaque aliases."""

    aliases = build_aliases(contract, sample_id=sample_id)
    tile_aliases = aliases["tiles"]
    node_aliases = aliases["nodes"]
    edge_aliases = aliases["edges"]
    port_aliases = aliases["ports"]
    atlas_by_id = {item["id"]: item for item in atlas_metadata()["tiles"]}

    edge_tiles: dict[str, list[str]] = defaultdict(list)
    for tile in _rows(contract, "tiles"):
        tile_name = tile_aliases[str(tile["id"])]
        for edge in _items(tile["edges"], "tile edges"):
            edge_tiles[_edge_key(_ints(edge, "tile edge"))].append(tile_name)

    tiles: list[FullTile] = []
    for tile in _rows(contract, "tiles"):
        atlas_tile = atlas_by_id[_int(tile["id"], "tile id")]
        tiles.append(
            {
                "id": tile_aliases[str(tile["id"])],
                "cube": _ints(tile["coord"], "tile coord"),
                "resource": _optional_str(tile.get("resource"), "tile resource") or "DESERT",
                "number": _optional_int(tile.get("number"), "tile number"),
                "robber": bool(tile.get("has_robber")),
                "corners": {
                    direction: node_aliases[str(atlas_tile["nodes"][direction])]
                    for direction in CORNER_ORDER
                },
                "sides": {
                    direction: edge_aliases[_edge_key(atlas_tile["edges"][direction])]
                    for direction in SIDE_ORDER
                },
            }
        )

    nodes: list[FullNode] = []
    for node in _rows(contract, "nodes"):
        nodes.append(
            {
                "id": node_aliases[str(node["id"])],
                "color": _optional_str(node.get("color"), "node color"),
                "building": _optional_str(node.get("building"), "node building"),
                "tiles": sorted(
                    tile_aliases[str(tile_id)]
                    for tile_id in _items(node["adjacent_tiles"], "node tiles")
                ),
                "edges": sorted(
                    edge_aliases[_edge_key(_ints(edge, "node edge"))]
                    for edge in _items(node["adjacent_edges"], "node edges")
                ),
                "ports": sorted(
                    port_aliases[str(port_id)] for port_id in _items(node["port_ids"], "node ports")
                ),
            }
        )

    edges: list[FullEdge] = []
    for edge in _rows(contract, "edges"):
        edge_id = _ints(edge["id"], "edge id")
        key = _edge_key(edge_id)
        edges.append(
            {
                "id": edge_aliases[key],
                "nodes": [node_aliases[str(node_id)] for node_id in edge_id],
                "road": _optional_str(edge.get("road_color"), "edge road"),
                "tiles": sorted(edge_tiles[key]),
            }
        )

    ports: list[FullPort] = []
    for port in _rows(contract, "ports"):
        ports.append(
            {
                "id": port_aliases[str(port["id"])],
                "cube": _ints(port["coord"], "port coord"),
                "direction": _str(port["direction"], "port direction"),
                "resource": _optional_str(port.get("resource"), "port resource") or "GENERIC",
                "ratio": _str(port["ratio"], "port ratio"),
                "nodes": [
                    node_aliases[str(node_id)]
                    for node_id in _items(port["attached_nodes"], "port nodes")
                ],
            }
        )

    facts = canonicalize_full_facts(
        {
            "schema": FACT_SCHEMA,
            "tiles": tiles,
            "nodes": nodes,
            "edges": edges,
            "ports": ports,
        }
    )
    return facts, aliases


def build_aliases(contract: Contract, *, sample_id: str) -> AliasMap:
    seed = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:8], "big")
    rng = random.Random(seed)

    def permuted_map(values: Sequence[str], prefix: str) -> dict[str, str]:
        labels = [f"{prefix}{index:02d}" for index in range(len(values))]
        rng.shuffle(labels)
        return dict(zip(values, labels))

    tile_ids = [str(tile["id"]) for tile in _sorted_by_id(_rows(contract, "tiles"))]
    node_ids = [str(node["id"]) for node in _sorted_by_id(_rows(contract, "nodes"))]
    edge_ids = [
        _edge_key(_ints(edge["id"], "edge id"))
        for edge in sorted(
            _rows(contract, "edges"), key=lambda x: tuple(_ints(x["id"], "edge id"))
        )
    ]
    port_ids = [str(port["id"]) for port in _sorted_by_id(_rows(contract, "ports"))]
    return {
        "sample_id": sample_id,
        "seed": seed,
        "tiles": permuted_map(tile_ids, "T"),
        "nodes": permuted_map(node_ids, "N"),
        "edges": permuted_map(edge_ids, "E"),
        "ports": permuted_map(port_ids, "P"),
    }


def canonicalize_full_facts(facts: FullFacts) -> FullFacts:
    return {
        "schema": FACT_SCHEMA,
        "tiles": sorted(facts["tiles"], key=lambda item: item["id"]),
        "nodes": sorted(facts["nodes"], key=lambda item: item["id"]),
        "edges": sorted(facts["edges"], key=lambda item: item["id"]),
        "ports": sorted(facts["ports"], key=lambda item: item["id"]),
    }


def full_fact_digest(facts: FullFacts) -> str:
    return _json_digest(canonicalize_full_facts(facts))


def _dynamic_density(contract: Contract) -> int:
    return sum(node.get("building") is not None for node in _rows(contract, "nodes")) + sum(
        edge.get("road_color") is not None for edge in _rows(contract, "edges")
    )


def _tile_at(facts: FullFacts, cube: Sequence[int]) -> FullTile:
    target = list(cube)
    for tile in facts["tiles"]:
        if tile["cube"] == target:
            return tile
    raise ValueError(f"no land tile at cube {target}")


def _add_cube(left: Sequence[int], right: Sequence[int]) -> list[int]:
    return [left[index] + right[index] for index in range(3)]


def _edge_key(edge: Sequence[int]) -> str:
    left, right = canonical_edge((int(edge[0]), int(edge[1])))
    return f"{left},{right}"


def _rows(contract: Contract, key: str) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for row in _items(contract[key], f"contract {key}"):
        if not isinstance(row, dict):
            raise TypeError(f"contract {key} entry is not an object")
        rows.append(row)
    return rows


def _sorted_by_id(rows: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return sorted(rows, key=lambda row: _int(row["id"], "contract id"))


def _items(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} is not a list")
    return value


def _ints(value: object, label: str) -> list[int]:
    items = _items(value, label)
    ints = [item for item in items if isinstance(item, int)]
    if len(ints) != len(items):
        raise TypeError(f"{label} is not a list of integers")
    return ints


def _int(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{label} is not an integer")
    return value


def _str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} is not a string")
    return value


def _optional_str(value: object, label: str) -> str | None:
    return None if value is None else _str(value, label)


def _optional_int(value: object, label: str) -> int | None:
    return None if value is None else _int(value, label)
