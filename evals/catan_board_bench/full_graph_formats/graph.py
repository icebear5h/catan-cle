"""Minimal/full graph projection, incidence reconstruction, and validation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from evals.catan_board_bench.ascii_variations import (
    CORNER_ORDER,
    FACT_SCHEMA,
    SIDE_ORDER,
    canonicalize_full_facts,
)
from evals.catan_board_bench.ascii_variations.facts import (
    FullFacts,
    MinimalGraph,
    is_full_port,
    is_full_tile,
    is_minimal_edge,
    is_minimal_node,
    typed_items,
)
from evals.catan_board_bench.full_graph_formats.schema import (
    _EXPECTED_COUNTS,
    _SIDE_ENDPOINTS,
    MINIMAL_SCHEMA,
)


def minimal_graph_facts(facts: FullFacts) -> MinimalGraph:
    """Remove redundant incidence lists while retaining all source information."""

    facts = _validated_full_facts(facts)
    return {
        "schema": MINIMAL_SCHEMA,
        "tiles": [
            {
                "id": tile["id"],
                "cube": list(tile["cube"]),
                "resource": tile["resource"],
                "number": tile["number"],
                "robber": tile["robber"],
                "corners": {direction: tile["corners"][direction] for direction in CORNER_ORDER},
                "sides": {direction: tile["sides"][direction] for direction in SIDE_ORDER},
            }
            for tile in facts["tiles"]
        ],
        "nodes": [
            {
                "id": node["id"],
                "color": node["color"],
                "building": node["building"],
            }
            for node in facts["nodes"]
        ],
        "edges": [
            {
                "id": edge["id"],
                "nodes": list(edge["nodes"]),
                "road": edge["road"],
            }
            for edge in facts["edges"]
        ],
        "ports": [
            {
                "id": port["id"],
                "cube": list(port["cube"]),
                "direction": port["direction"],
                "resource": port["resource"],
                "ratio": port["ratio"],
                "nodes": list(port["nodes"]),
            }
            for port in facts["ports"]
        ],
    }


def expand_minimal_graph(payload: Mapping[str, object]) -> FullFacts:
    """Reconstruct redundant incidence fields from a minimal typed graph."""

    graph = _validate_minimal_payload(payload)
    tiles = graph["tiles"]
    nodes = graph["nodes"]
    edges = graph["edges"]
    ports = graph["ports"]

    node_ids = {node["id"] for node in nodes}
    edge_ids = {edge["id"] for edge in edges}
    node_tiles: dict[str, set[str]] = defaultdict(set)
    node_edges: dict[str, set[str]] = defaultdict(set)
    node_ports: dict[str, set[str]] = defaultdict(set)
    edge_tiles: dict[str, set[str]] = defaultdict(set)

    for tile in tiles:
        for node_id in tile["corners"].values():
            if node_id not in node_ids:
                raise ValueError(f"tile {tile['id']} references unknown node {node_id}")
            node_tiles[node_id].add(tile["id"])
        for direction, edge_id in tile["sides"].items():
            if edge_id not in edge_ids:
                raise ValueError(f"tile {tile['id']} references unknown edge {edge_id}")
            edge_tiles[edge_id].add(tile["id"])
            endpoint_directions = _SIDE_ENDPOINTS[direction]
            expected_nodes = {
                tile["corners"][endpoint_directions[0]],
                tile["corners"][endpoint_directions[1]],
            }
            edge = next(item for item in edges if item["id"] == edge_id)
            if set(edge["nodes"]) != expected_nodes:
                raise ValueError(f"edge {edge_id} endpoints disagree with tile {tile['id']}")

    for edge in edges:
        for node_id in edge["nodes"]:
            if node_id not in node_ids:
                raise ValueError(f"edge {edge['id']} references unknown node {node_id}")
            node_edges[node_id].add(edge["id"])
    for port in ports:
        for node_id in port["nodes"]:
            if node_id not in node_ids:
                raise ValueError(f"port {port['id']} references unknown node {node_id}")
            node_ports[node_id].add(port["id"])

    facts: FullFacts = {
        "schema": FACT_SCHEMA,
        "tiles": tiles,
        "nodes": [
            {
                **node,
                "tiles": sorted(node_tiles[node["id"]]),
                "edges": sorted(node_edges[node["id"]]),
                "ports": sorted(node_ports[node["id"]]),
            }
            for node in nodes
        ],
        "edges": [{**edge, "tiles": sorted(edge_tiles[edge["id"]])} for edge in edges],
        "ports": ports,
    }
    return _validated_full_facts(facts)


def _validated_full_facts(facts: FullFacts) -> FullFacts:
    canonical = canonicalize_full_facts(facts)
    counts = (
        len(canonical["tiles"]),
        len(canonical["nodes"]),
        len(canonical["edges"]),
        len(canonical["ports"]),
    )
    if counts != _EXPECTED_COUNTS:
        raise ValueError(f"expected full 19/54/72/9 graph, found {counts}")
    if canonical.get("schema") != FACT_SCHEMA:
        raise ValueError(f"unexpected canonical fact schema: {canonical.get('schema')}")
    collection_ids = (
        ("tiles", [item["id"] for item in canonical["tiles"]]),
        ("nodes", [item["id"] for item in canonical["nodes"]]),
        ("edges", [item["id"] for item in canonical["edges"]]),
        ("ports", [item["id"] for item in canonical["ports"]]),
    )
    for collection, ids in collection_ids:
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate IDs in {collection}")
    return canonical


def _validate_minimal_payload(payload: Mapping[str, object]) -> MinimalGraph:
    if not isinstance(payload, dict):
        raise ValueError("minimal graph must be an object")
    if set(payload) != {"schema", "tiles", "nodes", "edges", "ports"}:
        raise ValueError(f"unexpected minimal graph keys: {sorted(payload)}")
    if payload["schema"] != MINIMAL_SCHEMA:
        raise ValueError(f"unexpected minimal graph schema: {payload['schema']!r}")
    counts = tuple(len(_collection(payload, key)) for key in ("tiles", "nodes", "edges", "ports"))
    if counts != _EXPECTED_COUNTS:
        raise ValueError(f"incomplete minimal graph: {counts}")
    expected_keys = {
        "tiles": {"id", "cube", "resource", "number", "robber", "corners", "sides"},
        "nodes": {"id", "color", "building"},
        "edges": {"id", "nodes", "road"},
        "ports": {"id", "cube", "direction", "resource", "ratio", "nodes"},
    }
    for collection, keys in expected_keys.items():
        ids = []
        for item in _collection(payload, collection):
            if not isinstance(item, dict) or set(item) != keys:
                raise ValueError(f"invalid {collection} item keys")
            ids.append(item["id"])
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate minimal IDs in {collection}")
    graph: MinimalGraph = {
        "schema": MINIMAL_SCHEMA,
        "tiles": typed_items(payload["tiles"], is_full_tile, "tiles"),
        "nodes": typed_items(payload["nodes"], is_minimal_node, "nodes"),
        "edges": typed_items(payload["edges"], is_minimal_edge, "edges"),
        "ports": typed_items(payload["ports"], is_full_port, "ports"),
    }
    for tile in graph["tiles"]:
        if tuple(tile["corners"]) != CORNER_ORDER:
            raise ValueError(f"tile {tile['id']} has invalid corner mapping")
        if tuple(tile["sides"]) != SIDE_ORDER:
            raise ValueError(f"tile {tile['id']} has invalid side mapping")
        if len(tile["cube"]) != 3 or sum(tile["cube"]) != 0:
            raise ValueError(f"tile {tile['id']} has invalid cube coordinate")
    for edge in graph["edges"]:
        if len(edge["nodes"]) != 2 or edge["nodes"][0] == edge["nodes"][1]:
            raise ValueError(f"edge {edge['id']} has invalid endpoints")
    for port in graph["ports"]:
        if len(port["cube"]) != 3 or sum(port["cube"]) != 0:
            raise ValueError(f"port {port['id']} has invalid cube coordinate")
        if len(port["nodes"]) != 2 or port["nodes"][0] == port["nodes"][1]:
            raise ValueError(f"port {port['id']} has invalid nodes")
    return graph


def _collection(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"minimal graph {key} must be a list")
    return value
