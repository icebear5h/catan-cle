"""Canonical atlas geometry/topology oracle and contract row helpers."""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from typing import cast

from cle.game_engine.models.board import STATIC_GRAPH
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens
from sft.board.symbolic_board_tasks._constants import COLORS, OFFSETS, _require
from sft.board.symbolic_board_tasks._types import Atlas, TokenSets
from sft.json_types import JsonDict, JsonList


@lru_cache(maxsize=1)
def _atlas() -> Atlas:
    atlas = atlas_metadata()
    positions: dict[str, tuple[int, int]] = {}
    node_tiles: TokenSets = {}
    node_edges: TokenSets = {}
    node_ports: TokenSets = {}
    nodes = {n["id"]: n["token"] for n in atlas["nodes"]}
    edges = {e["token"]: tuple(nodes[n] for n in e["id"]) for e in atlas["edges"]}
    touching: TokenSets = {token: set() for token in atlas_tokens()}
    for tile in atlas["tiles"]:
        q, _, r = tile["coord"]
        center = (2 * q + r, 3 * r)
        positions[tile["token"]] = center
        for ref, nid in tile["nodes"].items():
            dx, dy = OFFSETS[ref]
            point = (center[0] + dx, center[1] + dy)
            token = nodes[nid]
            _require(positions.setdefault(token, point) == point, "inconsistent atlas geometry")
            node_tiles.setdefault(token, set()).add(tile["token"])
            touching[token].add(tile["token"])
            touching[tile["token"]].add(token)
        for endpoints in tile["edges"].values():
            edge = f"<E{min(endpoints):02d}_{max(endpoints):02d}>"
            touching[edge].add(tile["token"])
            touching[tile["token"]].add(edge)
    graph: TokenSets = {token: set() for token in nodes.values()}
    for edge, (a, b) in edges.items():
        graph[a].add(b)
        graph[b].add(a)
        for node in (a, b):
            node_edges.setdefault(node, set()).add(edge)
            touching[node].add(edge)
            touching[edge].add(node)
    for port in atlas["ports"]:
        for nid in port["attached_nodes"]:
            node = nodes[nid]
            node_ports.setdefault(node, set()).add(port["token"])
            touching[node].add(port["token"])
            touching[port["token"]].add(node)
    tile_neighbors = {
        t["token"]: {u["token"] for u in atlas["tiles"] if u != t and
                     len(set(t["nodes"].values()) & set(u["nodes"].values())) == 2}
        for t in atlas["tiles"]
    }
    _require(len(positions) == 73 and len(set(positions.values())) == 73,
             "atlas positions must be unique")
    canonical_edges = {tuple(e["id"]) for e in atlas["edges"]}
    _require({tuple(sorted(e)) for e in STATIC_GRAPH.subgraph(range(54)).edges} == canonical_edges,
             "engine graph differs from canonical atlas")
    return Atlas(raw=atlas, positions=positions, graph=graph, edges=edges,
                 touching=touching, tile_neighbors=tile_neighbors, node_tiles=node_tiles,
                 node_edges=node_edges, node_ports=node_ports,
                 tokens=tuple(atlas_tokens()))


def atlas_geometry() -> Atlas:
    """Detached oracle-only integer geometry/topology; never serialize into prompts."""
    return copy.deepcopy(_atlas())


def _token(value: object, families: str) -> str:
    _require(isinstance(value, str) and value in _atlas()["tokens"]
             and value[1] in families, f"invalid canonical {families} token: {value!r}")
    return cast("str", value)


def _participants(colors: object) -> tuple[str, ...]:
    _require(isinstance(colors, list) and len(colors) == 4
             and all(isinstance(c, str) and c in COLORS for c in colors)
             and len(set(colors)) == 4, "expected four unique participant colors")
    return tuple(cast("list[str]", colors))


def _same(actual: object, expected: object, label: str) -> None:
    # JSON equality with type distinctions (True must not masquerade as node 1).
    _require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
             f"noncanonical {label}")


def _rows(contract: JsonDict, family: str, expected: list[JsonDict]) -> list[JsonDict]:
    rows = contract.get(family)
    _require(isinstance(rows, list) and len(rows) == len(expected), f"incomplete {family}")
    typed: list[JsonDict] = []
    for row, gold in zip(cast("JsonList", rows), expected, strict=True):
        _require(isinstance(row, dict), f"invalid {family} row")
        entry = cast("JsonDict", row)
        _same(entry.get("id"), gold["id"], family + " identity/order")
        _same(entry.get("token"), gold["token"], family + " token")
        typed.append(entry)
    return typed
