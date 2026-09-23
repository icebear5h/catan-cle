"""Structural types for the atlas oracle and decoded detached board state."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

from evals.catan_board_bench.tokens import AtlasMetadata

TokenSets: TypeAlias = dict[str, set[str]]


class Atlas(TypedDict):
    """Detached integer geometry/topology derived once from the engine atlas."""

    raw: AtlasMetadata
    positions: dict[str, tuple[int, int]]
    graph: TokenSets
    edges: dict[str, tuple[str, ...]]
    touching: TokenSets
    tile_neighbors: TokenSets
    node_tiles: TokenSets
    node_edges: TokenSets
    node_ports: TokenSets
    tokens: tuple[str, ...]


class DecodedState(TypedDict):
    """Facts parsed out of a minimal `board`/`colors` state payload."""

    tiles: dict[str, tuple[str, int | None]]
    ports: dict[str, str]
    buildings: dict[str, tuple[str, str]]
    roads: dict[str, str]
    colors: tuple[str, ...]
    robber: str


class StatePayload(TypedDict):
    """The minimal serialized state accepted by `decode_state`."""

    board: str
    colors: list[str]


class Route(TypedDict):
    """A BFS route between two nodes, with `None` arms when unreachable."""

    nodes: list[str] | None
    edges: list[str] | None


class TransferFacts(TypedDict):
    """Cached longest-road and settlement-placement facts for a fixed board."""

    lengths: dict[str, int]
    award: str | None
    settlements: dict[tuple[str, str], frozenset[str]]
