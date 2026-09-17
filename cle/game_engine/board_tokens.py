"""Literal trained board-location tokens shared by the engine and evaluations."""

from __future__ import annotations

from cle.game_engine.models.map import NUM_NODES, NUM_TILES


def node_token(node_id: int) -> str:
    _check_range("node_id", node_id, NUM_NODES)
    return f"<N{node_id:02d}>"


def tile_token(tile_id: int) -> str:
    _check_range("tile_id", tile_id, NUM_TILES)
    return f"<T{tile_id:02d}>"


def port_token(port_id: int) -> str:
    _check_range("port_id", port_id, 9)
    return f"<P{port_id:02d}>"


def edge_token(edge: tuple[int, int]) -> str:
    a, b = canonical_edge(edge)
    return f"<E{a:02d}_{b:02d}>"


def canonical_edge(edge: tuple[int, int]) -> tuple[int, int]:
    a, b = edge
    if a == b:
        raise ValueError(f"edge endpoints must differ: {edge}")
    _check_range("edge node", a, NUM_NODES)
    _check_range("edge node", b, NUM_NODES)
    return (a, b) if a < b else (b, a)


def _check_range(name: str, value: int, size: int) -> None:
    if value < 0 or value >= size:
        raise ValueError(f"{name} must be in [0, {size - 1}], got {value}")
