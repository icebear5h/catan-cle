"""Atlas tokens and the static land-graph basis for every fact table."""

from __future__ import annotations

import functools
from typing import Dict, Tuple

from cle.game_engine.models.board import STATIC_GRAPH, base_map
from cle.game_engine.models.board.graph import NodeGraph
from sft.board.symbolic_board_tasks import _atlas as symbolic_atlas

SCHEMA = "catan_board_atlas/v1"

LAND_NODES: Tuple[int, ...] = tuple(sorted(base_map.land_nodes))
LAND_GRAPH: NodeGraph = STATIC_GRAPH.subgraph(LAND_NODES).copy()

NONE_ANSWER = "NONE"


@functools.lru_cache(maxsize=1)
def _positions() -> Dict[str, Tuple[int, int]]:
    """Render-frame (x, y) per token, the basis for every orientation claim here."""
    return symbolic_atlas()["positions"]


def node_token(node_id: int) -> str:
    return f"<N{node_id:02d}>"


def tile_token(tile_id: int) -> str:
    return f"<T{tile_id:02d}>"


def port_token(port_id: int) -> str:
    return f"<P{port_id:02d}>"


def edge_token(a: int, b: int) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"<E{lo:02d}_{hi:02d}>"
