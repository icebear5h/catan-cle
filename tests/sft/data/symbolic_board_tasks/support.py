"""Shared helpers for symbolic board task geometry, routes, and projection contracts."""

from collections.abc import Container, Iterable
from functools import lru_cache

from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens, base_catan_map
from sft.board.symbolic_board_tasks import (
    decode_state,
    score_symbolic_task,
    symbolic_answer,
)

SymbolicState = dict[str, object]
RoadSpec = tuple[tuple[int, int], ...] | list[tuple[int, int]] | dict[tuple[int, int], str]


def reduced_state(
    roads: RoadSpec = (), buildings: dict[int, tuple[str, ...]] | None = None
) -> SymbolicState:
    """Canonical test board with explicitly reduced pieces, not a reachable-history claim."""
    board_map = base_catan_map()
    owners = roads if isinstance(roads, dict) else {tuple(sorted(e)): "RED" for e in roads}
    buildings = buildings or {}
    values: dict[str, str] = {}
    for tile in board_map.tiles_by_id.values():
        values[f"<T{tile.id:02d}>"] = f"{tile.resource.lower() if tile.resource else 'desert'} {tile.number if tile.number else 'none'}"
    for n in range(54):
        values[f"<N{n:02d}>"] = (" ".join(buildings[n]).lower() if n in buildings else "empty")
    for edge in atlas_metadata()["edges"]:
        owner = owners.get(tuple(edge["id"]))
        values[edge["token"]] = owner.lower() + " road" if owner else "empty"
    for port in board_map.ports_by_id.values():
        values[f"<P{port.id:02d}>"] = f"{port.resource.lower() if port.resource else '3:1'} port"
    values["robber"] = next(t for t, v in values.items() if v == "desert none")
    keys = [t for family in "TNEP" for t in sorted(atlas_tokens()) if t[1] == family] + ["robber"]
    return {"board": "; ".join(f"{t} {values[t]}" for t in keys),
            "colors": ["RED", "BLUE", "WHITE", "ORANGE"]}


def target(state: SymbolicState | None = None, **query: object) -> dict[str, object]:
    return {"state": state, "query": query}


def answer(task: str, state: SymbolicState | None = None, **query: object) -> str:
    return symbolic_answer("symbolic_" + task, target(state, **query))


def score(
    task: str, response: str, state: SymbolicState | None = None, **query: object
) -> dict[str, object] | None:
    return score_symbolic_task("untrusted expected", response,
                               {"task_type": "symbolic_" + task, "target": target(state, **query)})


def bitmask_trail(roads: Iterable[tuple[int, int]], blocked: Container[int]) -> int:
    """Independent exhaustive edge-bitmask DP, with no engine traversal helpers."""
    edges = sorted(roads)
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for i, (a, b) in enumerate(edges):
        adjacency.setdefault(a, []).append((b, 1 << i))
        adjacency.setdefault(b, []).append((a, 1 << i))

    @lru_cache(None)
    def visit(node: int, mask: int) -> int:
        if mask and node in blocked:
            return 0
        return max((1 + visit(other, mask | bit) for other, bit in adjacency.get(node, [])
                    if not mask & bit), default=0)

    return max((visit(n, 0) for n in adjacency), default=0)


def settlement_predicate(state: SymbolicState, color: str, setup: bool) -> set[str]:
    data = decode_state(state)
    edges = [tuple(e["id"]) for e in atlas_metadata()["edges"]]
    occupied = {int(n[2:-1]) for n in data["buildings"]}
    owned = {tuple(e["id"]) for e in atlas_metadata()["edges"] if data["roads"].get(e["token"]) == color}
    return {f"<N{n:02d}>" for n in range(54) if n not in occupied
            and not any((a == n and b in occupied) or (b == n and a in occupied) for a, b in edges)
            and (setup or any(n in e for e in owned))}
