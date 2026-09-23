"""Route ties, longest chains, and award leadership."""

import json

import pytest

from sft.board.symbolic_board_tasks import (
    owned_route,
)

from .support import answer, bitmask_trail, reduced_state, score


def test_route_ties_zero_unreachable_and_symmetric_enemy_endpoints() -> None:
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    state = reduced_state(ring)
    q = dict(color="RED", start="<N00>", end="<N03>")
    for nodes, edges in (([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)]),
                         ([0, 5, 4, 3], [(0, 5), (4, 5), (3, 4)])):
        response = json.dumps({"nodes": [f"<N{n:02d}>" for n in nodes],
                               "edges": [f"<E{a:02d}_{b:02d}>" for a, b in edges]})
        assert score("shortest_route", response, state, **q)["correct"]
    for bad in ('{"nodes":null,"edges":null}',
                '{"nodes":["<N00>","<N03>"],"edges":["<E00_01>"]}',
                '{"nodes":[],"edges":[],"extra":0}',
                '{"nodes":null,"nodes":[],"edges":[]}',
                '[]'):
        assert not score("shortest_route", bad, state, **q)["correct"]
    assert owned_route(state, "BLUE", "<N00>", "<N03>") == {"nodes": None, "edges": None}
    assert score("shortest_route", '{"nodes":null,"edges":null}', state,
                 color="BLUE", start="<N00>", end="<N03>")["correct"]
    assert owned_route(state, "RED", "<N00>", "<N00>") == {"nodes": ["<N00>"], "edges": []}
    assert score("shortest_route", '{"nodes":["<N00>"],"edges":[]}', state,
                 color="RED", start="<N00>", end="<N00>")["correct"]
    blocked = reduced_state([(0, 1), (1, 2), (2, 3)], {1: ("BLUE", "city")})
    for a, b in ((1, 3), (3, 1), (0, 1), (1, 0)):
        assert owned_route(blocked, "RED", f"<N{a:02d}>", f"<N{b:02d}>")["nodes"] is not None
    for a, b in ((0, 3), (3, 0)):
        assert owned_route(blocked, "RED", f"<N{a:02d}>", f"<N{b:02d}>")["nodes"] is None
    illegal = '{"nodes":["<N00>","<N01>","<N02>","<N03>"],"edges":["<E00_01>","<E01_02>","<E02_03>"]}'
    assert not score("shortest_route", illegal, blocked, color="RED", start="<N00>", end="<N03>")["correct"]


@pytest.mark.parametrize("roads,blocked,length", [
    ([(0, 1), (1, 2), (2, 3)], [], 3),
    ([(0, 1), (1, 2), (1, 6)], [], 2),
    ([(0, 1), (1, 2), (2, 3)], [1], 2),
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)], [], 6),
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 20), (20, 22)], [], 8),
    ([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 20), (20, 22)], [0], 6),
])
def test_longest_chain_branch_cycle_revisited_vertices_and_blockers(
    roads: list[tuple[int, int]], blocked: list[int], length: int
) -> None:
    state = reduced_state(roads, {n: ("BLUE", "settlement") for n in blocked})
    assert bitmask_trail(roads, set(blocked)) == length
    result = json.loads(answer("longest_lengths", state))
    assert result == {"RED": length, "BLUE": 0, "WHITE": 0, "ORANGE": 0}
    assert score("longest_lengths", json.dumps(result), state)["correct"]
    for value in (True, float(length), -1, "3"):
        result["RED"] = value
        assert not score("longest_lengths", json.dumps(result), state)["correct"]


def test_exhaustive_ring_subgraphs_against_independent_bitmask_oracle() -> None:
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    for mask in range(1 << len(edges)):
        roads = [edge for i, edge in enumerate(edges) if mask & (1 << i)]
        for blocked in (set(), {0}):
            state = reduced_state(roads, {n: ("BLUE", "settlement") for n in blocked})
            assert json.loads(answer("longest_lengths", state))["RED"] == bitmask_trail(roads, blocked)


def test_all_max_leaders_zero_and_award_unknown_incumbent() -> None:
    state = reduced_state()
    assert set(answer("longest_leaders", state).split()) == set(state["colors"])
    assert answer("longest_award", state) == "NONE"
    red = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    blue = [(24, 25), (25, 26), (26, 27), (27, 28), (28, 29)]
    tied = reduced_state({**dict.fromkeys(red, "RED"), **dict.fromkeys(blue, "BLUE")})
    assert answer("longest_leaders", tied) == "BLUE RED"
    assert not score("longest_leaders", "RED RED BLUE", tied)["correct"]
    with pytest.raises(ValueError, match="incumbent"):
        score("longest_award", "BLUE", tied)
    unique = reduced_state(red)
    assert answer("longest_award", unique) == "RED"
    assert answer("longest_award", reduced_state(red[:4])) == "NONE"
