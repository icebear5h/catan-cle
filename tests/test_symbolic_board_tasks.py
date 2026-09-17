"""Offline oracle checks; reduced graph fixtures are tests, never dataset sources."""

import copy
import json
import random
from functools import lru_cache
from itertools import combinations

import pytest

from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens import atlas_metadata, atlas_tokens, base_catan_map
from sft.scripts.build_symbolic_board_dataset import graph_case_coverage, transfer_projection
from sft.symbolic_board_tasks import (
    STATIC_TASKS, SYMBOLIC_TASKS, TRAIN_TASKS, TRANSFER_TASKS, PhysicalStateError,
    atlas_geometry, decode_state, detached_board, owned_route, score_symbolic_task,
    symbolic_answer, symbolic_prompt,
)


def reduced_state(roads=(), buildings=None):
    """Canonical test board with explicitly reduced pieces, not a reachable-history claim."""
    board_map = base_catan_map()
    owners = roads if isinstance(roads, dict) else {tuple(sorted(e)): "RED" for e in roads}
    buildings = buildings or {}
    values = {}
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


def target(state=None, **query):
    return {"state": state, "query": query}


def answer(task, state=None, **query):
    return symbolic_answer("symbolic_" + task, target(state, **query))


def score(task, response, state=None, **query):
    return score_symbolic_task("untrusted expected", response,
                               {"task_type": "symbolic_" + task, "target": target(state, **query)})


def bitmask_trail(roads, blocked):
    """Independent exhaustive edge-bitmask DP, with no engine traversal helpers."""
    edges = sorted(roads)
    adjacency = {}
    for i, (a, b) in enumerate(edges):
        adjacency.setdefault(a, []).append((b, 1 << i))
        adjacency.setdefault(b, []).append((a, 1 << i))

    @lru_cache(None)
    def visit(node, mask):
        if mask and node in blocked:
            return 0
        return max((1 + visit(other, mask | bit) for other, bit in adjacency.get(node, [])
                    if not mask & bit), default=0)

    return max((visit(n, 0) for n in adjacency), default=0)


def settlement_predicate(state, color, setup):
    data = decode_state(state)
    edges = [tuple(e["id"]) for e in atlas_metadata()["edges"]]
    occupied = {int(n[2:-1]) for n in data["buildings"]}
    owned = {tuple(e["id"]) for e in atlas_metadata()["edges"] if data["roads"].get(e["token"]) == color}
    return {f"<N{n:02d}>" for n in range(54) if n not in occupied
            and not any((a == n and b in occupied) or (b == n and a in occupied) for a, b in edges)
            and (setup or any(n in e for e in owned))}


def test_task_boundary_and_malformed_metadata_fail_closed():
    assert not TRAIN_TASKS & TRANSFER_TASKS
    assert STATIC_TASKS < TRAIN_TASKS < SYMBOLIC_TASKS
    assert score_symbolic_task("", "", {"task_type": "not_this_task"}) is None
    for metadata in ({"task_type": "symbolic_longest_lengths"},
                     {"task_type": "symbolic_direction", "target": {}},
                     {"task_type": "symbolic_direction", "target": target(a=[], b="<N01>", direction="left")},
                     {"task_type": []}):
        with pytest.raises(ValueError):
            score_symbolic_task("yes", "yes", metadata)
    with pytest.raises(ValueError):
        answer("local_constraint", reduced_state(), color="RED", node="<N00>",
               predicate="empty_and_distance_and_owned_road")


def test_integer_geometry_independent_axis_and_exact_neighbor_orientation():
    atlas = atlas_geometry()
    assert atlas["positions"]["<T00>"] == (0, 0)
    assert atlas["positions"]["<N00>"] == (0, -2)
    for tile in atlas_metadata()["tiles"]:
        q, _, r = tile["coord"]
        assert atlas["positions"][tile["token"]] == (2 * q + r, 3 * r)
    assert answer("direction", a="<N00>", b="<N03>", direction="left") == "no"
    assert answer("direction", a="<N00>", b="<N03>", direction="above") == "yes"
    # Both axes true on an asymmetric diagonal; a dominant-axis helper would fail one.
    pair = next((a, b) for a, b in combinations(sorted(atlas["graph"]), 2)
                if atlas["positions"][a][0] < atlas["positions"][b][0]
                and atlas["positions"][a][1] < atlas["positions"][b][1])
    assert answer("direction", a=pair[0], b=pair[1], direction="left") == "yes"
    assert answer("direction", a=pair[0], b=pair[1], direction="above") == "yes"
    assert answer("oriented_step", token="<N00>", direction="SOUTHEAST") == "<N01>"
    assert answer("oriented_step", token="<N00>", direction="SOUTH") == "NONE"
    assert answer("oriented_step", token="<T00>", direction="EAST") == "<T01>"
    assert max(len(v) for v in atlas["graph"].values()) == 3
    assert max(len(v) for v in atlas["tile_neighbors"].values()) == 6
    for node, neighbors in atlas["graph"].items():
        assert set(answer("neighbors", token=node).split()) == neighbors
    for tile, neighbors in atlas["tile_neighbors"].items():
        assert set(answer("neighbors", token=tile).split()) == neighbors
    with pytest.raises(ValueError):
        answer("direction_choice", a="<N00>", b="<N03>", direction="left", choices=["<N00>", "<N03>"])


def test_strict_boolean_choice_sets_and_gold_recomputation():
    q = dict(a="<N00>", b="<N03>", direction="above")
    assert score("direction", " yes\n", **q)["correct"]
    for value in ("Yes", "yes because", "yes no", "true", '"yes"'):
        assert not score("direction", value, **q)["correct"]
    assert score("direction_choice", "<N00>", choices=["<N03>", "<N00>"], **q)["correct"]
    assert not score("direction_choice", "<N00> <N03>", choices=["<N03>", "<N00>"], **q)["correct"]
    gold = answer("incidence", token="<N00>", family="T").split()
    assert score("incidence", " ".join(reversed(gold)), token="<N00>", family="T")["correct"]
    for bad in (" ".join(gold + [gold[0]]), " ".join(gold + ["<T18>"]), "answer: " + " ".join(gold)):
        assert not score("incidence", bad, token="<N00>", family="T")["correct"]
    state = reduced_state([(0, 1)])
    assert score("owned_roads", "<E00_01>", state, color="RED")["correct"]
    assert not score("owned_roads", "<E00_01>", state, color="BLUE")["correct"]
    assert not score("owned_roads", "NONE", state, color="RED")["correct"]
    assert score("owned_roads", "NONE", state, color="BLUE")["correct"]
    assert not score("owned_roads", "", state, color="BLUE")["correct"]


def test_detachment_bidirectional_roads_and_distance_cache_no_mutation():
    state = reduced_state([(0, 1), (1, 2)], {4: ("BLUE", "settlement")})
    before = copy.deepcopy(state)
    graph_before = (dict(STATIC_GRAPH.nodes(data=True)), list(STATIC_GRAPH.edges(data=True)))
    random_before = random.getstate()
    board = detached_board(state)
    assert board.roads[0, 1] == board.roads[1, 0] == Color.RED
    assert not {3, 4, 5, 15} & board.board_buildable_ids
    board.buildable_subgraph.remove_edge(0, 1)
    board.map.tiles_by_id[0].number = 12
    board.roads.clear()
    again = detached_board(state)
    assert again.roads[1, 0] == Color.RED
    assert again.map.tiles_by_id[0].number == decode_state(state)["tiles"]["<T00>"][1]
    assert state == before and random.getstate() == random_before
    assert graph_before == (dict(STATIC_GRAPH.nodes(data=True)), list(STATIC_GRAPH.edges(data=True)))
    geometry = atlas_geometry()
    geometry["graph"]["<N00>"].clear()
    assert atlas_geometry()["graph"]["<N00>"]


def test_setup_normal_settlement_and_touching_filters_against_independent_predicate():
    state = reduced_state([(0, 1), (1, 2), (2, 3)], {5: ("BLUE", "settlement")})
    for phase in ("setup", "normal"):
        gold = settlement_predicate(state, "RED", phase == "setup")
        got = answer("settlement_locations", state, color="RED", phase=phase, near=None)
        assert (set() if got == "NONE" else set(got.split())) == gold
        assert score("settlement_locations", got, state, color="RED", phase=phase, near=None)["correct"]
    assert answer("settlement_locations", state, color="BLUE", phase="normal", near=None) == "NONE"
    assert answer("settlement_locations", state, color="BLUE", phase="setup", near=None) != "NONE"
    atlas = atlas_geometry()
    for near in ({"kind": "tile", "value": "<T00>"}, {"kind": "port", "value": "<P00>"},
                 {"kind": "resource", "value": "wood"}):
        touching = set(answer("near_nodes", state, near=near).split()) - {"NONE"}
        for node in atlas["graph"]:
            assert answer("near", state, node=node, near=near) == ("yes" if node in touching else "no")
        got = set(answer("settlement_locations", state, color="RED", phase="setup", near=near).split()) - {"NONE"}
        assert got == touching & settlement_predicate(state, "RED", True)


def test_route_ties_zero_unreachable_and_symmetric_enemy_endpoints():
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
def test_longest_chain_branch_cycle_revisited_vertices_and_blockers(roads, blocked, length):
    state = reduced_state(roads, {n: ("BLUE", "settlement") for n in blocked})
    assert bitmask_trail(roads, set(blocked)) == length
    result = json.loads(answer("longest_lengths", state))
    assert result == {"RED": length, "BLUE": 0, "WHITE": 0, "ORANGE": 0}
    assert score("longest_lengths", json.dumps(result), state)["correct"]
    for value in (True, float(length), -1, "3"):
        result["RED"] = value
        assert not score("longest_lengths", json.dumps(result), state)["correct"]


def test_exhaustive_ring_subgraphs_against_independent_bitmask_oracle():
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    for mask in range(1 << len(edges)):
        roads = [edge for i, edge in enumerate(edges) if mask & (1 << i)]
        for blocked in (set(), {0}):
            state = reduced_state(roads, {n: ("BLUE", "settlement") for n in blocked})
            assert json.loads(answer("longest_lengths", state))["RED"] == bitmask_trail(roads, blocked)


def test_all_max_leaders_zero_and_award_unknown_incumbent():
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


def test_state_shape_physical_invariants_and_prompt_no_derived_input():
    state = reduced_state([(0, 1)], {3: ("BLUE", "city")})
    for bad in ({**state, "lengths": {"RED": 10}}, {**state, "colors": ["RED"] * 4},
                {**state, "board": state["board"] + "; <N00> empty"},
                {**state, "board": state["board"].replace("<N01> empty", "<N00> empty")},
                {**state, "board": state["board"].replace("red road", "green road")}):
        with pytest.raises(ValueError):
            decode_state(bad)
    with pytest.raises(PhysicalStateError, match="distance"):
        decode_state(reduced_state(buildings={0: ("RED", "settlement"), 1: ("BLUE", "city")}))
    with pytest.raises(PhysicalStateError, match="supply"):
        decode_state(reduced_state([e["id"] for e in atlas_metadata()["edges"][:16]]))
    with pytest.raises(ValueError):
        answer("owned_roads", state, color="GREEN")
    t = target(state, color="RED", phase="normal", near=None)
    prompt = symbolic_prompt("symbolic_settlement_locations", t)
    assert prompt.count(state["board"]) == 1
    for forbidden in ("coord", "node_tokens", "adjacent_tiles", "road_lengths", "has_longest_road", "legal_nodes", "<image>"):
        assert forbidden not in prompt
    # Only state and authored query/rules can enter the prompt.
    assert "Ignore hand, piece supply, and whose turn it is" in prompt
    assert t == target(state, color="RED", phase="normal", near=None)


def test_relevant_projection_removes_irrelevant_facts_but_keeps_rule_inputs():
    state = reduced_state([(0, 1)], {3: ("BLUE", "settlement")})
    setup = dict(color="RED", phase="setup", near=None)
    other_color = dict(setup, color="BLUE")
    assert transfer_projection("symbolic_settlement_locations", state, setup) == transfer_projection(
        "symbolic_settlement_locations", state, other_color)
    normal = dict(setup, phase="normal")
    assert transfer_projection("symbolic_settlement_locations", state, normal) != transfer_projection(
        "symbolic_settlement_locations", state, dict(normal, color="BLUE"))
    changed = copy.deepcopy(state)
    changed["colors"] = list(reversed(state["colors"]))
    changed["board"] = changed["board"].rsplit("robber ", 1)[0] + "robber <T01>"
    for task, query in (("symbolic_settlement_locations", normal), ("symbolic_longest_lengths", {})):
        assert transfer_projection(task, state, query) == transfer_projection(task, changed, query)
    # A city versus settlement is the same blocker; the blocker itself is essential.
    blocked = reduced_state([(0, 1), (1, 2)], {1: ("BLUE", "settlement")})
    city = reduced_state([(0, 1), (1, 2)], {1: ("BLUE", "city")})
    clear = reduced_state([(0, 1), (1, 2)])
    assert transfer_projection("symbolic_longest_lengths", blocked, {}) == transfer_projection("symbolic_longest_lengths", city, {})
    assert transfer_projection("symbolic_longest_lengths", blocked, {}) != transfer_projection("symbolic_longest_lengths", clear, {})


def test_graph_case_audit_detects_positive_cases_not_just_zero_real_pool_counts():
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    states = [reduced_state(ring + [(0, 20)], {0: ("BLUE", "settlement")}),
              reduced_state({**dict.fromkeys(ring[:-1], "RED"),
                             **dict.fromkeys([(24, 25), (25, 26), (26, 27), (27, 28), (28, 29)], "BLUE")})]
    records = [{"state": state, "provenance": {"state_id": f"reduced_fixture_{i}",
                                               "board_fact_sha256": "unit_test_only"}}
               for i, state in enumerate(states)]
    coverage = graph_case_coverage(records)
    assert coverage["counts"]["cycle"] == 1
    assert coverage["counts"]["effective_blocker"] == 1
    assert coverage["counts"]["branch"] == 1
    assert coverage["counts"]["tied_max_ge5"] == 1
    assert coverage["missing_coverage"] == []
