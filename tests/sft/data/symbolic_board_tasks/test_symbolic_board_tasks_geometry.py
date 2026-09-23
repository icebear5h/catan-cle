"""Task boundaries, geometry, and boolean choice sets."""

import copy
import random
from itertools import combinations

import pytest

from cle.game_engine.models.board import STATIC_GRAPH
from cle.game_engine.models.player import Color
from evals.catan_board_bench.tokens import atlas_metadata
from sft.board.symbolic_board_tasks import (
    STATIC_TASKS,
    SYMBOLIC_TASKS,
    TRAIN_TASKS,
    TRANSFER_TASKS,
    atlas_geometry,
    decode_state,
    detached_board,
    score_symbolic_task,
)

from .support import answer, reduced_state, score, settlement_predicate, target


def test_task_boundary_and_malformed_metadata_fail_closed() -> None:
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


def test_integer_geometry_independent_axis_and_exact_neighbor_orientation() -> None:
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


def test_strict_boolean_choice_sets_and_gold_recomputation() -> None:
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


def test_detachment_bidirectional_roads_and_distance_cache_no_mutation() -> None:
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


def test_setup_normal_settlement_and_touching_filters_against_independent_predicate() -> None:
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
