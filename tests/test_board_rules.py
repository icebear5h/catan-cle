"""Reduced board positions for independent road and award rule checks."""

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.board import STATIC_GRAPH, longest_acyclic_path
from cle.game_engine.models.enums import CITY, ROAD, SETTLEMENT, VICTORY_POINT
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    build_road,
    build_settlement,
    maintain_longest_road,
    player_key,
)


COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


def _position(paths=(), settlements=()):
    """Reconstruct free pieces, not a claim of a reachable full game history."""
    game = GameEngine(COLORS, seed=1, shuffle_players=False)
    state = game.state
    for color, node in settlements:
        state.board.build_settlement(color, node, initial_build_phase=True)
        build_settlement(state, color, node, is_free=True)
    for color, path in paths:
        for left, right in zip(path, path[1:]):
            assert STATIC_GRAPH.has_edge(left, right)
            assert left in state.board.map.land_nodes and right in state.board.map.land_nodes
            assert (left, right) not in state.board.roads
            state.board.roads[left, right] = state.board.roads[right, left] = color
            build_road(state, color, tuple(sorted((left, right))), is_free=True)
        assert len(state.buildings_by_color[color][ROAD]) <= 15
    maintain_longest_road(state, *state.board.recompute_road_state())
    return game


@pytest.mark.parametrize(
    "paths,length",
    [
        ((), 0),
        (((0, 1, 2, 9, 10),), 4),
        (((1, 0, 20), (1, 2, 9), (1, 6, 23)), 4),
        (((0, 1, 2, 3, 4, 5, 0),), 6),
        (((0, 1, 2, 3, 4, 5, 0), (0, 20)), 7),
        (((1, 0, 5, 4, 3, 2), (1, 6, 7, 8, 9, 2), (1, 2)), 11),
        (((0, 1, 2, 9, 10), (29, 30, 31, 32)), 4),
    ],
    ids=["empty", "chain", "three-arms", "loop", "loop-tail", "theta", "disconnected"],
)
def test_longest_road_counts_trails_not_vertices_or_total_pieces(paths, length):
    game = _position(tuple((Color.RED, path) for path in paths))
    board = game.state.board
    assert board.road_lengths[Color.RED] == length
    assert board.road_color == (Color.RED if length >= 5 else None)
    assert game.observe(Color.RED).my_longest_road_length == length
    for trail in board.continuous_roads_by_player(Color.RED):
        assert len(trail) == len(set(trail))
    assert longest_acyclic_path(board, set(), Color.RED) == []


@pytest.mark.parametrize("piece", [SETTLEMENT, CITY])
def test_enemy_endpoints_count_terminal_edges_without_joining_roads(piece):
    game = _position(
        ((Color.RED, (29, 10, 11, 12, 13, 14, 15)),),
        ((Color.BLUE, 29), (Color.WHITE, 15)),
    )
    board = game.state.board
    if piece == CITY:
        board.build_city(Color.BLUE, 29)
        board.build_city(Color.WHITE, 15)
        board.recompute_road_state()
    assert board.road_lengths[Color.RED] == 6
    assert len(board.continuous_roads_by_player(Color.RED)[0]) == 6
    assert (15, 17) not in board.buildable_edges(Color.RED)
    assert (28, 29) not in board.buildable_edges(Color.RED)


@pytest.mark.parametrize("blockers,length,components", [((0,), 6, 1), ((0, 3), 3, 2)])
def test_blocked_loop_can_end_but_cannot_pass_through_enemy(blockers, length, components):
    game = _position(
        ((Color.RED, (0, 1, 2, 3, 4, 5, 0)),),
        tuple((Color.BLUE, node) for node in blockers),
    )
    board = game.state.board
    assert board.road_lengths[Color.RED] == length
    assert len(board.connected_components[Color.RED]) == components
    assert max(map(len, board.continuous_roads_by_player(Color.RED))) == length


def test_own_settlement_does_not_cut_road():
    game = _position(((Color.RED, (0, 1, 2, 3, 4, 5, 0)),), ((Color.RED, 0),))
    assert game.state.board.road_lengths[Color.RED] == 6


def test_recomputed_three_road_junction_splits_all_arms():
    game = _position(
        ((Color.RED, (1, 0, 20)), (Color.RED, (1, 2, 9, 10)),
         (Color.RED, (1, 6, 23, 52, 51)))
    )
    board = game.state.board
    assert board.road_lengths[Color.RED] == 7
    result = board.build_settlement(Color.BLUE, 1, initial_build_phase=True)
    assert result[:2] == (Color.RED, None)
    assert board.road_lengths[Color.RED] == 4
    assert len(board.connected_components[Color.RED]) == 3
    assert sorted(map(len, board.continuous_roads_by_player(Color.RED))) == [2, 3, 4]


@pytest.mark.parametrize("reverse", [False, True])
def test_road_may_approach_enemy_but_cannot_extend_from_it(reverse):
    game = _position(((Color.RED, (37, 14)),), ((Color.RED, 37), (Color.BLUE, 15)))
    board = game.state.board
    edge = (15, 14) if reverse else (14, 15)
    assert (14, 15) in board.buildable_edges(Color.RED)
    board.build_road(Color.RED, edge)
    assert board.road_lengths[Color.RED] == 2
    before = dict(board.roads)
    assert (4, 15) not in board.buildable_edges(Color.RED)
    with pytest.raises(ValueError, match="Invalid Road Placement"):
        board.build_road(Color.RED, (15, 4) if reverse else (4, 15))
    assert board.roads == before


def test_enemy_endpoint_is_legal_when_other_endpoint_independently_connects():
    game = _position(
        ((Color.RED, (37, 14, 15)), (Color.RED, (0, 5, 4))),
        ((Color.RED, 37), (Color.RED, 0), (Color.BLUE, 15)),
    )
    board = game.state.board
    assert (4, 15) in board.buildable_edges(Color.RED)
    board.build_road(Color.RED, (4, 15))
    assert board.road_lengths[Color.RED] == 3
    assert len(board.connected_components[Color.RED]) == 2


@pytest.mark.parametrize("incumbent", [None, Color.RED, Color.BLUE])
def test_five_road_ties_keep_only_an_eligible_incumbent(incumbent):
    game = _position(
        ((Color.RED, (29, 30, 31, 32, 33, 34)),
         (Color.BLUE, (49, 50, 51, 52, 23, 6)))
    )
    board = game.state.board
    board.road_color = incumbent
    previous, holder, lengths = board.recompute_road_state()
    assert previous == incumbent
    assert holder == incumbent
    assert lengths[Color.RED] == lengths[Color.BLUE] == 5
    assert board.road_length == 5


def test_cut_revokes_award_when_two_challengers_tie_then_unique_leader_acquires():
    game = _position(
        ((Color.RED, (29, 30, 31, 32, 33, 34)),
         (Color.WHITE, (37, 38, 39, 17, 18, 40)),
         (Color.BLUE, (49, 50, 51, 52, 23, 6, 1, 2)))
    )
    state = game.state
    assert state.board.road_color == Color.BLUE
    result = state.board.build_settlement(Color.ORANGE, 23, initial_build_phase=True)
    build_settlement(state, Color.ORANGE, 23, is_free=True)
    maintain_longest_road(state, *result)
    assert result[:2] == (Color.BLUE, None)
    assert state.board.road_lengths[Color.BLUE] == 4
    assert all(not state.player_state[f"{player_key(state, color)}_HAS_ROAD"] for color in COLORS)
    assert game.observe(Color.BLUE).my_vp == 0
    result = state.board.build_road(Color.RED, (34, 35))
    build_road(state, Color.RED, (34, 35), is_free=True)
    maintain_longest_road(state, *result)
    assert result[:2] == (None, Color.RED)
    assert game.observe(Color.RED).my_vp == 2


def test_award_revocation_below_five_and_reacquisition_preserve_hidden_vp():
    game = _position(
        ((Color.RED, (29, 30, 31, 32, 33, 34)), (Color.BLUE, (12, 11, 32))),
        ((Color.RED, 29), (Color.BLUE, 12)),
    )
    state = game.state
    key = player_key(state, Color.RED)
    state.development_listdeck.remove(VICTORY_POINT)
    state.player_state[f"{key}_VICTORY_POINT_IN_HAND"] += 1
    state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += 1
    assert game.observe(Color.RED).my_vp == 3
    assert state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == 4

    result = state.board.build_settlement(Color.BLUE, 32)
    build_settlement(state, Color.BLUE, 32, is_free=True)
    maintain_longest_road(state, *result)
    maintain_longest_road(state, *result)
    assert result[:2] == (Color.RED, None)
    assert state.board.road_lengths[Color.RED] == 3
    assert game.observe(Color.RED).my_vp == 1
    assert state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == 2

    for edge, expected_length in (((29, 10), 4), ((10, 9), 5)):
        result = state.board.build_road(Color.RED, edge)
        build_road(state, Color.RED, edge, is_free=True)
        maintain_longest_road(state, *result)
        assert state.player_state[f"{key}_LONGEST_ROAD_LENGTH"] == expected_length
        assert state.board.road_color == (Color.RED if expected_length == 5 else None)
    assert game.observe(Color.RED).my_vp == 3
    assert state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] == 4


def test_recompute_replaces_stale_replay_components_lengths_and_edge_cache():
    game = _position(((Color.RED, (0, 1, 2, 9, 10)),), ((Color.RED, 0),))
    board = game.state.board
    board.buildable_edges(Color.RED)
    board.roads[10, 11] = board.roads[11, 10] = Color.RED
    board.connected_components[Color.RED] = [{0}]
    board.road_lengths[Color.RED] = 15
    previous, holder, lengths = board.recompute_road_state()
    assert previous is None and holder == Color.RED
    assert lengths[Color.RED] == 5
    assert (11, 12) in board.buildable_edges(Color.RED)
    branch = board.copy()
    assert branch.recompute_road_state() == board.recompute_road_state()
    assert branch.connected_components == board.connected_components
    branch.build_road(Color.RED, (11, 12))
    assert branch.road_lengths[Color.RED] == 6
    assert board.road_lengths[Color.RED] == 5
