"""Road connectivity and Longest Road award evidence."""
from typing import Any

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
)
from cle.game_engine.models.enums import (
    SETTLEMENT,
    Action,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    get_player_freqdeck,
)

from .support import (
    COLORS,
    _assert_inventory,
    _assert_road_scores,
    _expect_rule,
    _fund,
    _position,
    _turn,
)


@pytest.mark.parametrize("surface", ["menu", "step"])
def test_road_cannot_extend_through_enemy_settlement(surface: str) -> None:
    game: Any = _position(
        ((Color.RED, 37), (Color.BLUE, 18)),
        ((Color.RED, (37, 14, 15)), (Color.BLUE, (18, 17, 15))),
    )
    _turn(game, Color.BLUE)
    _fund(game, Color.BLUE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 15))
    _turn(game, Color.RED)
    _fund(game, Color.RED, ROAD_COST_FREQDECK)
    _assert_inventory(game)
    blocked = Action(Color.RED, ActionType.BUILD_ROAD, (4, 15))
    assert game.state.board.buildings[15] == (Color.BLUE, SETTLEMENT)
    assert 4 not in game.state.board.buildings
    assert game.state.board.roads[(14, 15)] == Color.RED
    assert not any(
        4 in edge and owner == Color.RED for edge, owner in game.state.board.roads.items()
    )
    assert get_player_freqdeck(game.state, Color.RED) == ROAD_COST_FREQDECK
    if surface == "menu":
        observed = {
            "in_menu": blocked in game.state.playable_actions,
            "is_valid": game.is_action_valid(blocked),
        }
        _assert_inventory(game)
        _expect_rule(observed, {"in_menu": False, "is_valid": False})
    else:
        revision = game.revision
        roads_before = dict(game.state.board.roads)
        rejected = False
        try:
            game.step(blocked)
        except ValueError:
            rejected = True
        _assert_inventory(game)
        observed = {
            "rejected": rejected,
            "revision_delta": game.revision - revision,
            "roads_unchanged": game.state.board.roads == roads_before,
        }
        _expect_rule(observed, {"rejected": True, "revision_delta": 0, "roads_unchanged": True})


def test_longest_road_requires_five_after_a_cut() -> None:
    game = _position(
        ((Color.RED, 12), (Color.BLUE, 2)),
        ((Color.RED, (12, 11, 10, 29)), (Color.BLUE, (2, 9, 10))),
    )
    assert game.state.board.road_color is None
    _turn(game, Color.BLUE)
    _fund(game, Color.BLUE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.BLUE, ActionType.BUILD_SETTLEMENT, 10))
    _assert_inventory(game)
    assert game.state.board.buildings[10] == (Color.BLUE, SETTLEMENT)
    observed = {
        "holder": game.state.board.road_color,
        "red_vp": game.observe(Color.RED).my_vp,
        "blue_vp": game.observe(Color.BLUE).my_vp,
    }
    _expect_rule(observed, {"holder": None, "red_vp": 1, "blue_vp": 2})


def test_longest_road_holder_keeps_a_six_road_tie_after_cut() -> None:
    game = _position(
        ((Color.RED, 29), (Color.BLUE, 49), (Color.WHITE, 20)),
        (
            (Color.RED, (29, 30, 31, 32, 33, 34, 35)),
            (Color.BLUE, (49, 50, 51, 52, 23, 6, 1, 2)),
            (Color.WHITE, (20, 0, 1)),
        ),
    )
    assert game.state.board.road_color == Color.BLUE
    _turn(game, Color.WHITE)
    _fund(game, Color.WHITE, SETTLEMENT_COST_FREQDECK)
    game.step(Action(Color.WHITE, ActionType.BUILD_SETTLEMENT, 1))
    _assert_inventory(game)
    assert game.state.board.buildings[1] == (Color.WHITE, SETTLEMENT)
    # BLUE retains the six-edge chain 49 -> 50 -> 51 -> 52 -> 23 -> 6 -> 1.
    observed = {
        "holder": game.state.board.road_color,
        "blue_vp": game.observe(Color.BLUE).my_vp,
        "red_vp": game.observe(Color.RED).my_vp,
    }
    _expect_rule(observed, {"holder": Color.BLUE, "blue_vp": 3, "red_vp": 1})


def test_initial_road_is_visible_in_own_road_length() -> None:
    game = GameEngine(COLORS, seed=7, shuffle_players=False)
    game.step(game.state.playable_actions[0])
    game.step(game.state.playable_actions[0])
    _assert_inventory(game)
    observation = game.observe(Color.RED)
    assert len(observation.my_roads) == 1
    _expect_rule(observation.my_longest_road_length, 1)


def test_ten_points_from_off_turn_road_transfer_must_wait() -> None:
    cities = (29, 34, 17, 43)
    game = _position(
        tuple((Color.RED, node) for node in cities) + ((Color.BLUE, 49), (Color.WHITE, 25)),
        (
            (Color.RED, (29, 30, 31, 32, 33, 34, 35)),
            (Color.BLUE, (49, 50, 51, 52, 23, 6, 1, 2)),
            (Color.WHITE, (25, 24, 7, 6)),
        ),
    )
    for node in cities:
        _fund(game, Color.RED, CITY_COST_FREQDECK)
        game.step(Action(Color.RED, ActionType.BUILD_CITY, node))
    assert game.observe(Color.RED).my_vp == 8
    _turn(game, Color.WHITE)
    _fund(game, Color.WHITE, SETTLEMENT_COST_FREQDECK)
    transition = game.step(Action(Color.WHITE, ActionType.BUILD_SETTLEMENT, 6))
    _assert_inventory(game)
    assert game.state.board.road_color == Color.RED
    assert game.observe(Color.RED).my_vp == 10
    assert game.state.colors[game.state.current_turn_index] == Color.WHITE
    observed = {
        "transition_winner": transition.winner,
        "immediate_winner": game.winning_color(),
    }
    _turn(game, Color.RED, rolled=False)
    _assert_inventory(game)
    assert game.state.colors[game.state.current_turn_index] == Color.RED
    observed["own_turn_winner"] = game.winning_color()
    _expect_rule(
        observed,
        {
            "transition_winner": None,
            "immediate_winner": None,
            "own_turn_winner": Color.RED,
        },
    )


@pytest.mark.parametrize(
    "paths,length",
    [
        (((0, 1, 2, 3, 4, 5, 0),), 6),
        (((1, 0, 20), (1, 2, 9), (1, 6, 23)), 4),
        (((1, 0, 5, 4, 3, 2), (1, 6, 7, 8, 9, 2), (1, 2)), 11),
    ],
    ids=["loop", "branch", "theta"],
)
def test_independent_road_oracle_checks_known_trails(
    paths: tuple[tuple[int, ...], ...], length: int
) -> None:
    game = _position(
        ((Color.RED, paths[0][0]),),
        tuple((Color.RED, path) for path in paths),
    )
    holder = Color.RED if length >= 5 else None
    assert _assert_road_scores(game, holder)[Color.RED] == length
    game.state.board.road_lengths[Color.RED] += 1
    with pytest.raises(AssertionError):
        _assert_road_scores(game, holder)
