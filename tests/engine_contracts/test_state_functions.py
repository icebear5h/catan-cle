"""Observable helper contracts across the state_functions package move."""

import pickle
import random
import subprocess
import sys

import pytest

from cle.game_engine import state_functions as helpers
from cle.game_engine.models.enums import CITY, ROAD, SETTLEMENT, FastDevCard, FastResource
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState


def fresh_state() -> GameState:
    return GameState(
        (Color.RED, Color.BLUE), shuffle_players=False, rng=random.Random(123),
    )


def test_filter_is_lazy_and_single_use() -> None:
    colors = [Color.RED, Color.BLUE]
    enemies = helpers.get_enemy_colors(colors, Color.RED)
    assert isinstance(enemies, filter)
    colors.append(Color.WHITE)
    assert next(enemies) is Color.BLUE
    assert list(enemies) == [Color.WHITE]
    assert list(enemies) == []


def test_building_cache_defaults_identity_and_city_upgrade() -> None:
    state = fresh_state()
    for color in (Color.RED, Color.WHITE):
        first = helpers.get_player_buildings(state, color, CITY)
        second = helpers.get_player_buildings(state, color, CITY)
        assert first == second == [] and first is not second
    assert Color.WHITE not in state.buildings_by_color
    assert CITY not in state.buildings_by_color[Color.RED]
    helpers.player_freqdeck_add(state, Color.RED, (1, 1, 0, 2, 3))
    helpers.build_settlement(state, Color.RED, 0, is_free=True)
    settlements = helpers.get_player_buildings(state, Color.RED, SETTLEMENT)
    assert settlements is state.buildings_by_color[Color.RED][SETTLEMENT]
    helpers.build_road(state, Color.RED, (0, 1), is_free=False)
    roads = helpers.get_player_buildings(state, Color.RED, ROAD)
    assert roads is state.buildings_by_color[Color.RED][ROAD]
    assert roads == [(0, 1)] and state.resource_freqdeck == [20, 20, 19, 19, 19]
    helpers.build_city(state, Color.RED, 0)
    assert settlements == []
    assert helpers.get_player_buildings(state, Color.RED, CITY) == [0]
    assert helpers.get_player_freqdeck(state, Color.RED) == [0, 0, 0, 0, 0]
    assert helpers.get_visible_victory_points(state, Color.RED) == 2


def test_development_eligibility_returns_stored_scalar_without_coercion() -> None:
    state = fresh_state()
    state.player_state["P0_KNIGHT_IN_HAND"] = 3
    state.player_state["P0_KNIGHT_OWNED_AT_START"] = 2
    result = helpers.player_can_play_dev(state, Color.RED, "KNIGHT")
    assert type(result) is int and result == 2
    state.player_state["P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN"] = True
    assert helpers.player_can_play_dev(state, Color.RED, "KNIGHT") is False
    helpers.player_clean_turn(state, Color.RED)
    assert helpers.player_can_play_dev(state, Color.RED, "KNIGHT") is True
    assert helpers.player_has_rolled(state, Color.RED) is False
    state.player_state["P0_KNIGHT_IN_HAND"] = 0
    assert helpers.player_can_play_dev(state, Color.RED, "KNIGHT") is False


def test_random_draw_keeps_resource_order_consumption_and_empty_failure() -> None:
    state = fresh_state()
    helpers.player_freqdeck_add(state, Color.RED, (2, 1, 1, 1, 2))
    expected: list[FastResource] = ["WOOD", "WOOD", "BRICK", "SHEEP", "WHEAT", "ORE", "ORE"]
    assert helpers.player_deck_to_array(state, Color.RED) == expected
    oracle = random.Random()
    oracle.setstate(state.rng.getstate())
    while expected:
        selected = oracle.choice(expected)
        expected.remove(selected)
        assert helpers.player_deck_random_draw(state, Color.RED) == selected
        assert helpers.player_deck_to_array(state, Color.RED) == expected
        assert state.rng.getstate() == oracle.getstate()
    with pytest.raises(IndexError):
        helpers.player_deck_random_draw(state, Color.RED)
    assert state.rng.getstate() == oracle.getstate()


def test_random_draw_calls_replaced_public_draw_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    state = fresh_state()
    helpers.player_deck_replenish(state, Color.RED, "WOOD")
    real_draw = helpers.player_deck_draw
    random_draw = helpers.player_deck_random_draw
    calls: list[tuple[Color, FastResource | FastDevCard, int]] = []

    def record_draw(
        target: GameState, color: Color, card: FastResource | FastDevCard, amount: int = 1,
    ) -> None:
        assert target is state
        calls.append((color, card, amount))
        real_draw(target, color, card, amount)

    monkeypatch.setattr("cle.game_engine.state_functions.player_deck_draw", record_draw)
    assert helpers.player_deck_random_draw is random_draw
    assert random_draw(state, Color.RED) == "WOOD"
    assert calls == [(Color.RED, "WOOD", 1)]
    assert helpers.get_player_freqdeck(state, Color.RED) == [0, 0, 0, 0, 0]


def test_knight_award_threshold_tie_transfer_and_hidden_points() -> None:
    state = fresh_state()
    assert helpers.get_largest_army(state) == (None, None)
    state.player_state["P0_ACTUAL_VICTORY_POINTS"] = 1
    state.player_state["P0_VICTORY_POINT_IN_HAND"] = 1
    state.player_state["P0_KNIGHT_IN_HAND"] = 3
    state.player_state["P1_KNIGHT_IN_HAND"] = 4
    for _ in range(2):
        helpers.play_dev_card(state, Color.RED, "KNIGHT")
        assert helpers.get_largest_army(state) == (None, None)
    helpers.play_dev_card(state, Color.RED, "KNIGHT")
    assert helpers.get_largest_army(state) == (Color.RED, 3)
    for _ in range(3):
        helpers.play_dev_card(state, Color.BLUE, "KNIGHT")
    assert helpers.get_largest_army(state) == (Color.RED, 3)
    assert helpers.get_actual_victory_points(state, Color.RED) == 3
    helpers.play_dev_card(state, Color.BLUE, "KNIGHT")
    assert helpers.get_largest_army(state) == (Color.BLUE, 4)
    assert helpers.get_visible_victory_points(state, Color.RED) == 0
    assert helpers.get_actual_victory_points(state, Color.RED) == 1
    assert helpers.get_actual_victory_points(state, Color.BLUE) == 2


@pytest.mark.parametrize("first", [
    "cle.game_engine.state", "cle.game_engine.state_functions", "cle.game_engine.models.actions",
])
def test_import_orders_and_historical_function_reference(first: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", f"import {first}; from cle.game_engine.state import GameState"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert pickle.loads(b"ccle.game_engine.state_functions\nplayer_key\n.") is helpers.player_key
