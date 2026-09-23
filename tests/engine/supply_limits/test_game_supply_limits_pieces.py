"""Building-piece supply exhaustion and return."""

import pytest

from cle.game_engine.models.actions import (
    city_possibilities,
    generate_playable_actions,
    initial_road_possibilities,
    road_building_possibilities,
    settlement_possibilities,
)
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    RESOURCE_CARDS_PER_TYPE,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    SETTLEMENT,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_key

from .support import (
    assert_resource_conservation,
    make_game,
    piece_count,
    transfer_from_bank_to_player,
)


def test_exhausted_piece_supplies_generate_no_build_actions() -> None:
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] = 0
    state.player_state[f"{key}_ROADS_AVAILABLE"] = 0
    state.player_state[f"{key}_CITIES_AVAILABLE"] = 0

    assert settlement_possibilities(state, Color.RED, initial_build_phase=True) == []
    assert road_building_possibilities(state, Color.RED, check_money=False) == []
    assert initial_road_possibilities(state, Color.RED) == []
    assert city_possibilities(state, Color.RED) == []
    assert generate_playable_actions(state) == []


def test_exhausted_settlement_supply_is_rejected_before_board_mutation() -> None:
    game = make_game()
    action = game.state.playable_actions[0]
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_SETTLEMENTS_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No settlement pieces available"):
        game.step(action)

    assert action.value not in game.state.board.buildings
    assert game.revision == 0


def test_exhausted_road_supply_is_rejected_before_board_mutation() -> None:
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    road = game.state.playable_actions[0]
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_ROADS_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No road pieces available"):
        game.step(road)

    assert road.value not in game.state.board.roads
    assert (road.value[1], road.value[0]) not in game.state.board.roads
    assert game.revision == 1


def test_exhausted_city_supply_is_rejected_without_consuming_settlement() -> None:
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_CITIES_AVAILABLE"] = 0

    with pytest.raises(ValueError, match="No city pieces available"):
        game.step(
            Action(Color.RED, ActionType.BUILD_CITY, node_id),
            force=True,
        )

    assert game.state.board.buildings[node_id] == (Color.RED, SETTLEMENT)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 4
    assert piece_count(game, Color.RED, "CITIES") == 0
    assert game.revision == 1


def test_unaffordable_forced_city_cannot_create_cards_in_bank() -> None:
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    bank_before = list(game.state.resource_freqdeck)

    with pytest.raises(ValueError, match="cannot afford to build a city"):
        game.step(
            Action(Color.RED, ActionType.BUILD_CITY, node_id),
            force=True,
        )

    assert game.state.board.buildings[node_id] == (Color.RED, SETTLEMENT)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 4
    assert piece_count(game, Color.RED, "CITIES") == 4
    assert game.state.resource_freqdeck == bank_before
    assert game.revision == 1


def test_road_building_stops_after_last_available_road_piece() -> None:
    game = make_game()
    game.step(game.state.playable_actions[0])
    state = game.state
    key = player_key(state, Color.RED)
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.is_road_building = True
    state.free_roads_available = 2
    state.player_state[f"{key}_ROADS_AVAILABLE"] = 1
    state.playable_actions = generate_playable_actions(state)
    road = state.playable_actions[0]
    bank_before = list(state.resource_freqdeck)

    game.step(road)

    assert piece_count(game, Color.RED, "ROADS") == 0
    assert state.is_road_building is False
    assert state.free_roads_available == 0
    assert not any(action.action_type == ActionType.BUILD_ROAD for action in state.playable_actions)
    assert state.resource_freqdeck == bank_before


def test_city_upgrade_returns_settlement_piece_and_replenishes_bank() -> None:
    game = make_game()
    settlement = game.state.playable_actions[0]
    game.step(settlement)
    node_id = settlement.value
    transfer_from_bank_to_player(game, Color.RED, CITY_COST_FREQDECK)
    key = player_key(game.state, Color.RED)
    game.state.player_state[f"{key}_HAS_ROLLED"] = True
    game.state.current_prompt = ActionPrompt.PLAY_TURN
    game.state.playable_actions = generate_playable_actions(game.state)
    city_action = Action(Color.RED, ActionType.BUILD_CITY, node_id)

    assert city_action in game.state.playable_actions
    game.step(city_action)

    assert game.state.board.buildings[node_id] == (Color.RED, CITY)
    assert piece_count(game, Color.RED, "SETTLEMENTS") == 5
    assert piece_count(game, Color.RED, "CITIES") == 3
    assert len(game.state.buildings_by_color[Color.RED][SETTLEMENT]) == 0
    assert game.state.buildings_by_color[Color.RED][CITY] == [node_id]
    assert_resource_conservation(game)
    assert game.state.resource_freqdeck == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)
