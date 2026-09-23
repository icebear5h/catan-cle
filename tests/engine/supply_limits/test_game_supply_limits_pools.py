"""Development-card pool and resource-bank supply limits."""

from collections import Counter

import pytest

from cle.game_engine.models.actions import (
    generate_playable_actions,
)
from cle.game_engine.models.decks import (
    DEVELOPMENT_CARD_COST_FREQDECK,
    DEVELOPMENT_CARD_COUNTS,
    DEVELOPMENT_CARDS_PER_DECK,
    RESOURCE_CARDS_PER_TYPE,
    freqdeck_draw,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_resource_bank,
)
from cle.game_engine.models.enums import (
    KNIGHT,
    MONOPOLY,
    RESOURCES,
    WOOD,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    CITIES_PER_PLAYER,
    ROADS_PER_PLAYER,
    SETTLEMENTS_PER_PLAYER,
)
from cle.game_engine.state_functions import player_key

from .support import (
    COLORS,
    assert_resource_conservation,
    make_game,
    piece_count,
    transfer_from_bank_to_player,
)


def test_standard_four_player_game_starts_with_official_supplies() -> None:
    game = make_game()

    assert starting_resource_bank() == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)
    assert game.state.resource_freqdeck == [RESOURCE_CARDS_PER_TYPE] * len(RESOURCES)
    for color in COLORS:
        assert piece_count(game, color, "ROADS") == ROADS_PER_PLAYER == 15
        assert piece_count(game, color, "SETTLEMENTS") == SETTLEMENTS_PER_PLAYER == 5
        assert piece_count(game, color, "CITIES") == CITIES_PER_PLAYER == 4


def test_standard_development_card_pool_has_official_25_card_mix() -> None:
    game = make_game()

    assert DEVELOPMENT_CARD_COUNTS == {
        "KNIGHT": 14,
        "YEAR_OF_PLENTY": 2,
        "ROAD_BUILDING": 2,
        "MONOPOLY": 2,
        "VICTORY_POINT": 5,
    }
    assert DEVELOPMENT_CARDS_PER_DECK == 25
    assert Counter(starting_devcard_bank()) == Counter(DEVELOPMENT_CARD_COUNTS)
    assert Counter(game.state.development_listdeck) == Counter(DEVELOPMENT_CARD_COUNTS)


def test_development_card_purchase_depletes_pool_and_play_does_not_recycle() -> None:
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.development_listdeck = [KNIGHT]
    state.is_initial_build_phase = False
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.player_state[f"{key}_HAS_ROLLED"] = True
    transfer_from_bank_to_player(
        game,
        Color.RED,
        DEVELOPMENT_CARD_COST_FREQDECK,
    )
    state.playable_actions = generate_playable_actions(state)
    buy_action = Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None)

    assert buy_action in state.playable_actions
    transition = game.step(buy_action)

    assert transition.resolved_action == Action(
        Color.RED,
        ActionType.BUY_DEVELOPMENT_CARD,
        KNIGHT,
    )
    assert state.development_listdeck == []
    assert state.player_state[f"{key}_KNIGHT_IN_HAND"] == 1
    assert not any(
        action.action_type == ActionType.BUY_DEVELOPMENT_CARD for action in state.playable_actions
    )
    assert_resource_conservation(game)

    state.player_state[f"{key}_KNIGHT_OWNED_AT_START"] = True
    state.playable_actions = generate_playable_actions(state)
    game.step(Action(Color.RED, ActionType.PLAY_KNIGHT_CARD, None))

    assert state.development_listdeck == []
    assert state.player_state[f"{key}_KNIGHT_IN_HAND"] == 0
    assert state.player_state[f"{key}_PLAYED_KNIGHT"] == 1


def test_explicit_development_card_draw_cannot_overdraw_card_type_or_empty_pool() -> None:
    game = make_game()
    state = game.state
    key = player_key(state, Color.RED)
    state.development_listdeck = [KNIGHT]
    transfer_from_bank_to_player(
        game,
        Color.RED,
        DEVELOPMENT_CARD_COST_FREQDECK,
    )
    bank_before = list(state.resource_freqdeck)
    hand_before = [state.player_state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES]

    with pytest.raises(ValueError, match="Not enough MONOPOLY cards available"):
        game.step(
            Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, MONOPOLY),
            force=True,
        )

    assert state.development_listdeck == [KNIGHT]
    assert state.resource_freqdeck == bank_before
    assert [
        state.player_state[f"{key}_{resource}_IN_HAND"] for resource in RESOURCES
    ] == hand_before
    assert game.revision == 0

    state.development_listdeck = []
    with pytest.raises(ValueError, match="No more development cards"):
        game.step(
            Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, KNIGHT),
            force=True,
        )

    assert state.resource_freqdeck == bank_before
    assert game.revision == 0


def test_resource_bank_frequency_deck_cannot_be_overdrawn() -> None:
    bank = starting_resource_bank()

    freqdeck_draw(bank, RESOURCE_CARDS_PER_TYPE, WOOD)
    assert bank[RESOURCES.index(WOOD)] == 0

    with pytest.raises(ValueError, match="Not enough WOOD"):
        freqdeck_draw(bank, 1, WOOD)
    with pytest.raises(ValueError, match="unavailable cards"):
        freqdeck_subtract(bank, [1, 0, 0, 0, 0])

    assert bank[RESOURCES.index(WOOD)] == 0
