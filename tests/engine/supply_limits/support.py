"""Shared helpers for official supply limits for cards, pieces, and the resource bank."""

from cle.game_engine.game import GameEngine
from cle.game_engine.models.decks import (
    RESOURCE_CARDS_PER_TYPE,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import (
    RESOURCES,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import player_key

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def make_game() -> GameEngine:
    return GameEngine(COLORS, seed=1, shuffle_players=False)


def piece_count(game: GameEngine, color: Color, piece: str) -> int:
    key = player_key(game.state, color)
    return game.state.player_state[f"{key}_{piece}_AVAILABLE"]


def transfer_from_bank_to_player(
    game: GameEngine,
    color: Color,
    cards: list[int],
) -> None:
    state = game.state
    key = player_key(state, color)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards)
    for resource, count in zip(RESOURCES, cards):
        state.player_state[f"{key}_{resource}_IN_HAND"] += count


def assert_resource_conservation(game: GameEngine) -> None:
    state = game.state
    for resource_index, resource in enumerate(RESOURCES):
        cards_in_hands = sum(
            state.player_state[f"{player_key(state, color)}_{resource}_IN_HAND"] for color in COLORS
        )
        assert state.resource_freqdeck[resource_index] + cards_in_hands == RESOURCE_CARDS_PER_TYPE
