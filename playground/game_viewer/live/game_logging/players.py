"""Spectator projections of player hands: resources, dev cards, and both paired."""

from typing import TypedDict

from cle.game_engine.state import GameState

__all__ = [
    "DevCardCounts",
    "PlayerHand",
    "get_player_dev_cards",
    "get_player_hands",
    "get_player_resources",
]


class DevCardCounts(TypedDict):
    """One player's development cards, split by whether they are still held."""

    in_hand: dict[str, int]
    played: dict[str, int]
    total_in_hand: int


class PlayerHand(TypedDict):
    """The spectator view of one player's hand: resources beside dev cards."""

    resources: dict[str, int]
    dev_cards: DevCardCounts


def get_player_resources(game_state: GameState) -> dict[str, dict[str, int]]:
    """Get resource counts for all players before an action."""
    resources: dict[str, dict[str, int]] = {}
    player_state = game_state.player_state
    colors = game_state.colors

    for idx, color in enumerate(colors):
        player_key = f"P{idx}"
        resources[color.value] = {
            "WOOD": player_state.get(f"{player_key}_WOOD_IN_HAND", 0),
            "BRICK": player_state.get(f"{player_key}_BRICK_IN_HAND", 0),
            "SHEEP": player_state.get(f"{player_key}_SHEEP_IN_HAND", 0),
            "WHEAT": player_state.get(f"{player_key}_WHEAT_IN_HAND", 0),
            "ORE": player_state.get(f"{player_key}_ORE_IN_HAND", 0)
        }

    return resources


def get_player_dev_cards(game_state: GameState) -> dict[str, DevCardCounts]:
    """Get development card counts for all players."""
    dev_cards: dict[str, DevCardCounts] = {}
    player_state = game_state.player_state
    colors = game_state.colors

    for idx, color in enumerate(colors):
        player_key = f"P{idx}"

        in_hand = {
            "KNIGHT": player_state.get(f"{player_key}_KNIGHT_IN_HAND", 0),
            "YEAR_OF_PLENTY": player_state.get(f"{player_key}_YEAR_OF_PLENTY_IN_HAND", 0),
            "MONOPOLY": player_state.get(f"{player_key}_MONOPOLY_IN_HAND", 0),
            "ROAD_BUILDING": player_state.get(f"{player_key}_ROAD_BUILDING_IN_HAND", 0),
            "VICTORY_POINT": player_state.get(f"{player_key}_VICTORY_POINT_IN_HAND", 0),
        }

        played = {
            "KNIGHT": player_state.get(f"{player_key}_PLAYED_KNIGHT", 0),
            "YEAR_OF_PLENTY": player_state.get(f"{player_key}_PLAYED_YEAR_OF_PLENTY", 0),
            "MONOPOLY": player_state.get(f"{player_key}_PLAYED_MONOPOLY", 0),
            "ROAD_BUILDING": player_state.get(f"{player_key}_PLAYED_ROAD_BUILDING", 0),
            "VICTORY_POINT": player_state.get(f"{player_key}_PLAYED_VICTORY_POINT", 0),
        }

        total_in_hand = sum(in_hand.values())

        dev_cards[color.value] = {
            "in_hand": in_hand,
            "played": played,
            "total_in_hand": total_in_hand,
        }

    return dev_cards


def get_player_hands(game_state: GameState) -> dict[str, PlayerHand]:
    """Spectator hand contents: the exact resource and dev-card breakdown.

    The viewer's public projection collapses hands to totals, so this is the
    parallel field the playground reveals when the operator asks to see
    contents. It is a viewer artifact only - model prompts are built from the
    engine's privacy projection, never from a viewer snapshot.
    """
    resources = get_player_resources(game_state)
    dev_cards = get_player_dev_cards(game_state)

    return {
        color: {
            "resources": counts,
            "dev_cards": dev_cards[color],
        }
        for color, counts in resources.items()
    }
