"""State readers and mutators, reexported at their historical public path."""

from cle.game_engine.models.decks import ROAD_COST_FREQDECK, freqdeck_add

# Implementations resolve cross-helper calls through this module at call time,
# preserving replacement of the historical module globals without wrapping exports.
from .awards import (
    maintain_largest_army,
    maintain_longest_road,
)
from .buildings import (
    build_city,
    build_road,
    build_settlement,
)
from .cards import (
    buy_dev_card,
    play_dev_card,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_deck_to_array,
    player_freqdeck_add,
    player_freqdeck_subtract,
    player_num_dev_cards,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)
from .getters import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_enemy_colors,
    get_largest_army,
    get_longest_road_color,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_buildings,
    get_player_freqdeck,
    get_state_index,
    get_visible_victory_points,
    player_has_rolled,
    player_key,
)

__all__ = [
    "ROAD_COST_FREQDECK",
    "build_city",
    "build_road",
    "build_settlement",
    "buy_dev_card",
    "freqdeck_add",
    "get_actual_victory_points",
    "get_dev_cards_in_hand",
    "get_enemy_colors",
    "get_largest_army",
    "get_longest_road_color",
    "get_longest_road_length",
    "get_played_dev_cards",
    "get_player_buildings",
    "get_player_freqdeck",
    "get_state_index",
    "get_visible_victory_points",
    "maintain_largest_army",
    "maintain_longest_road",
    "play_dev_card",
    "player_can_afford_dev_card",
    "player_can_play_dev",
    "player_clean_turn",
    "player_deck_draw",
    "player_deck_random_draw",
    "player_deck_replenish",
    "player_deck_to_array",
    "player_freqdeck_add",
    "player_freqdeck_subtract",
    "player_has_rolled",
    "player_key",
    "player_num_dev_cards",
    "player_num_resource_cards",
    "player_resource_freqdeck_contains",
]
