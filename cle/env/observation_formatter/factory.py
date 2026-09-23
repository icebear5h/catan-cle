"""Legacy shallow observation factory, separate from engine.observe_state."""

from __future__ import annotations

from cle.env import observation_formatter
from cle.game_engine.models.enums import CITY, RESOURCES, ROAD, SETTLEMENT, Action
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.game_engine.state_functions import (
    get_dev_cards_in_hand,
    get_longest_road_length,
    get_player_buildings,
    get_player_freqdeck,
    get_visible_victory_points,
    player_key,
)


def create_observation_from_state(
    game_state: GameState,
    player_color: Color,
    recent_events: list[Action] | None = None,
) -> observation_formatter.CatanObservation:
    """Convert state without changing legacy defaults, aliases, or menu visibility."""
    resource_freqdeck = get_player_freqdeck(game_state, player_color)
    my_resources = {
        str(RESOURCES[i]): count for i, count in enumerate(resource_freqdeck)
    }
    my_dev_cards = {
        'KNIGHT': get_dev_cards_in_hand(game_state, player_color, 'KNIGHT'),
        'VICTORY_POINT': get_dev_cards_in_hand(game_state, player_color, 'VICTORY_POINT'),
        'ROAD_BUILDING': get_dev_cards_in_hand(game_state, player_color, 'ROAD_BUILDING'),
        'MONOPOLY': get_dev_cards_in_hand(game_state, player_color, 'MONOPOLY'),
        'YEAR_OF_PLENTY': get_dev_cards_in_hand(game_state, player_color, 'YEAR_OF_PLENTY'),
    }
    my_settlements = get_player_buildings(game_state, player_color, SETTLEMENT)
    my_cities = get_player_buildings(game_state, player_color, CITY)
    my_roads = get_player_buildings(game_state, player_color, ROAD)
    opponent_colors = [c for c in game_state.colors if c != player_color]
    opponent_settlements = {
        color: get_player_buildings(game_state, color, SETTLEMENT)
        for color in opponent_colors
    }
    opponent_cities = {
        color: get_player_buildings(game_state, color, CITY)
        for color in opponent_colors
    }
    opponent_roads = {
        color: get_player_buildings(game_state, color, ROAD)
        for color in opponent_colors
    }
    opponent_vps = {
        color: get_visible_victory_points(game_state, color) for color in opponent_colors
    }
    opponent_resource_counts = {
        color: sum(get_player_freqdeck(game_state, color)) for color in opponent_colors
    }
    opponent_dev_card_counts = {
        color: get_dev_cards_in_hand(game_state, color) for color in opponent_colors
    }
    if game_state.is_initial_build_phase:
        phase = "initial_placement"
    elif game_state.is_discarding:
        phase = "discarding"
    elif game_state.is_moving_knight:
        phase = "moving_robber"
    else:
        phase = "main_game"
    longest_road_holder = game_state.board.road_color
    largest_army_holder = None
    for color in game_state.colors:
        key = player_key(game_state, color)
        if game_state.player_state.get(f"{key}_HAS_ARMY"):
            largest_army_holder = color
            break
    turn_player_color = game_state.colors[game_state.current_turn_index]
    is_my_turn = player_color == turn_player_color
    return observation_formatter.CatanObservation(
        my_color=player_color,
        my_settlements=my_settlements,
        my_cities=my_cities,
        my_roads=my_roads,
        opponent_settlements=opponent_settlements,
        opponent_cities=opponent_cities,
        opponent_roads=opponent_roads,
        my_resources=my_resources,
        my_dev_cards=my_dev_cards,
        opponent_resource_counts=opponent_resource_counts,
        opponent_dev_card_counts=opponent_dev_card_counts,
        current_turn=game_state.num_turns,
        current_phase=phase,
        turn_order=tuple(game_state.colors),
        last_dice_roll=None,
        robber_position=game_state.board.robber_coordinate,
        my_vp=get_visible_victory_points(game_state, player_color),
        opponent_vps=opponent_vps,
        longest_road_holder=longest_road_holder,
        largest_army_holder=largest_army_holder,
        my_longest_road_length=get_longest_road_length(game_state, player_color),
        valid_actions=game_state.playable_actions,
        board_map=game_state.board.map,
        buildings_dict=game_state.board.buildings,
        trade_window=getattr(game_state, "trade_window", None),
        is_my_turn=is_my_turn,
        turn_player_color=turn_player_color,
        recent_events=recent_events if recent_events is not None else [],
    )
