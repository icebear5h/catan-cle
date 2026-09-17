"""Perspective-safe structured observations produced by the game engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from cle.game_engine.models.enums import Action, CITY, RESOURCES, ROAD, SETTLEMENT
from cle.game_engine.models.player import Color
from cle.game_engine.state_functions import (
    get_actual_victory_points,
    get_dev_cards_in_hand,
    get_longest_road_length,
    get_player_buildings,
    get_player_freqdeck,
    get_visible_victory_points,
    player_has_rolled,
    player_key,
)


@dataclass(slots=True)
class PlayerObservation:
    my_color: Color
    my_settlements: list[int]
    my_cities: list[int]
    my_roads: list[tuple]
    opponent_settlements: dict[Color, list[int]]
    opponent_cities: dict[Color, list[int]]
    opponent_roads: dict[Color, list[tuple]]
    my_resources: dict[str, int]
    my_dev_cards: dict[str, int]
    opponent_resource_counts: dict[Color, int]
    opponent_dev_card_counts: dict[Color, int]
    current_turn: int
    current_phase: str
    turn_order: tuple[Color, ...]
    last_dice_roll: Any
    robber_position: Any
    my_vp: int
    opponent_vps: dict[Color, int]
    longest_road_holder: Color | None
    largest_army_holder: Color | None
    my_longest_road_length: int
    valid_actions: list[Action]
    board_map: Any
    buildings_dict: dict[int, tuple]
    trade_window: Any
    is_my_turn: bool
    turn_player_color: Color
    recent_events: list[Action] = field(default_factory=list)
    my_actual_vp: int | None = None
    current_prompt: str = ""
    setup_road_anchor: int | None = None
    free_roads_available: int = 0
    turn_player_has_rolled: bool | None = None

    def __setstate__(self, state) -> None:
        """Keep older slotted observation pickles readable after additive facts."""
        _, stored = state
        values = {
            "my_actual_vp": None,
            "current_prompt": "",
            "setup_road_anchor": None,
            "free_roads_available": 0,
            "turn_player_has_rolled": None,
            **stored,
        }
        for name, value in values.items():
            setattr(self, name, value)


def observe_state(
    game_state,
    player_color: Color,
    recent_events: list[Action] | None = None,
) -> PlayerObservation:
    """Return exactly the structured facts visible to one participant."""
    if player_color not in game_state.colors:
        raise ValueError(f"Color {player_color} is not a participant")

    resource_freqdeck = get_player_freqdeck(game_state, player_color)
    my_resources = {
        str(RESOURCES[index]): count
        for index, count in enumerate(resource_freqdeck)
    }
    my_dev_cards = {
        card: get_dev_cards_in_hand(game_state, player_color, card)
        for card in (
            "KNIGHT",
            "VICTORY_POINT",
            "ROAD_BUILDING",
            "MONOPOLY",
            "YEAR_OF_PLENTY",
        )
    }
    opponent_colors = [color for color in game_state.colors if color != player_color]

    if game_state.is_initial_build_phase:
        phase = "initial_placement"
    elif game_state.is_discarding:
        phase = "discarding"
    elif game_state.is_moving_knight:
        phase = "moving_robber"
    else:
        phase = "main_game"

    largest_army_holder = next(
        (
            color
            for color in game_state.colors
            if game_state.player_state.get(f"{player_key(game_state, color)}_HAS_ARMY")
        ),
        None,
    )
    turn_player_color = game_state.colors[game_state.current_turn_index]
    return deepcopy(PlayerObservation(
        my_color=player_color,
        my_settlements=get_player_buildings(game_state, player_color, SETTLEMENT),
        my_cities=get_player_buildings(game_state, player_color, CITY),
        my_roads=get_player_buildings(game_state, player_color, ROAD),
        opponent_settlements={
            color: get_player_buildings(game_state, color, SETTLEMENT)
            for color in opponent_colors
        },
        opponent_cities={
            color: get_player_buildings(game_state, color, CITY)
            for color in opponent_colors
        },
        opponent_roads={
            color: get_player_buildings(game_state, color, ROAD)
            for color in opponent_colors
        },
        my_resources=my_resources,
        my_dev_cards=my_dev_cards,
        opponent_resource_counts={
            color: sum(get_player_freqdeck(game_state, color))
            for color in opponent_colors
        },
        opponent_dev_card_counts={
            color: get_dev_cards_in_hand(game_state, color)
            for color in opponent_colors
        },
        current_turn=game_state.num_turns,
        current_phase=phase,
        turn_order=tuple(game_state.colors),
        last_dice_roll=game_state.last_dice_roll,
        robber_position=game_state.board.robber_coordinate,
        my_vp=get_visible_victory_points(game_state, player_color),
        opponent_vps={
            color: get_visible_victory_points(game_state, color)
            for color in opponent_colors
        },
        longest_road_holder=game_state.board.road_color,
        largest_army_holder=largest_army_holder,
        my_longest_road_length=get_longest_road_length(game_state, player_color),
        valid_actions=(
            list(game_state.playable_actions)
            if player_color == game_state.current_color()
            else []
        ),
        board_map=game_state.board.map,
        buildings_dict=game_state.board.buildings,
        trade_window=game_state.trade_window,
        is_my_turn=player_color == turn_player_color,
        turn_player_color=turn_player_color,
        recent_events=list(recent_events or ()),
        my_actual_vp=get_actual_victory_points(game_state, player_color),
        current_prompt=game_state.current_prompt.value,
        setup_road_anchor=(
            game_state.buildings_by_color[game_state.current_color()][SETTLEMENT][-1]
            if game_state.is_initial_build_phase
            and game_state.current_prompt.value == "BUILD_INITIAL_ROAD" else None
        ),
        free_roads_available=game_state.free_roads_available if game_state.is_road_building else 0,
        turn_player_has_rolled=bool(player_has_rolled(game_state, turn_player_color)),
    ))
