"""Text-based observation formatter for LLM agents.

Public classes stay here so historical imports and pickles retain their identity.
"""

from dataclasses import dataclass

from cle.game_engine.models.coordinate_system import Coordinate
from cle.game_engine.models.enums import Action, FastBuildingType
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.state import GameState
from cle.game_engine.trading import ResourceBundle, TradeWindow

from . import actions, board, events, factory, formatting, phase, players, trade
from . import resources as resource_formatting


@dataclass
class CatanObservation:
    """Legacy structured observation; the factory intentionally shares state objects."""

    my_color: Color
    my_settlements: list[int]
    my_cities: list[int]
    my_roads: list[tuple[int, int]]
    opponent_settlements: dict[Color, list[int]]
    opponent_cities: dict[Color, list[int]]
    opponent_roads: dict[Color, list[tuple[int, int]]]
    my_resources: dict[str, int]
    my_dev_cards: dict[str, int]
    opponent_resource_counts: dict[Color, int]
    opponent_dev_card_counts: dict[Color, int]
    current_turn: int
    current_phase: str
    turn_order: tuple[Color, ...]
    last_dice_roll: int | None
    robber_position: Coordinate
    my_vp: int
    opponent_vps: dict[Color, int]
    longest_road_holder: Color | None
    largest_army_holder: Color | None
    my_longest_road_length: int
    valid_actions: list[Action]
    board_map: CatanMap
    buildings_dict: dict[int, tuple[Color, FastBuildingType]]
    trade_window: TradeWindow | None
    is_my_turn: bool
    turn_player_color: Color
    recent_events: list[Action] | None = None

    def __post_init__(self) -> None:
        if self.recent_events is None:
            self.recent_events = []


@dataclass
class FormattedObservation:
    """Semantic text representation of game state for LLM."""

    raw_str: str
    board_state: str
    resources: str
    opponents: str
    valid_actions: str
    strategic_context: str
    trade_context: str


Observation = CatanObservation | PlayerObservation


class CatanObservationFormatter:
    """Format legacy and live-engine observations, preserving method dispatch."""

    # Annotation only: historically the cache first exists after format().
    _node_coords: dict[int, str]

    def format(
        self,
        obs: Observation,
        *,
        include_legal_actions: bool = True,
        include_initial_placement_order: bool = True,
        shared: bool = False,
    ) -> FormattedObservation:
        return formatting.format_observation(
            self, obs, include_legal_actions=include_legal_actions,
            include_initial_placement_order=include_initial_placement_order, shared=shared,
        )

    def _format_board_state(self, obs: Observation) -> str:
        return board.format_board_state(self, obs)

    def _get_node_strategic_context(self, node_id: int, obs: Observation) -> str:
        return board.get_node_strategic_context(self, node_id, obs)

    def _number_to_pips(self, number: int) -> int:
        return board.number_to_pips(number)

    def _format_resources(self, obs: Observation, *, shared: bool = False) -> str:
        return resource_formatting.format_resources(self, obs, shared=shared)

    def _get_affordable_buildings(self, resources: dict[str, int]) -> list[str]:
        return resource_formatting.get_affordable_buildings(resources)

    def _format_opponents(self, obs: Observation, *, shared: bool = False) -> str:
        return players.format_opponents(self, obs, shared=shared)

    def _format_valid_actions(self, obs: Observation) -> str:
        return actions.format_valid_actions(self, obs)

    def _describe_node(self, node_id: int, obs: Observation) -> str:
        return board.describe_node(self, node_id, obs)

    def _format_single_action(
        self, action: Action, obs: Observation, *, discard_count: int | None = None,
    ) -> str:
        return actions.format_single_action(self, action, obs, discard_count=discard_count)

    def _format_trade_context(self, obs: Observation) -> str:
        return trade.format_trade_context(self, obs)

    def _format_resource_tuple(self, resources: ResourceBundle) -> str:
        return trade.format_resource_tuple(resources)

    def _format_resource_tuple_with_any(self, resources: ResourceBundle, any_count: int) -> str:
        return trade.format_resource_tuple_with_any(resources, any_count)

    def _color_name(self, color: Color | str) -> str:
        return players.color_name(color)

    def _build_node_coordinate_map(self, board_map: CatanMap) -> dict[int, str]:
        return board.build_node_coordinate_map(board_map)

    def _format_node(self, node_id: int, node_coords: dict[int, str]) -> str:
        return board.format_node(node_id, node_coords)

    def _format_edge(self, edge: tuple[int, int], node_coords: dict[int, str]) -> str:
        return board.format_edge(self, edge, node_coords)

    def _format_events(self, obs: Observation) -> str:
        return events.format_events(self, obs)

    def _format_strategic_context(
        self, obs: Observation, *, include_initial_placement_order: bool, shared: bool = False,
    ) -> str:
        return phase.format_strategic_context(
            self, obs, include_initial_placement_order=include_initial_placement_order, shared=shared,
        )


def create_observation_from_state(
    game_state: GameState, player_color: Color, recent_events: list[Action] | None = None,
) -> CatanObservation:
    """Convert engine state using the historical, shallow legacy factory."""
    return factory.create_observation_from_state(game_state, player_color, recent_events)
