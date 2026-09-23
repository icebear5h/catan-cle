"""GameState: the mutable game record, its blueprint constants, and copying."""

import copy
import pickle
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import Board
from cle.game_engine.models.decks import starting_devcard_bank, starting_resource_bank
from cle.game_engine.models.enums import (
    DEVELOPMENT_CARDS,
    RESOURCES,
    Action,
    ActionPrompt,
    FastDevCard,
)
from cle.game_engine.models.map import BASE_MAP_TEMPLATE, CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.trading import TradeLimits, TradeWindow

if TYPE_CHECKING:
    from cle.game_engine.state_functions.getters import PlayerBuildings


ROADS_PER_PLAYER = 15
SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4


# These will be prefixed by P0_, P1_, ...
# Create Player GameState blueprint
PLAYER_INITIAL_STATE: dict[str, int | bool] = {
    "VICTORY_POINTS": 0,
    "ROADS_AVAILABLE": ROADS_PER_PLAYER,
    "SETTLEMENTS_AVAILABLE": SETTLEMENTS_PER_PLAYER,
    "CITIES_AVAILABLE": CITIES_PER_PLAYER,
    "HAS_ROAD": False,
    "HAS_ARMY": False,
    "HAS_ROLLED": False,
    "HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN": False,
    # de-normalized features (for performance since we think they are good features)
    "ACTUAL_VICTORY_POINTS": 0,
    "LONGEST_ROAD_LENGTH": 0,
    "KNIGHT_OWNED_AT_START": False,
    "MONOPOLY_OWNED_AT_START": False,
    "YEAR_OF_PLENTY_OWNED_AT_START": False,
    "ROAD_BUILDING_OWNED_AT_START": False,
}
for resource in RESOURCES:
    PLAYER_INITIAL_STATE[f"{resource}_IN_HAND"] = 0
for dev_card in DEVELOPMENT_CARDS:
    PLAYER_INITIAL_STATE[f"{dev_card}_IN_HAND"] = 0
    PLAYER_INITIAL_STATE[f"PLAYED_{dev_card}"] = 0


class GameState:
    """Collection of variables representing state

    Attributes:
        board (Board): Board state. Settlement locations, cities,
            roads, ect... See Board class.
        player_state (Dict[str, Any]): See PLAYER_INITIAL_STATE. It will
            contain one of each key in PLAYER_INITIAL_STATE but prefixed
            with "P<index_of_player>".
            Example: { P0_HAS_ROAD: False, P1_SETTLEMENTS_AVAILABLE: 18, ... }
        color_to_index (Dict[Color, int]): Color to seating location cache
        colors (Tuple[Color]): Represents seating order.
        resource_freqdeck (List[int]): Represents resource cards in the bank.
            Each element is the amount of [WOOD, BRICK, SHEEP, WHEAT, ORE].
        development_listdeck (List[FastDevCard]): Represents development cards in
            the bank. Already shuffled.
        buildings_by_color (Dict[Color, Dict[FastBuildingType, List]]): Cache of
            buildings. Can be used like: `buildings_by_color[Color.RED][SETTLEMENT]`
            to get a list of all node ids where RED has settlements.
        actions (List[Action]): Log of all actions taken. Fully-specified actions.
        num_turns (int): number of turns thus far
        current_player_index (int): index per colors array of player that should be
            making a decision now. Not necesarilly the same as current_turn_index
            because there are out-of-turn decisions like discarding.
        current_turn_index (int): index per colors array of player whose turn is it.
        current_prompt (ActionPrompt): DEPRECATED. Not needed; use is_initial_build_phase,
            is_moving_knight, etc... instead.
        is_discarding (bool): If current player needs to discard.
        is_moving_knight (bool): If current player needs to move robber.
        is_road_building (bool): If current player needs to build free roads per Road
            Building dev card.
        free_roads_available (int): Number of roads available left in Road Building
            phase.
        playable_actions (List[Action]): List of playable actions by current player.
    """

    __module__ = "cle.game_engine.state"  # pickle identity of saved snapshots

    def __init__(
        self,
        colors: Sequence[Color],
        catan_map: CatanMap | None = None,
        discard_limit: int = 7,
        initialize: bool = True,
        shuffle_players: bool = True,
        rng: random.Random | None = None,
        trade_limits: TradeLimits | None = None,
    ) -> None:
        if initialize:
            self.rng: random.Random = rng if rng is not None else random.Random()
            ordered_colors = (
                self.rng.sample(colors, len(colors))
                if shuffle_players
                else list(colors)
            )
            self.colors: tuple[Color, ...] = tuple(ordered_colors)
            self.board = Board(
                catan_map or CatanMap.from_template(BASE_MAP_TEMPLATE, rng=self.rng)
            )
            self.discard_limit: int = discard_limit

            # feature-ready dictionary
            self.player_state: dict[str, int | bool] = dict()
            for index in range(len(self.colors)):
                for key, value in PLAYER_INITIAL_STATE.items():
                    self.player_state[f"P{index}_{key}"] = value
            self.color_to_index: dict[Color, int] = {
                color: index for index, color in enumerate(self.colors)
            }

            self.resource_freqdeck: list[int] = starting_resource_bank()
            self.development_listdeck: list[FastDevCard] = starting_devcard_bank()
            self.rng.shuffle(self.development_listdeck)

            # Auxiliary attributes to implement game logic
            self.buildings_by_color: dict[Color, PlayerBuildings] = {
                color: cast("PlayerBuildings", defaultdict(list)) for color in self.colors
            }
            self.actions: list[Action] = []  # log of all action taken by players
            self.num_turns: int = 0  # num_completed_turns

            # Current prompt / player
            # Two variables since there can be out-of-turn plays
            self.current_player_index = 0
            self.current_turn_index = 0

            # TODO: Deprecate self.current_prompt in favor of indicator variables
            self.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
            self.is_initial_build_phase = True
            self.is_discarding = False
            self.is_moving_knight = False
            self.is_road_building = False
            self.free_roads_available = 0

            self.trade_limits = trade_limits or TradeLimits()
            self.trade_window: TradeWindow | None = None

            # Track last dice roll for logging
            self.last_dice_roll: tuple[int, int] | None = None

            self.playable_actions: list[Action] = generate_playable_actions(self)

    def current_color(self) -> Color:
        """Helper for accessing color (player) who should decide next"""
        return self.colors[self.current_player_index]

    def copy(self) -> "GameState":
        """Creates a copy of this GameState class that can be modified without
        repercusions to this one. Immutable values are just copied over.

        Returns:
            GameState: GameState copy.
        """
        state_copy = GameState([], None, initialize=False)
        state_copy.rng = random.Random()
        state_copy.rng.setstate(self.rng.getstate())
        state_copy.discard_limit = self.discard_limit  # immutable

        state_copy.board = self.board.copy()

        state_copy.player_state = self.player_state.copy()
        state_copy.color_to_index = self.color_to_index.copy()
        state_copy.colors = self.colors  # immutable

        state_copy.resource_freqdeck = self.resource_freqdeck.copy()
        state_copy.development_listdeck = self.development_listdeck.copy()

        state_copy.buildings_by_color = pickle.loads(
            pickle.dumps(self.buildings_by_color)
        )
        # Action records are immutable, but replay dice and trade payloads are not.
        payload_memo: dict[int, object] = {}
        state_copy.actions = [
            Action(action.color, action.action_type, copy.deepcopy(action.value, payload_memo))
            for action in self.actions
        ]
        state_copy.num_turns = self.num_turns

        # Current prompt / player
        # Two variables since there can be out-of-turn plays
        state_copy.current_player_index = self.current_player_index
        state_copy.current_turn_index = self.current_turn_index

        state_copy.current_prompt = self.current_prompt
        state_copy.is_initial_build_phase = self.is_initial_build_phase
        state_copy.is_discarding = self.is_discarding
        state_copy.is_moving_knight = self.is_moving_knight
        state_copy.is_road_building = self.is_road_building
        state_copy.free_roads_available = self.free_roads_available

        state_copy.trade_limits = self.trade_limits
        state_copy.trade_window = copy.deepcopy(self.trade_window, payload_memo)

        state_copy.last_dice_roll = copy.deepcopy(self.last_dice_roll, payload_memo)

        state_copy.playable_actions = [
            Action(action.color, action.action_type, copy.deepcopy(action.value, payload_memo))
            for action in self.playable_actions
        ]
        return state_copy
