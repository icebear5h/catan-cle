"""
Module with main GameState class and main apply_action call (game controller).
"""

import copy
import random
import pickle
from collections import defaultdict
from typing import Any, List, Sequence, Dict

from cle.game_engine.models.map import BASE_MAP_TEMPLATE, CatanMap
from cle.game_engine.models.board import Board
from cle.game_engine.models.enums import (
    DEVELOPMENT_CARDS,
    MONOPOLY,
    RESOURCES,
    YEAR_OF_PLENTY,
    SETTLEMENT,
    CITY,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
    ROAD_COST_FREQDECK,
    SETTLEMENT_COST_FREQDECK,
    draw_from_listdeck,
    freqdeck_add,
    freqdeck_can_draw,
    freqdeck_contains,
    freqdeck_draw,
    freqdeck_from_listdeck,
    freqdeck_replenish,
    freqdeck_subtract,
    starting_devcard_bank,
    starting_resource_bank,
)
from cle.game_engine.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
    steal_possibilities,
)
from cle.game_engine.state_functions import (
    build_city,
    build_road,
    build_settlement,
    buy_dev_card,
    maintain_longest_road,
    play_dev_card,
    get_player_freqdeck,
    player_can_afford_dev_card,
    player_can_play_dev,
    player_clean_turn,
    player_freqdeck_add,
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_freqdeck_subtract,
    player_deck_to_array,
    player_key,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)
from cle.game_engine.models.player import Color
from cle.game_engine.models.enums import FastResource
from cle.game_engine.trading import (
    TradeCandidate,
    TradeLimits,
    TradeOffer,
    TradeWindow,
    TradeWindowStatus,
)

ROADS_PER_PLAYER = 15
SETTLEMENTS_PER_PLAYER = 5
CITIES_PER_PLAYER = 4


# These will be prefixed by P0_, P1_, ...
# Create Player GameState blueprint
PLAYER_INITIAL_STATE = {
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

    def __init__(
        self,
        colors: Sequence[Color],
        catan_map=None,
        discard_limit=7,
        initialize=True,
        shuffle_players=True,
        rng=None,
        trade_limits: TradeLimits | None = None,
    ):
        if initialize:
            self.rng = rng if rng is not None else random.Random()
            ordered_colors = (
                self.rng.sample(colors, len(colors))
                if shuffle_players
                else list(colors)
            )
            self.colors = tuple(ordered_colors)
            self.board = Board(
                catan_map or CatanMap.from_template(BASE_MAP_TEMPLATE, rng=self.rng)
            )
            self.discard_limit = discard_limit

            # feature-ready dictionary
            self.player_state = dict()
            for index in range(len(self.colors)):
                for key, value in PLAYER_INITIAL_STATE.items():
                    self.player_state[f"P{index}_{key}"] = value
            self.color_to_index = {
                color: index for index, color in enumerate(self.colors)
            }

            self.resource_freqdeck = starting_resource_bank()
            self.development_listdeck = starting_devcard_bank()
            self.rng.shuffle(self.development_listdeck)

            # Auxiliary attributes to implement game logic
            self.buildings_by_color: Dict[Color, Dict[Any, Any]] = {
                color: defaultdict(list) for color in self.colors
            }
            self.actions: List[Action] = []  # log of all action taken by players
            self.num_turns = 0  # num_completed_turns

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

            self.last_dice_roll = None  # Track last dice roll for logging

            self.playable_actions = generate_playable_actions(self)

    def current_color(self):
        """Helper for accessing color (player) who should decide next"""
        return self.colors[self.current_player_index]

    def copy(self):
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
        payload_memo = {}
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


def roll_dice(rng=None):
    """Yield two dice from the supplied game-local random stream."""
    source = rng if rng is not None else random.SystemRandom()
    return (source.randint(1, 6), source.randint(1, 6))


def yield_resources(board: Board, resource_freqdeck, number):
    """Computes resource payouts for given board and dice roll number.

    Args:
        board (Board): Board state
        resource_freqdeck (List[int]): Bank's resource freqdeck
        number (int): Sum of dice roll

    Returns:
        (dict, List[int]): 2-tuple.
            First element is color => freqdeck mapping. e.g. {Color.RED: [0,0,0,3,0]}.
            Second lists resources whose demand exceeded the bank, including
            partial payouts to a sole entitled player.
    """
    intented_payout: Dict[Color, Dict[FastResource, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    resource_totals: Dict[FastResource, int] = defaultdict(int)
    for coordinate, tile in board.map.land_tiles.items():
        if tile.number != number or board.robber_coordinate == coordinate:
            continue  # doesn't yield

        for node_id in tile.nodes.values():
            building = board.buildings.get(node_id, None)
            assert tile.resource is not None
            if building is None:
                continue
            elif building[1] == SETTLEMENT:
                intented_payout[building[0]][tile.resource] += 1
                resource_totals[tile.resource] += 1
            elif building[1] == CITY:
                intented_payout[building[0]][tile.resource] += 2
                resource_totals[tile.resource] += 2

    # for each resource, check enough in deck to yield.
    depleted = []
    for resource in RESOURCES:
        total = resource_totals[resource]
        if not freqdeck_can_draw(resource_freqdeck, total, resource):
            depleted.append(resource)

    # build final data color => freqdeck structure
    payout = {}
    for player, player_payout in intented_payout.items():
        payout[player] = [0, 0, 0, 0, 0]

        for resource, count in player_payout.items():
            if resource in depleted:
                # Multiple buildings still count as one entitled player.
                count = (
                    resource_freqdeck[RESOURCES.index(resource)]
                    if count == resource_totals[resource]
                    else 0
                )
            freqdeck_replenish(payout[player], count, resource)

    return payout, depleted


def advance_turn(state, direction=1):
    """Sets .current_player_index"""
    next_index = next_player_index(state, direction)
    state.current_player_index = next_index
    state.current_turn_index = next_index
    state.num_turns += 1


def next_player_index(state, direction=1):
    return (state.current_player_index + direction) % len(state.colors)


def assert_forced_action_is_explicit(action: Action):
    """Reject forced actions that would otherwise rely on engine randomness."""
    if action.action_type == ActionType.ROLL:
        if not isinstance(action.value, (tuple, list)) or len(action.value) != 2:
            raise ValueError("Forced ROLL requires explicit dice tuple")
    elif action.action_type == ActionType.DISCARD:
        if action.value is None:
            raise ValueError("Forced DISCARD requires explicit discarded cards")
    elif action.action_type == ActionType.BUY_DEVELOPMENT_CARD:
        if action.value is None:
            raise ValueError("Forced BUY_DEVELOPMENT_CARD requires explicit card type")
    elif action.action_type == ActionType.STEAL:
        if (
            not isinstance(action.value, (tuple, list))
            or len(action.value) != 2
            or action.value[0] is None
            or action.value[1] is None
        ):
            raise ValueError("Forced STEAL requires explicit victim and resource")
    elif action.action_type == ActionType.MOVE_ROBBER:
        if action.value is None:
            raise ValueError("Forced MOVE_ROBBER requires explicit tile coordinate")
    elif action.action_type == ActionType.PLAY_YEAR_OF_PLENTY:
        if not action.value or any(resource is None for resource in action.value):
            raise ValueError("Forced PLAY_YEAR_OF_PLENTY requires explicit resources")
    elif action.action_type == ActionType.PLAY_MONOPOLY:
        if action.value is None:
            raise ValueError("Forced PLAY_MONOPOLY requires explicit resource")


def apply_action(state: GameState, action: Action, force: bool = False):
    """Main controller call. Follows redux-like pattern and
    routes the given action to the appropiate state-changing calls.

    Responsible for maintaining:
        .current_player_index, .current_turn_index,
        .current_prompt (and similars), .playable_actions.

    Appends given action to the list of actions, as fully-specified action.

    Args:
        state (GameState): GameState to mutate
        action (Action): Action to carry out

    Raises:
        ValueError: If invalid action given

    Returns:
        Action: Fully-specified action
    """

    if force:
        assert_forced_action_is_explicit(action)

    executed_action = None

    match action.action_type:
        case ActionType.END_TURN:
            executed_action = apply_end_turn(state, action, force=force)
        case ActionType.BUILD_SETTLEMENT:
            executed_action = apply_build_settlement(state, action, force=force)
        case ActionType.BUILD_ROAD:
            executed_action = apply_build_road(state, action, force=force)
        case ActionType.BUILD_CITY:
            executed_action = apply_build_city(state, action, force=force)
        case ActionType.BUY_DEVELOPMENT_CARD:
            executed_action = apply_buy_development_card(state, action, force=force)
        case ActionType.ROLL:
            executed_action = apply_roll(state, action, force=force)
        case ActionType.DISCARD:
            executed_action = apply_discard(state, action, force=force)
        case ActionType.MOVE_ROBBER:
            executed_action = apply_move_robber(state, action, force=force)
        case ActionType.STEAL:
            executed_action = apply_steal(state, action, force=force)
        case ActionType.PLAY_KNIGHT_CARD:
            executed_action = apply_play_knight_card(state, action, force=force)
        case ActionType.PLAY_YEAR_OF_PLENTY:
            executed_action = apply_play_year_of_plenty(state, action, force=force)
        case ActionType.PLAY_MONOPOLY:
            executed_action = apply_play_monopoly(state, action, force=force)
        case ActionType.PLAY_ROAD_BUILDING:
            executed_action = apply_play_road_building(state, action, force=force)
        case ActionType.MARITIME_TRADE:
            executed_action = apply_maritime_trade(state, action, force=force)
        case ActionType.OFFER_TRADE:
            executed_action = apply_offer_trade(state, action, force=force)
        case ActionType.ACCEPT_TRADE:
            executed_action = apply_accept_trade(state, action, force=force)
        case ActionType.REJECT_TRADE:
            executed_action = apply_reject_trade(state, action, force=force)
        case ActionType.CONFIRM_TRADE:
            executed_action = apply_confirm_trade(state, action, force=force)
        case ActionType.COUNTER_OFFER:
            executed_action = apply_counter_offer(state, action, force=force)
        case ActionType.CANCEL_TRADE:
            executed_action = apply_cancel_trade(state, action, force=force)
        case _:
            raise ValueError("Unknown ActionType " + str(action.action_type))

    if executed_action is None:
        executed_action = action

    state.actions.append(executed_action)
    return executed_action


def new_trade_window(state: GameState) -> TradeWindow:
    """Preview a new window without mutating state or consuming an identity."""
    return TradeWindow(
        id=f"turn-{state.num_turns}-trade-{len(state.actions)}",
        turn_player=state.colors[state.current_turn_index],
        participants=state.colors,
        limits=state.trade_limits,
    )


def ensure_trade_window(state: GameState) -> TradeWindow:
    window = state.trade_window
    if window is None or window.status == TradeWindowStatus.CLOSED:
        window = new_trade_window(state)
        state.trade_window = window
    return window


def latest_trade_offer(
    state: GameState,
    offered_by: Color,
    *,
    root: bool | None = None,
):
    if state.trade_window is None:
        return None
    offers = [
        offer
        for offer in state.trade_window.active_offers
        if offer.offered_by == offered_by
        and (root is None or (offer.parent_offer_id is None) == root)
    ]
    return offers[-1] if offers else None


def offer_for_response(state: GameState, value, *, root: bool = True):
    if isinstance(value, str) and state.trade_window is not None:
        offer = state.trade_window.offers.get(value)
        if offer is not None and offer.active:
            return offer
    return latest_trade_offer(state, value, root=root)


def reset_trading_state(state):
    """Close the turn-scoped offer board."""
    if state.trade_window is not None:
        state.trade_window.close()


# ===== Apply Action Handlers =====
def require_available_piece(
    state: GameState,
    color: Color,
    field: str,
    label: str,
):
    """Reject construction before the board mutates if no piece remains."""
    key = player_key(state, color)
    if state.player_state[f"{key}_{field}_AVAILABLE"] <= 0:
        raise ValueError(f"No {label} pieces available for {color.value}")


def require_build_resources(
    state: GameState,
    color: Color,
    cost: List[int],
    label: str,
):
    """Reject paid construction before it can create cards in the bank."""
    if not player_resource_freqdeck_contains(state, color, cost):
        raise ValueError(f"{color.value} cannot afford to build a {label}")


def apply_end_turn(state: GameState, action: Action, force: bool = False):
    # Reset any pending trade state (like Colonist's auto-cancel on end turn)
    reset_trading_state(state)
    player_clean_turn(state, action.color)
    advance_turn(state)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_build_settlement(state: GameState, action: Action, force: bool = False):
    require_available_piece(
        state,
        action.color,
        "SETTLEMENTS",
        "settlement",
    )
    if not state.is_initial_build_phase:
        require_build_resources(
            state,
            action.color,
            SETTLEMENT_COST_FREQDECK,
            "settlement",
        )
    node_id = action.value
    if state.is_initial_build_phase:
        result = state.board.build_settlement(action.color, node_id, True)
        build_settlement(state, action.color, node_id, True)
        maintain_longest_road(state, *result)
        buildings = state.buildings_by_color[action.color][SETTLEMENT]

        # yield resources if second settlement
        is_second_house = len(buildings) == 2
        if is_second_house:
            key = player_key(state, action.color)
            for tile in state.board.map.adjacent_tiles[node_id]:
                if tile.resource is not None:
                    freqdeck_draw(state.resource_freqdeck, 1, tile.resource)  # type: ignore
                    state.player_state[f"{key}_{tile.resource}_IN_HAND"] += 1

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.BUILD_INITIAL_ROAD
        state.playable_actions = generate_playable_actions(state)
    else:
        (
            previous_road_color,
            road_color,
            road_lengths,
        ) = state.board.build_settlement(action.color, node_id, False)
        build_settlement(state, action.color, node_id, False)
        state.resource_freqdeck = freqdeck_add(
            state.resource_freqdeck, SETTLEMENT_COST_FREQDECK
        )  # replenish bank
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)


def apply_build_road(state: GameState, action: Action, force: bool = False):
    require_available_piece(state, action.color, "ROADS", "road")
    is_free = state.is_initial_build_phase or (
        state.is_road_building and state.free_roads_available > 0
    )
    if not is_free:
        require_build_resources(state, action.color, ROAD_COST_FREQDECK, "road")
    edge = action.value
    if state.is_initial_build_phase:
        result = state.board.build_road(action.color, edge)
        build_road(state, action.color, edge, True)
        maintain_longest_road(state, *result)

        # state.current_player_index depend on what index are we
        # state.current_prompt too
        buildings = [
            len(state.buildings_by_color[color][SETTLEMENT])
            for color in state.color_to_index.keys()
        ]
        num_buildings = sum(buildings)
        num_players = len(buildings)
        going_forward = num_buildings < num_players
        at_the_end = num_buildings == num_players
        if going_forward:
            advance_turn(state)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif at_the_end:
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        elif num_buildings == 2 * num_players:
            state.is_initial_build_phase = False
            state.current_prompt = ActionPrompt.PLAY_TURN
        else:
            advance_turn(state, -1)
            state.current_prompt = ActionPrompt.BUILD_INITIAL_SETTLEMENT
        state.playable_actions = generate_playable_actions(state)
    elif state.is_road_building and state.free_roads_available > 0:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, True)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        state.free_roads_available -= 1
        if (
            state.free_roads_available == 0
            or len(road_building_possibilities(state, action.color, False)) == 0
        ):
            state.is_road_building = False
            state.free_roads_available = 0
            # state.current_player_index stays the same
            # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)
    else:
        result = state.board.build_road(action.color, edge)
        previous_road_color, road_color, road_lengths = result
        build_road(state, action.color, edge, False)
        maintain_longest_road(state, previous_road_color, road_color, road_lengths)

        # state.current_player_index stays the same
        # state.current_prompt stays as PLAY
        state.playable_actions = generate_playable_actions(state)


def apply_build_city(state: GameState, action: Action, force: bool = False):
    require_available_piece(state, action.color, "CITIES", "city")
    require_build_resources(state, action.color, CITY_COST_FREQDECK, "city")
    node_id = action.value
    state.board.build_city(action.color, node_id)
    build_city(state, action.color, node_id)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, CITY_COST_FREQDECK
    )  # replenish bank

    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    state.playable_actions = generate_playable_actions(state)


def apply_buy_development_card(state: GameState, action: Action, force: bool = False):
    if len(state.development_listdeck) == 0:
        raise ValueError("No more development cards")
    if not player_can_afford_dev_card(state, action.color):
        raise ValueError("No money to buy development card")

    if action.value is None:
        if force:
            raise ValueError("Forced BUY_DEVELOPMENT_CARD requires explicit card type")
        card = state.development_listdeck.pop()  # already shuffled
    else:
        card = action.value
        draw_from_listdeck(state.development_listdeck, 1, card)

    buy_dev_card(state, action.color, card)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, DEVELOPMENT_CARD_COST_FREQDECK
    )

    action = Action(action.color, action.action_type, card)
    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    state.playable_actions = generate_playable_actions(state)
    return action


def apply_roll(state: GameState, action: Action, force: bool = False):
    key = player_key(state, action.color)
    state.player_state[f"{key}_HAS_ROLLED"] = True

    if action.value is None and force:
        raise ValueError("Forced ROLL requires explicit dice tuple")
    dices = action.value or roll_dice(state.rng)
    number = dices[0] + dices[1]

    # Store dice values for tracking/logging
    state.last_dice_roll = dices

    action = Action(action.color, action.action_type, dices)

    if number == 7:
        discarders = [
            player_num_resource_cards(state, color) > state.discard_limit
            for color in state.colors
        ]
        should_enter_discarding_sequence = any(discarders)

        if should_enter_discarding_sequence:
            state.current_player_index = discarders.index(True)
            state.current_prompt = ActionPrompt.DISCARD
            state.is_discarding = True
        else:
            # state.current_player_index stays the same
            state.current_prompt = ActionPrompt.MOVE_ROBBER
            state.is_moving_knight = True
        state.playable_actions = generate_playable_actions(state)
    else:
        payout, _ = yield_resources(state.board, state.resource_freqdeck, number)
        for color, resource_freqdeck in payout.items():
            # Atomically add to player's hand and remove from bank
            player_freqdeck_add(state, color, resource_freqdeck)
            state.resource_freqdeck = freqdeck_subtract(
                state.resource_freqdeck, resource_freqdeck
            )

        # state.current_player_index stays the same
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.playable_actions = generate_playable_actions(state)
    return action


def validate_discard(
    state: GameState, action: Action, *, force: bool = False
) -> tuple[str, ...]:
    """Validate exact cards without mutation; replay force skips turn/phase eligibility."""
    if not isinstance(action, Action) or action.action_type != ActionType.DISCARD:
        raise ValueError("Expected a DISCARD action")
    if action.color not in state.colors:
        raise ValueError("Discard player must be a participant")
    if not force and (
        state.current_prompt != ActionPrompt.DISCARD or action.color != state.current_color()
    ):
        raise ValueError("Discard is not requested from this player right now")
    if not isinstance(action.value, (list, tuple)):
        raise ValueError("Discard cards must be a list or tuple of resource names")
    if any(not isinstance(resource, str) or resource not in RESOURCES for resource in action.value):
        raise ValueError("Discard cards must contain only recognized resource names")
    hand_size = player_num_resource_cards(state, action.color)
    if not force and hand_size <= state.discard_limit:
        raise ValueError("Player's hand does not exceed the discard limit")
    num_to_discard = hand_size // 2
    if len(action.value) != num_to_discard:
        raise ValueError(f"Must discard exactly {num_to_discard} resource cards")
    discarded = tuple(action.value)
    if not player_resource_freqdeck_contains(
        state, action.color, freqdeck_from_listdeck(discarded)
    ):
        raise ValueError("Cannot discard resource cards the player does not hold")
    return discarded


def apply_discard(state: GameState, action: Action, force: bool = False):
    if action.value is None:
        if force:
            raise ValueError("Forced DISCARD requires explicit discarded cards")
        if action.color not in state.colors:
            raise ValueError("Discard player must be a participant")
        hand = player_deck_to_array(state, action.color)
        num_to_discard = len(hand) // 2
        # Check the request before consuming randomness for the shipped auto path.
        validate_discard(state, Action(action.color, action.action_type, hand[:num_to_discard]))
        discarded = tuple(state.rng.sample(hand, k=num_to_discard))
    else:
        discarded = validate_discard(state, action, force=force)
    to_discard = freqdeck_from_listdeck(discarded)

    player_freqdeck_subtract(state, action.color, to_discard)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, to_discard)
    action = Action(action.color, action.action_type, discarded)

    # Advance turn
    discarders_left = [
        player_num_resource_cards(state, color) > state.discard_limit for color in state.colors
    ][state.current_player_index + 1 :]
    if any(discarders_left):
        to_skip = discarders_left.index(True)
        state.current_player_index = state.current_player_index + 1 + to_skip
        state.current_prompt = ActionPrompt.DISCARD
        state.is_discarding = True
    else:
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_discarding = False
        state.is_moving_knight = True

    state.playable_actions = generate_playable_actions(state)
    return action


def apply_move_robber(state: GameState, action: Action, force: bool = False):
    """Move robber to new tile. STEAL is now a separate action."""
    coordinate = action.value
    state.board.robber_coordinate = coordinate

    # Check if there are players to steal from at this tile
    possible_steals = steal_possibilities(state, action.color)

    if len(possible_steals) > 0:
        # Must steal from someone
        state.current_prompt = ActionPrompt.STEAL
        state.is_moving_knight = False  # Clear knight flag since robber is moved
    else:
        # No one to steal from, go back to playing
        state.current_prompt = ActionPrompt.PLAY_TURN
        state.is_moving_knight = False

    state.playable_actions = generate_playable_actions(state)


def apply_steal(state: GameState, action: Action, force: bool = False):
    """Steal a card from a player at the robber's tile."""
    (robbed_color, robbed_resource) = action.value

    if robbed_resource is None:
        if force:
            raise ValueError("Forced STEAL requires explicit victim and resource")
        robbed_resource = player_deck_random_draw(state, robbed_color)
        action = Action(
            action.color,
            action.action_type,
            (robbed_color, robbed_resource),
        )
    else:  # for replay functionality
        player_deck_draw(state, robbed_color, robbed_resource)

    player_deck_replenish(state, action.color, robbed_resource)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
    return action


def apply_play_knight_card(state: GameState, action: Action, force: bool = False):
    if not player_can_play_dev(state, action.color, "KNIGHT"):
        raise ValueError("Player cant play knight card now")

    play_dev_card(state, action.color, "KNIGHT")

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.MOVE_ROBBER
    state.playable_actions = generate_playable_actions(state)


def apply_play_year_of_plenty(state: GameState, action: Action, force: bool = False):
    cards_selected = freqdeck_from_listdeck(action.value)
    if not player_can_play_dev(state, action.color, YEAR_OF_PLENTY):
        raise ValueError("Player cant play year of plenty now")
    if not freqdeck_contains(state.resource_freqdeck, cards_selected):
        raise ValueError("Not enough resources of this type (these types?) in bank")
    player_freqdeck_add(state, action.color, cards_selected)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, cards_selected)
    play_dev_card(state, action.color, YEAR_OF_PLENTY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_play_monopoly(state: GameState, action: Action, force: bool = False):
    mono_resource = action.value
    cards_stolen = [0, 0, 0, 0, 0]
    if not player_can_play_dev(state, action.color, MONOPOLY):
        raise ValueError("Player cant play monopoly now")
    for color in state.colors:
        if not color == action.color:
            key = player_key(state, color)
            number_of_cards_to_steal = state.player_state[
                f"{key}_{mono_resource}_IN_HAND"
            ]
            freqdeck_replenish(cards_stolen, number_of_cards_to_steal, mono_resource)
            player_deck_draw(state, color, mono_resource, number_of_cards_to_steal)
    player_freqdeck_add(state, action.color, cards_stolen)
    play_dev_card(state, action.color, MONOPOLY)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_play_road_building(state: GameState, action: Action, force: bool = False):
    if not player_can_play_dev(state, action.color, "ROAD_BUILDING"):
        raise ValueError("Player cant play road building now")

    play_dev_card(state, action.color, "ROAD_BUILDING")
    state.is_road_building = True
    state.free_roads_available = 2

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_maritime_trade(state: GameState, action: Action, force: bool = False):
    trade_offer = action.value

    # Support two formats:
    # 1. New format: (given_freqdeck, received_freqdeck) - two 5-tuples for multi-resource trades
    # 2. Old format: (res, res, res, res, res_asked) - single resource trade
    if isinstance(trade_offer, tuple) and len(trade_offer) == 2:
        # New format: multi-resource maritime trade
        offering, asking = trade_offer
        # Ensure they're tuples/lists of length 5
        if not (len(offering) == 5 and len(asking) == 5):
            raise ValueError(f"Invalid maritime trade format: {trade_offer}")
    else:
        # Old format: single resource trade
        offering = freqdeck_from_listdeck(filter(lambda r: r is not None, trade_offer[:-1]))
        asking = freqdeck_from_listdeck(trade_offer[-1:])

    if not player_resource_freqdeck_contains(state, action.color, offering):
        raise ValueError("Trying to trade without money")
    if not freqdeck_contains(state.resource_freqdeck, asking):
        raise ValueError("Bank doenst have those cards")
    player_freqdeck_subtract(state, action.color, offering)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, offering)
    player_freqdeck_add(state, action.color, asking)
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, asking)

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_offer_trade(state: GameState, action: Action, force: bool = False):
    offer = action.value
    if not isinstance(offer, TradeOffer) or offer.parent_offer_id is not None:
        raise ValueError("OFFER_TRADE requires one root TradeOffer")
    if not force and offer.id is not None:
        raise ValueError("Trade offer IDs are assigned by the engine")
    materialized = ensure_trade_window(state).create_offer(
        offer,
        allow_duplicate=force,
    )
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
    return Action(action.color, action.action_type, copy.deepcopy(materialized))


def apply_accept_trade(state: GameState, action: Action, force: bool = False):
    offer = offer_for_response(state, action.value, root=None)
    if offer is None:
        raise ValueError(f"No active offer for {action.value}")
    state.trade_window.signal_willingness(offer.id, action.color)
    state.playable_actions = generate_playable_actions(state)


def apply_reject_trade(state: GameState, action: Action, force: bool = False):
    offer = offer_for_response(state, action.value, root=None)
    if offer is None:
        raise ValueError(f"No active offer for {action.value}")
    state.trade_window.decline(offer.id, action.color)
    state.playable_actions = generate_playable_actions(state)


def apply_confirm_trade(state: GameState, action: Action, force: bool = False):
    window = state.trade_window
    if window is None:
        raise ValueError("No active trade window")
    candidate = action.value
    if not isinstance(candidate, TradeCandidate):
        raise ValueError("CONFIRM_TRADE requires a typed TradeCandidate")
    offer = window.offers.get(candidate.offer_id)
    if offer is None or not offer.active:
        raise ValueError(f"No active offer {candidate.offer_id}")
    window.select(action.color, candidate)
    counterparty = candidate.counterparty

    if offer.offered_by == action.color:
        actor_gives, actor_receives = offer.give, offer.receive
    else:
        actor_gives, actor_receives = offer.receive, offer.give
    if not freqdeck_contains(get_player_freqdeck(state, action.color), actor_gives):
        raise ValueError(f"{action.color} can no longer afford this trade")
    if not freqdeck_contains(get_player_freqdeck(state, counterparty), actor_receives):
        raise ValueError(f"{counterparty} can no longer afford this trade")

    player_freqdeck_subtract(state, action.color, actor_gives)
    player_freqdeck_add(state, action.color, actor_receives)
    player_freqdeck_subtract(state, counterparty, actor_receives)
    player_freqdeck_add(state, counterparty, actor_gives)
    window.mark_executed()
    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_counter_offer(state: GameState, action: Action, force: bool = False):
    offer = action.value
    if not isinstance(offer, TradeOffer) or offer.parent_offer_id is None:
        raise ValueError("COUNTER_OFFER requires one counter TradeOffer")
    if not force and offer.id is not None:
        raise ValueError("Trade offer IDs are assigned by the engine")
    window = ensure_trade_window(state)
    if offer.parent_offer_id not in window.offers:
        raise ValueError(f"No active root offer {offer.parent_offer_id!r}")
    materialized = window.create_offer(offer, allow_duplicate=force)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
    return Action(action.color, action.action_type, copy.deepcopy(materialized))


def apply_cancel_trade(state: GameState, action: Action, force: bool = False):
    window = state.trade_window
    if window is not None:
        if isinstance(action.value, str):
            window.withdraw(action.value, action.color)
        else:
            for offer in tuple(window.active_offers):
                if offer.offered_by == action.color:
                    window.withdraw(offer.id, action.color)
    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)
