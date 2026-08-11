"""
Module with main State class and main apply_action call (game controller).
"""

import random
import pickle
from collections import defaultdict
from typing import Any, List, Sequence, Tuple, Dict

from engine.models.map import BASE_MAP_TEMPLATE, CatanMap
from engine.models.board import Board
from engine.models.enums import (
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
from engine.models.decks import (
    CITY_COST_FREQDECK,
    DEVELOPMENT_CARD_COST_FREQDECK,
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
from engine.models.actions import (
    generate_playable_actions,
    road_building_possibilities,
    steal_possibilities,
)
from engine.state_functions import (
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
from engine.models.player import Color, Player
from engine.models.enums import FastResource

# These will be prefixed by P0_, P1_, ...
# Create Player State blueprint
PLAYER_INITIAL_STATE = {
    "VICTORY_POINTS": 0,
    "ROADS_AVAILABLE": 15,
    "SETTLEMENTS_AVAILABLE": 5,
    "CITIES_AVAILABLE": 4,
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


class State:
    """Collection of variables representing state

    Attributes:
        players (List[Player]): DEPRECATED. Reference to list of players.
            Use .colors instead, and move this reference to the Game class.
            Deprecated because we want this class to only contain state
            information that can be easily copiable.
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
        players: Sequence[Player],
        catan_map=None,
        discard_limit=7,
        initialize=True,
        shuffle_players=True,
    ):
        if initialize:
            self.players = random.sample(players, len(players)) if shuffle_players else list(players)
            self.colors = tuple([player.color for player in self.players])
            self.board = Board(catan_map or CatanMap.from_template(BASE_MAP_TEMPLATE))
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
            random.shuffle(self.development_listdeck)

            # Auxiliary attributes to implement game logic
            self.buildings_by_color: Dict[Color, Dict[Any, Any]] = {
                p.color: defaultdict(list) for p in players
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

            # Multi-trade support: dict mapping creator Color -> trade info
            # Each trade info is a dict with:
            # {
            #   'offered': Tuple[5],  # Resources offered
            #   'wanted': Tuple[5],   # Resources wanted
            #   'offered_any': int,   # Wildcard resources offered
            #   'wanted_any': int,    # Wildcard resources wanted
            #   'acceptees': Set[Color],  # Who accepted this trade
            #   'rejecters': Set[Color],  # Who rejected this trade
            # }
            self.active_trades: Dict = {}

            # Counter-offers: offers directed at the turn player
            # Each counter-offer is a dict with:
            # {
            #   'offered': Tuple[5],  # Resources offered
            #   'wanted': Tuple[5],   # Resources wanted
            #   'offered_any': int,   # Wildcard resources offered
            #   'wanted_any': int,    # Wildcard resources wanted
            #   'acceptees': Set[Color],  # Others willing to make the same offer
            # }
            # Only the turn player can execute (ACCEPT_COUNTER_OFFER) these
            self.counter_offers: Dict = {}

            # Legacy fields (DEPRECATED - kept for backward compatibility)
            self.is_resolving_trade = False
            self.current_trade: Tuple = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            self.acceptees = tuple(False for _ in self.colors)
            self.rejecters = tuple(False for _ in self.colors)

            self.last_dice_roll = None  # Track last dice roll for logging

            self.playable_actions = generate_playable_actions(self)

    def current_player(self):
        """Helper for accessing Player instance who should decide next"""
        return self.players[self.current_player_index]

    def current_color(self):
        """Helper for accessing color (player) who should decide next"""
        return self.colors[self.current_player_index]

    def copy(self):
        """Creates a copy of this State class that can be modified without
        repercusions to this one. Immutable values are just copied over.

        Returns:
            State: State copy.
        """
        state_copy = State([], None, initialize=False)
        state_copy.players = self.players
        state_copy.discard_limit = self.discard_limit  # immutable

        state_copy.board = self.board.copy()

        state_copy.player_state = self.player_state.copy()
        state_copy.color_to_index = self.color_to_index
        state_copy.colors = self.colors  # immutable

        state_copy.resource_freqdeck = self.resource_freqdeck.copy()
        state_copy.development_listdeck = self.development_listdeck.copy()

        state_copy.buildings_by_color = pickle.loads(
            pickle.dumps(self.buildings_by_color)
        )
        state_copy.actions = self.actions.copy()
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

        # Multi-trade support
        state_copy.active_trades = pickle.loads(pickle.dumps(self.active_trades))
        state_copy.counter_offers = pickle.loads(pickle.dumps(self.counter_offers))

        # Legacy trade fields
        state_copy.is_resolving_trade = self.is_resolving_trade
        state_copy.current_trade = self.current_trade
        state_copy.acceptees = self.acceptees
        state_copy.rejecters = self.rejecters
        state_copy.last_dice_roll = self.last_dice_roll

        state_copy.playable_actions = self.playable_actions
        return state_copy


def roll_dice():
    """Yields two random numbers

    Returns:
        tuple[int, int]: 2-tuple of random numbers from 1 to 6 inclusive.
    """
    return (random.randint(1, 6), random.randint(1, 6))


def yield_resources(board: Board, resource_freqdeck, number):
    """Computes resource payouts for given board and dice roll number.

    Args:
        board (Board): Board state
        resource_freqdeck (List[int]): Bank's resource freqdeck
        number (int): Sum of dice roll

    Returns:
        (dict, List[int]): 2-tuple.
            First element is color => freqdeck mapping. e.g. {Color.RED: [0,0,0,3,0]}.
            Second is an array of resources that couldn't be yieleded
            because they depleted.
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
            if resource not in depleted:
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


def apply_action(state: State, action: Action, force: bool = False):
    """Main controller call. Follows redux-like pattern and
    routes the given action to the appropiate state-changing calls.

    Responsible for maintaining:
        .current_player_index, .current_turn_index,
        .current_prompt (and similars), .playable_actions.

    Appends given action to the list of actions, as fully-specified action.

    Args:
        state (State): State to mutate
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
        case ActionType.JOIN_COUNTER_OFFER:
            executed_action = apply_join_counter_offer(state, action, force=force)
        case ActionType.ACCEPT_COUNTER_OFFER:
            executed_action = apply_accept_counter_offer(state, action, force=force)
        case ActionType.CANCEL_TRADE:
            executed_action = apply_cancel_trade(state, action, force=force)
        case _:
            raise ValueError("Unknown ActionType " + str(action.action_type))

    if executed_action is None:
        executed_action = action

    state.actions.append(executed_action)
    return executed_action


def sync_legacy_trade_state(state):
    """Project color-keyed trade state into deprecated single-trade fields."""
    state.is_resolving_trade = bool(state.active_trades or state.counter_offers)

    selected_creator = None
    current_creator_idx = state.current_trade[10]
    if isinstance(current_creator_idx, int) and 0 <= current_creator_idx < len(state.colors):
        current_creator = state.colors[current_creator_idx]
        if current_creator in state.active_trades:
            selected_creator = current_creator

    if selected_creator is None and state.active_trades:
        selected_creator = next(reversed(state.active_trades))

    if selected_creator is None:
        state.current_trade = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        state.acceptees = tuple(False for _ in state.colors)
        state.rejecters = tuple(False for _ in state.colors)
        return

    trade_info = state.active_trades[selected_creator]
    state.current_trade = (
        *trade_info["offered"],
        *trade_info["wanted"],
        state.colors.index(selected_creator),
    )
    state.acceptees = tuple(
        color in trade_info["acceptees"] for color in state.colors
    )
    state.rejecters = tuple(
        color in trade_info["rejecters"] for color in state.colors
    )


def reset_trading_state(state):
    """Reset all trading state - called when a turn ends."""
    state.active_trades = {}
    state.counter_offers = {}
    sync_legacy_trade_state(state)


# ===== Apply Action Handlers =====
def apply_end_turn(state: State, action: Action, force: bool = False):
    # Reset any pending trade state (like Colonist's auto-cancel on end turn)
    reset_trading_state(state)
    player_clean_turn(state, action.color)
    advance_turn(state)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_build_settlement(state: State, action: Action, force: bool = False):
    node_id = action.value
    if state.is_initial_build_phase:
        state.board.build_settlement(action.color, node_id, True)
        build_settlement(state, action.color, node_id, True)
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


def apply_build_road(state: State, action: Action, force: bool = False):
    edge = action.value
    if state.is_initial_build_phase:
        state.board.build_road(action.color, edge)
        build_road(state, action.color, edge, True)

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


def apply_build_city(state: State, action: Action, force: bool = False):
    node_id = action.value
    state.board.build_city(action.color, node_id)
    build_city(state, action.color, node_id)
    state.resource_freqdeck = freqdeck_add(
        state.resource_freqdeck, CITY_COST_FREQDECK
    )  # replenish bank

    # state.current_player_index stays the same
    # state.current_prompt stays as PLAY
    state.playable_actions = generate_playable_actions(state)


def apply_buy_development_card(state: State, action: Action, force: bool = False):
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


def apply_roll(state: State, action: Action, force: bool = False):
    key = player_key(state, action.color)
    state.player_state[f"{key}_HAS_ROLLED"] = True

    if action.value is None and force:
        raise ValueError("Forced ROLL requires explicit dice tuple")
    dices = action.value or roll_dice()
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


def apply_discard(state: State, action: Action, force: bool = False):
    hand = player_deck_to_array(state, action.color)
    num_to_discard = len(hand) // 2
    if action.value is None:
        if force:
            raise ValueError("Forced DISCARD requires explicit discarded cards")
        # TODO: Forcefully discard randomly so that decision tree doesnt explode in possibilities.
        discarded = random.sample(hand, k=num_to_discard)
    else:
        discarded = action.value  # for replay functionality
    to_discard = freqdeck_from_listdeck(discarded)

    player_freqdeck_subtract(state, action.color, to_discard)
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, to_discard)
    action = Action(action.color, action.action_type, discarded)

    # Advance turn
    discarders_left = [
        player_num_resource_cards(state, color) > 7 for color in state.colors
    ][state.current_player_index + 1 :]
    if any(discarders_left):
        to_skip = discarders_left.index(True)
        state.current_player_index = state.current_player_index + 1 + to_skip
        # state.current_prompt stays the same
    else:
        state.current_player_index = state.current_turn_index
        state.current_prompt = ActionPrompt.MOVE_ROBBER
        state.is_discarding = False
        state.is_moving_knight = True

    state.playable_actions = generate_playable_actions(state)
    return action


def apply_move_robber(state: State, action: Action, force: bool = False):
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


def apply_steal(state: State, action: Action, force: bool = False):
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


def apply_play_knight_card(state: State, action: Action, force: bool = False):
    if not player_can_play_dev(state, action.color, "KNIGHT"):
        raise ValueError("Player cant play knight card now")

    play_dev_card(state, action.color, "KNIGHT")

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.MOVE_ROBBER
    state.playable_actions = generate_playable_actions(state)


def apply_play_year_of_plenty(state: State, action: Action, force: bool = False):
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


def apply_play_monopoly(state: State, action: Action, force: bool = False):
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


def apply_play_road_building(state: State, action: Action, force: bool = False):
    if not player_can_play_dev(state, action.color, "ROAD_BUILDING"):
        raise ValueError("Player cant play road building now")

    play_dev_card(state, action.color, "ROAD_BUILDING")
    state.is_road_building = True
    state.free_roads_available = 2

    # state.current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_maritime_trade(state: State, action: Action, force: bool = False):
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


def apply_offer_trade(state: State, action: Action, force: bool = False):
    # Multi-trade support: Add this trade to active_trades (don't cancel existing trades)
    offered = action.value[:5]
    wanted = action.value[5:10]
    # Any counts: how many "any" (wildcard) resources in offered/wanted side
    offered_any = action.value[10] if len(action.value) > 10 else 0
    wanted_any = action.value[11] if len(action.value) > 11 else 0

    state.active_trades[action.color] = {
        'offered': offered,
        'wanted': wanted,
        'offered_any': offered_any,
        'wanted_any': wanted_any,
        'acceptees': set(),
        'rejecters': set(),
    }

    # Update legacy fields for backward compatibility (legacy uses 11-tuple: 5 offered + 5 wanted + turn_index)
    state.is_resolving_trade = True
    state.current_trade = (*action.value[:10], state.current_turn_index)

    # Async trades: keep current_player_index on trade creator (like Colonist)
    # Other players can respond asynchronously
    # current_player_index stays the same
    state.current_prompt = ActionPrompt.PLAY_TURN

    state.playable_actions = generate_playable_actions(state)


def apply_accept_trade(state: State, action: Action, force: bool = False):
    # Multi-trade: action.value is the creator_color whose trade is being accepted
    creator_color = action.value
    acceptor_color = action.color

    if creator_color not in state.active_trades:
        raise ValueError(f"No active trade from {creator_color}")

    # A player's latest response replaces their previous response.
    state.active_trades[creator_color]['rejecters'].discard(acceptor_color)
    state.active_trades[creator_color]['acceptees'].add(acceptor_color)

    # Update legacy fields for backward compatibility
    # If this is the current_trade creator, update acceptees
    if state.current_trade[10] == state.colors.index(creator_color):
        index = state.colors.index(acceptor_color)
        new_acceptess = list(state.acceptees)
        new_acceptess[index] = True
        state.acceptees = tuple(new_acceptess)
        new_rejecters = list(state.rejecters)
        new_rejecters[index] = False
        state.rejecters = tuple(new_rejecters)

    # Async trades: don't advance player - stay on trade creator
    # Trade creator can now CONFIRM_TRADE with this acceptee
    # current_player_index stays the same
    # current_prompt stays PLAY_TURN

    state.playable_actions = generate_playable_actions(state)


def apply_reject_trade(state: State, action: Action, force: bool = False):
    # Multi-trade: action.value is the creator_color whose trade is being rejected
    creator_color = action.value
    rejector_color = action.color

    if creator_color not in state.active_trades:
        raise ValueError(f"No active trade from {creator_color}")

    # A player's latest response replaces their previous response.
    state.active_trades[creator_color]['acceptees'].discard(rejector_color)
    state.active_trades[creator_color]['rejecters'].add(rejector_color)

    # Update legacy fields for backward compatibility
    # If this is the current_trade creator, update rejecters
    if state.current_trade[10] == state.colors.index(creator_color):
        index = state.colors.index(rejector_color)
        new_rejecters = list(state.rejecters)
        new_rejecters[index] = True
        state.rejecters = tuple(new_rejecters)
        new_acceptees = list(state.acceptees)
        new_acceptees[index] = False
        state.acceptees = tuple(new_acceptees)

    # Async trades: just record the rejection, don't advance player
    # Stay on trade creator - they can continue playing or cancel the trade
    # current_player_index stays the same
    # current_prompt stays PLAY_TURN

    state.playable_actions = generate_playable_actions(state)


def apply_confirm_trade(state: State, action: Action, force: bool = False):
    # Multi-trade: action.color is trade creator, action.value is acceptee_color
    creator_color = action.color
    acceptee_color = action.value

    if creator_color not in state.active_trades:
        raise ValueError(f"No active trade from {creator_color}")

    trade_info = state.active_trades[creator_color]
    offering = trade_info['offered']
    asking = trade_info['wanted']

    if acceptee_color not in trade_info['acceptees']:
        raise ValueError(f"{acceptee_color} has not accepted this trade")
    if not freqdeck_contains(get_player_freqdeck(state, creator_color), offering):
        raise ValueError(f"{creator_color} can no longer afford this trade")
    if not freqdeck_contains(get_player_freqdeck(state, acceptee_color), asking):
        raise ValueError(f"{acceptee_color} can no longer afford this trade")

    # Execute the trade atomically after both affordability checks.
    player_freqdeck_subtract(state, creator_color, offering)
    player_freqdeck_add(state, creator_color, asking)
    player_freqdeck_subtract(state, acceptee_color, asking)
    player_freqdeck_add(state, acceptee_color, offering)

    # Remove this completed trade from active_trades
    del state.active_trades[creator_color]

    # Keep unrelated offers/counters and select a remaining legacy projection.
    sync_legacy_trade_state(state)

    # After confirming trade, return to the trade creator's turn
    state.current_player_index = state.colors.index(creator_color)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_counter_offer(state: State, action: Action, force: bool = False):
    """Handle a counter offer from a non-turn player.

    Counter-offers are proposals directed at the turn player. Other players can
    "accept" a counter-offer to indicate they'd also make the same deal, but only
    the turn player can execute (ACCEPT_COUNTER_OFFER) to complete the trade.

    action.value is the 10-tuple: (offered[5], wanted[5])
    """
    offered = action.value[:5]
    wanted = action.value[5:10]
    offered_any = action.value[10] if len(action.value) > 10 else 0
    wanted_any = action.value[11] if len(action.value) > 11 else 0

    state.counter_offers[action.color] = {
        'offered': offered,
        'wanted': wanted,
        'offered_any': offered_any,
        'wanted_any': wanted_any,
        'acceptees': set(),  # Others who'd make the same offer to turn player
    }
    state.is_resolving_trade = True

    if state.current_prompt == ActionPrompt.PLAY_TURN:
        # Async mode: stay on PLAY_TURN, regenerate actions
        # Turn player will see ACCEPT_COUNTER_OFFER in their playable actions
        state.playable_actions = generate_playable_actions(state)
        return

    # Legacy DECIDE_TRADE flow: cycle through players
    try:
        state.current_player_index = next(
            i
            for i, c in enumerate(state.colors)
            if c != action.color and i > state.current_player_index
        )
    except StopIteration:
        # All players have responded
        if sum(state.acceptees) == 0 and len(state.counter_offers) == 0:
            reset_trading_state(state)
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.PLAY_TURN
        elif len(state.counter_offers) > 0:
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.DECIDE_COUNTER_OFFERS
        else:
            state.current_player_index = state.current_turn_index
            state.current_prompt = ActionPrompt.DECIDE_ACCEPTEES

    state.playable_actions = generate_playable_actions(state)


def apply_join_counter_offer(state: State, action: Action, force: bool = False):
    """Non-turn player joins an existing counter-offer.

    This indicates they'd also make the same offer to the turn player.
    action.value is the color of the counter-offer creator to join.
    """
    counter_creator_color = action.value
    joiner_color = action.color

    if counter_creator_color not in state.counter_offers:
        raise ValueError(f"No counter offer from {counter_creator_color} to join")

    counter_offer = state.counter_offers[counter_creator_color]

    # Verify joiner has the resources to fulfill this offer
    joiner_freqdeck = get_player_freqdeck(state, joiner_color)
    if not freqdeck_contains(joiner_freqdeck, counter_offer['offered']):
        raise ValueError(f"{joiner_color} cannot afford to join this counter-offer")

    # Add joiner to acceptees
    counter_offer['acceptees'].add(joiner_color)

    state.playable_actions = generate_playable_actions(state)


def apply_accept_counter_offer(state: State, action: Action, force: bool = False):
    """Turn player accepts a counter offer, executing the trade.

    action.value can be:
    - A single color: trade with the counter-offer creator
    - A tuple (counter_creator_color, chosen_color): trade with chosen_color
      (who is either the creator or someone who accepted the counter)
    """
    # Parse action.value to determine who we're trading with
    if isinstance(action.value, tuple) and len(action.value) == 2:
        counter_creator_color, trade_partner_color = action.value
    else:
        counter_creator_color = action.value
        trade_partner_color = action.value  # Trade with the creator

    counter_offer = state.counter_offers.get(counter_creator_color)
    if counter_offer is None:
        raise ValueError(f"No counter offer from {counter_creator_color}")

    # Verify trade_partner is valid (either creator or an acceptee)
    valid_partners = {counter_creator_color} | counter_offer['acceptees']
    if trade_partner_color not in valid_partners:
        raise ValueError(f"{trade_partner_color} is not a valid trade partner for this counter")

    # Counter offer: creator offers counter_offer['offered'], wants counter_offer['wanted']
    # Turn player gives what they want, receives what they offer
    we_give = counter_offer['wanted']
    we_receive = counter_offer['offered']

    player_freqdeck_subtract(state, action.color, we_give)
    player_freqdeck_add(state, action.color, we_receive)
    player_freqdeck_subtract(state, trade_partner_color, we_receive)
    player_freqdeck_add(state, trade_partner_color, we_give)

    # Remove the completed counter-offer
    del state.counter_offers[counter_creator_color]

    sync_legacy_trade_state(state)

    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_cancel_trade(state: State, action: Action, force: bool = False):
    # Multi-trade: only cancel this player's trade
    if action.color in state.active_trades:
        del state.active_trades[action.color]

    # Keep unrelated offers/counters and select a remaining legacy projection.
    sync_legacy_trade_state(state)

    state.current_player_index = state.current_turn_index
    state.current_prompt = ActionPrompt.PLAY_TURN

    state.playable_actions = generate_playable_actions(state)
