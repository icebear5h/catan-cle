"""Dice, resource production, and turn rotation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TYPE_CHECKING

from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.board import Board
from cle.game_engine.models.decks import (
    freqdeck_can_draw,
    freqdeck_replenish,
    freqdeck_subtract,
)
from cle.game_engine.models.enums import (
    CITY,
    RESOURCES,
    SETTLEMENT,
    Action,
    ActionPrompt,
    FastResource,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state.trade import reset_trading_state
from cle.game_engine.state_functions import (
    player_clean_turn,
    player_freqdeck_add,
    player_key,
    player_num_resource_cards,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cle.game_engine.state.core import GameState


def roll_dice(rng: random.Random | None = None) -> tuple[int, int]:
    """Yield two dice from the supplied game-local random stream."""
    source = rng if rng is not None else random.SystemRandom()
    return (source.randint(1, 6), source.randint(1, 6))


def yield_resources(
    board: Board, resource_freqdeck: Sequence[int], number: int
) -> tuple[dict[Color, list[int]], list[FastResource]]:
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
    intented_payout: defaultdict[Color, defaultdict[FastResource, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    resource_totals: defaultdict[FastResource, int] = defaultdict(int)
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
    depleted: list[FastResource] = []
    for resource in RESOURCES:
        total = resource_totals[resource]
        if not freqdeck_can_draw(resource_freqdeck, total, resource):
            depleted.append(resource)

    # build final data color => freqdeck structure
    payout: dict[Color, list[int]] = {}
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


def advance_turn(state: GameState, direction: int = 1) -> None:
    """Sets .current_player_index"""
    next_index = next_player_index(state, direction)
    state.current_player_index = next_index
    state.current_turn_index = next_index
    state.num_turns += 1


def next_player_index(state: GameState, direction: int = 1) -> int:
    return (state.current_player_index + direction) % len(state.colors)


def apply_end_turn(state: GameState, action: Action, force: bool = False) -> None:
    # Reset any pending trade state (like Colonist's auto-cancel on end turn)
    reset_trading_state(state)
    player_clean_turn(state, action.color)
    advance_turn(state)
    state.current_prompt = ActionPrompt.PLAY_TURN
    state.playable_actions = generate_playable_actions(state)


def apply_roll(state: GameState, action: Action, force: bool = False) -> Action:
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
