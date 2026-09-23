"""Seven handling: discards, robber movement, and stealing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.models.actions import generate_playable_actions, steal_possibilities
from cle.game_engine.models.decks import freqdeck_add, freqdeck_from_listdeck
from cle.game_engine.models.enums import (
    RESOURCES,
    Action,
    ActionPrompt,
    ActionType,
    FastResource,
)
from cle.game_engine.state_functions import (
    player_deck_draw,
    player_deck_random_draw,
    player_deck_replenish,
    player_deck_to_array,
    player_freqdeck_subtract,
    player_num_resource_cards,
    player_resource_freqdeck_contains,
)

if TYPE_CHECKING:
    from cle.game_engine.state.core import GameState


def validate_discard(
    state: GameState, action: Action, *, force: bool = False
) -> tuple[FastResource, ...]:
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
    # Every member was checked against RESOURCES above.
    discarded: tuple[FastResource, ...] = tuple(action.value)
    if not player_resource_freqdeck_contains(
        state, action.color, freqdeck_from_listdeck(discarded)
    ):
        raise ValueError("Cannot discard resource cards the player does not hold")
    return discarded


def apply_discard(state: GameState, action: Action, force: bool = False) -> Action:
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


def apply_move_robber(state: GameState, action: Action, force: bool = False) -> None:
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


def apply_steal(state: GameState, action: Action, force: bool = False) -> Action:
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
