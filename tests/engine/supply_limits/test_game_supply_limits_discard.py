"""Discard validation, sequencing, and bank conservation."""

from collections.abc import Callable, Iterable, Sequence

import pytest

from cle.game_engine.models.enums import (
    RESOURCES,
    WOOD,
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import (
    apply_action,
    validate_discard,
)
from cle.game_engine.state_functions import get_player_freqdeck

from .support import COLORS, assert_resource_conservation, make_game, transfer_from_bank_to_player


def test_forced_discard_cannot_return_unheld_cards_to_bank() -> None:
    game = make_game()
    transfer_from_bank_to_player(game, Color.RED, [8, 0, 0, 0, 0])
    bank_before = list(game.state.resource_freqdeck)

    with pytest.raises(ValueError, match="does not hold"):
        game.step(
            Action(
                Color.RED,
                ActionType.DISCARD,
                [RESOURCES[1]] * 4,
            ),
            force=True,
        )

    assert game.state.resource_freqdeck == bank_before
    assert_resource_conservation(game)
    assert game.revision == 0


@pytest.mark.parametrize(
    "cards",
    [None, "WOOD", {"WOOD": 4}, (4, 0, 0, 0, 0), ("WOOD",) * 3,
     ("WOOD",) * 5, ("ORE",) * 4, ("GOLD",) * 4, (True,) * 4, ([],) * 4],
)
def test_exact_discard_validation_rejects_invalid_payloads_without_mutation(
    cards: object,
) -> None:
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [5, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    before = state.copy()
    action = Action(Color.RED, ActionType.DISCARD, cards)
    with pytest.raises(ValueError):
        validate_discard(state, action)
    if cards is not None:
        with pytest.raises(ValueError):
            apply_action(state, action)
    assert state.player_state == before.player_state
    assert state.resource_freqdeck == before.resource_freqdeck
    assert state.actions == before.actions
    assert state.current_prompt == before.current_prompt
    assert state.rng.getstate() == before.rng.getstate()


@pytest.mark.parametrize("container", [tuple, list])
def test_exact_discard_returns_an_owned_immutable_multiset_without_rng_use(
    container: Callable[[Iterable[str]], Sequence[str]],
) -> None:
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [5, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    rng = state.rng.getstate()
    action = Action(Color.RED, ActionType.DISCARD, container([WOOD] * 4))
    assert validate_discard(state, action) == (WOOD,) * 4
    result = apply_action(state, action)
    assert result.value == (WOOD,) * 4
    assert get_player_freqdeck(state, Color.RED) == [1, 4, 0, 0, 0]
    assert state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert state.rng.getstate() == rng
    assert_resource_conservation(game)


@pytest.mark.parametrize("reason", ["seat", "prompt", "action-type", "participant", "threshold"])
def test_discard_request_validation_is_strict_before_randomness_or_application(reason: str) -> None:
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [8, 0, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    actor, kind = Color.RED, ActionType.DISCARD
    if reason == "seat":
        state.current_player_index = 1
    elif reason == "prompt":
        state.current_prompt = ActionPrompt.PLAY_TURN
    elif reason == "action-type":
        kind = ActionType.END_TURN
    elif reason == "participant":
        actor = Color.BLACK
    else:
        state.discard_limit = 8
    before = state.copy()
    with pytest.raises(ValueError):
        validate_discard(state, Action(actor, kind, (WOOD,) * 4))
    if kind == ActionType.DISCARD:
        for value in ((WOOD,) * 4, None):
            with pytest.raises(ValueError):
                apply_action(state, Action(actor, kind, value))
    assert state.rng.getstate() == before.rng.getstate()
    assert state.player_state == before.player_state
    assert state.resource_freqdeck == before.resource_freqdeck
    assert state.actions == before.actions


def test_explicit_auto_discard_remains_seeded_and_force_requires_exact_cards() -> None:
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.RED, [4, 4, 0, 0, 0])
    state.current_prompt = ActionPrompt.DISCARD
    state.is_discarding = True
    branch = state.copy()
    auto = Action(Color.RED, ActionType.DISCARD, None)
    with pytest.raises(ValueError, match="Forced DISCARD"):
        apply_action(state, auto, force=True)
    assert apply_action(state, auto) == apply_action(branch, auto)
    assert sum(get_player_freqdeck(state, Color.RED)) == 4
    assert_resource_conservation(game)


def test_forced_discard_keeps_card_checks_but_bypasses_seat_and_prompt() -> None:
    game = make_game()
    state = game.state
    transfer_from_bank_to_player(game, Color.BLUE, [8, 0, 0, 0, 0])
    action = Action(Color.BLUE, ActionType.DISCARD, (WOOD,) * 4)
    with pytest.raises(ValueError, match="not requested"):
        validate_discard(state, action)
    assert validate_discard(state, action, force=True) == (WOOD,) * 4
    apply_action(state, action, force=True)
    assert get_player_freqdeck(state, Color.BLUE) == [4, 0, 0, 0, 0]
    assert_resource_conservation(game)


@pytest.mark.parametrize("limit", [5, 7, 10])
def test_discard_sequence_visits_each_eligible_seat_once_then_returns_to_turn_owner(limit: int) -> None:
    game = make_game()
    state = game.state
    state.discard_limit = limit
    state.current_player_index = state.current_turn_index = 2
    for index, color in enumerate(COLORS):
        cards = [0] * len(RESOURCES)
        cards[index] = 18
        transfer_from_bank_to_player(game, color, cards)
    apply_action(state, Action(Color.WHITE, ActionType.ROLL, (3, 4)), force=True)
    for index, color in enumerate(COLORS):
        assert state.current_color() == color
        assert state.current_prompt == ActionPrompt.DISCARD
        assert state.current_turn_index == 2
        apply_action(state, Action(color, ActionType.DISCARD, (RESOURCES[index],) * 9))
    assert state.current_color() == Color.WHITE
    assert state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert not state.is_discarding
    assert all(sum(get_player_freqdeck(state, color)) == 9 for color in COLORS)
    assert state.num_turns == 0
    assert_resource_conservation(game)
