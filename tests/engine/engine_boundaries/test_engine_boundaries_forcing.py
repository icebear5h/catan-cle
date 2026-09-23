"""Forced-outcome and terminal-guard boundaries."""
import pickle

import pytest

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import (
    Action,
    ActionPrompt,
    ActionType,
)
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState


def test_caller_owned_forced_dice_cannot_mutate_resolved_state(engine: GameEngine) -> None:
    dice = [1, 2]
    transition = engine.step(Action(Color.RED, ActionType.ROLL, dice), force=True)
    dice[0] = 6

    assert engine.state.last_dice_roll == [1, 2]
    assert engine.state.actions[-1].value == [1, 2]
    assert transition.resolved_action.value == [1, 2]
    assert engine.events[-1].public_payload == [1, 2]


def test_rejected_forced_discard_does_not_capture_an_undo_entry(engine: GameEngine) -> None:
    roll = Action(Color.RED, ActionType.ROLL, (3, 4))
    engine.step(roll, force=True)
    before = pickle.dumps(engine)

    with pytest.raises(ValueError):
        engine.step(Action(Color.RED, ActionType.DISCARD, ("ORE",) * 4), force=True)

    assert pickle.dumps(engine) == before
    assert engine.undo() == roll


@pytest.mark.parametrize("validate_action", [True, False])
@pytest.mark.parametrize("action_type", [ActionType.ROLL, ActionType.END_TURN])
def test_terminal_guard_precedes_history_and_rng(engine: GameEngine, monkeypatch: pytest.MonkeyPatch, validate_action: bool, action_type: ActionType) -> None:
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    engine.state.player_state["P0_HAS_ROLLED"] = action_type == ActionType.END_TURN
    engine.state.playable_actions = generate_playable_actions(engine.state)
    action = Action(Color.RED, action_type, None)
    assert action in engine.state.playable_actions
    before = pickle.dumps(engine)

    def reject_state_copy(self: object, memo: dict[int, object]) -> None:
        pytest.fail("Terminal rejection must precede history capture")

    monkeypatch.setattr(GameState, "__deepcopy__", reject_state_copy, raising=False)
    with pytest.raises(ValueError, match="terminal"):
        engine.step(action, validate_action=validate_action)

    assert pickle.dumps(engine) == before
    assert engine.rng is engine.state.rng
    assert not engine.is_action_valid(action)


@pytest.mark.parametrize(
    "action",
    [
        Action(Color.RED, ActionType.ROLL, None),
        Action(Color.RED, ActionType.DISCARD, None),
        Action(Color.RED, ActionType.BUY_DEVELOPMENT_CARD, None),
        Action(Color.RED, ActionType.STEAL, (Color.BLUE, None)),
    ],
)
def test_terminal_force_still_requires_explicit_outcomes_before_history(engine: GameEngine, action: Action) -> None:
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    before = pickle.dumps(engine)

    with pytest.raises(ValueError, match="Forced"):
        engine.step(action, force=True)

    assert pickle.dumps(engine) == before


def test_terminal_force_explicit_bypass_publishes_without_rng_and_can_undo(engine: GameEngine) -> None:
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    action = Action(Color.RED, ActionType.ROLL, (1, 2))
    rng = engine.rng.getstate()
    assert not engine.is_action_valid(action)

    transition = engine.step(action, force=True)

    assert (transition.before_revision, transition.after_revision) == (0, 1)
    assert transition.resolved_action == action
    assert transition.winner == Color.RED
    assert engine.events[0].public_payload == (1, 2)
    assert engine.rng.getstate() == rng
    assert len(engine.history) == 1
    assert engine.undo() == action
    assert engine.events == []
    assert engine.state.actions == []
    assert engine.winning_color() == Color.RED
    assert engine.rng is engine.state.rng
    assert engine.rng.getstate() == rng


@pytest.mark.parametrize(
    "owner_points, responder_points, winner",
    [(9, 10, None), (10, 9, Color.RED), (10, 12, Color.RED)],
)
def test_winner_is_turn_owner_not_discarding_actor(engine: GameEngine, owner_points: int, responder_points: int, winner: Color | None) -> None:
    engine.state.current_player_index = 1
    engine.state.current_prompt = ActionPrompt.DISCARD
    engine.state.is_discarding = True
    engine.state.player_state["P0_ACTUAL_VICTORY_POINTS"] = owner_points
    engine.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = responder_points
    assert engine.state.current_color() == Color.BLUE
    assert engine.state.colors[engine.state.current_turn_index] == Color.RED

    assert engine.winning_color() == winner


def test_end_turn_declares_next_turn_owner_not_the_responding_actor(engine: GameEngine) -> None:
    engine.state.player_state["P1_ACTUAL_VICTORY_POINTS"] = engine.vps_to_win
    assert engine.winning_color() is None
    action = Action(Color.RED, ActionType.END_TURN, None)
    assert engine.is_action_valid(action)

    transition = engine.step(action)

    assert transition.resolved_action.color == Color.RED
    assert transition.winner == engine.winning_color() == Color.BLUE
    assert engine.state.player_state["P1_HAS_ROLLED"] is False
    assert engine.observe(Color.BLUE).valid_actions == []
