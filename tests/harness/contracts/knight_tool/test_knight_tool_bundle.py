"""Canonical materialization and bundle execution."""
import pickle
from copy import deepcopy
from dataclasses import replace
from typing import Any, TypeVar

import pytest

from cle.game_engine.events import EngineTransition
from cle.game_engine.game import GameEngine
from cle.game_engine.json import action_from_json
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.harness.models import receipt_choice
from cle.players.contracts import (
    PlayerChoice,
)
from cle.players.validation import action_from_choice, choice_followup_action
from playground.game_viewer.live.reasoning_trace import build_live_reasoning_traces

from .support import KNIGHT, _choice, _knight_engine, _sandbox

_Copied = TypeVar("_Copied")


@pytest.mark.parametrize("rolled", [False, True])
def test_knight_materialization_is_pure_and_keeps_canonical_action(rolled: bool) -> None:
    engine, destination = _knight_engine(rolled=rolled)
    sandbox, _ = _sandbox(engine)
    context: Any = sandbox.decision_context()
    choice = _choice(engine, destination)
    before: Any = pickle.dumps((engine.snapshot(), context, choice))

    assert action_from_choice(context, choice) == KNIGHT
    assert choice_followup_action(context, choice) == Action(
        Color.RED,
        ActionType.MOVE_ROBBER,
        destination,
    )
    legacy = replace(choice, knight_destination=None)
    assert action_from_choice(context, legacy) == KNIGHT
    assert choice_followup_action(context, legacy) is None
    assert pickle.dumps((engine.snapshot(), context, choice)) == before
    assert action_from_json(["RED", "PLAY_KNIGHT_CARD", None]) == KNIGHT
    assert action_from_json(["RED", "MOVE_ROBBER", list(destination)]) == (
        choice_followup_action(context, choice)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rolled", [False, True])
@pytest.mark.parametrize("victims", [0, 1, 2])
async def test_bundle_matches_canonical_steps_and_defers_victim_rng(rolled: bool, victims: int, monkeypatch: pytest.MonkeyPatch) -> None:
    engine, destination = _knight_engine(rolled=rolled, victims=victims)
    choice: Any = _choice(engine, destination)
    sandbox, red = _sandbox(engine, [choice])
    canonical: Any = deepcopy(engine)
    move: Any = Action(Color.RED, ActionType.MOVE_ROBBER, destination)
    expected = (canonical.step(KNIGHT), canonical.step(move))
    initial_rng: Any = engine.rng.getstate()
    step_calls: list[tuple[bool, Action]] = []
    original_step = GameEngine.step

    def strict_step(
        self: GameEngine, action: Action, validate_action: bool = True, force: bool = False
    ) -> EngineTransition:
        assert validate_action is True
        assert force is False
        step_calls.append((self is engine, action))
        return original_step(self, action, validate_action=validate_action, force=force)

    monkeypatch.setattr(GameEngine, "step", strict_step)
    result: Any = await sandbox.step()

    assert step_calls == [(False, KNIGHT), (False, move), (True, KNIGHT), (True, move)]
    assert result.transitions == expected
    assert engine.events == canonical.events
    assert engine.state.actions == canonical.state.actions == [KNIGHT, move]
    assert engine.state.player_state == canonical.state.player_state
    assert engine.state.board.robber_coordinate == destination
    assert engine.rng.getstate() == canonical.rng.getstate() == initial_rng
    assert len(red.calls) == len(red.accepted) == 1
    assert len(result.contexts) == len(result.attempts) == 1
    assert result.before_revision == 0
    assert result.after_revision == 2
    assert red.session.receipts[result.context.context_id].after_revision == 2
    assert red.session.receipts[result.context.context_id].choice == receipt_choice(choice)
    assert engine.state.player_state["P0_KNIGHT_IN_HAND"] == 0
    assert engine.state.player_state["P0_PLAYED_KNIGHT"] == 1

    (trace,) = build_live_reasoning_traces(sandbox, result)
    assert trace["action_type"] == "PLAY_KNIGHT_CARD"
    assert trace["action_sequence"] == [str(KNIGHT), str(move)]
    assert trace["knight_destination"] == list(destination)
    assert trace["native_reasoning"] == choice.native_reasoning
    assert trace["provider_response_id"] == choice.provider_response_id

    saved = pickle.loads(pickle.dumps(sandbox.snapshot()))
    saved_result: Any = pickle.loads(pickle.dumps(result))
    assert saved_result.attempts[0].choice.knight_destination == destination
    assert saved_result.transitions == expected
    sandbox.restore(saved)
    receipt: Any = red.session.receipts[result.context.context_id]
    assert receipt.choice.knight_destination == destination
    assert receipt.after_revision == 2

    if victims:
        context = sandbox.decision_context()
        assert engine.state.current_prompt == ActionPrompt.STEAL
        assert len(context.legal_actions) == victims
        assert all(action.action_type == ActionType.STEAL for action in context.legal_actions)
        steal = next(action for action in context.legal_actions if action.value[0] == Color.BLUE)
        assert steal.value == (Color.BLUE, None)
        red.choices.append(PlayerChoice(context.legal_actions.index(steal)))
        expected_steal = canonical.step(steal)
        stolen: Any = await sandbox.step()
        assert stolen.transitions == (expected_steal,)
        assert engine.rng.getstate() == canonical.rng.getstate() != initial_rng
        assert len(red.calls) == len(red.accepted) == 2
        assert red.calls[-1][0].events[-1].event_type == "MOVE_ROBBER"
        assert stolen.attempts[0].choice.knight_destination is None
    else:
        assert engine.state.current_prompt == ActionPrompt.PLAY_TURN
    assert engine.state.player_state["P0_HAS_ROLLED"] is rolled


@pytest.mark.asyncio
async def test_legacy_knight_and_other_choices_do_not_stage_engine_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    engine, destination = _knight_engine(rolled=True)
    sandbox, red = _sandbox(
        engine, [replace(_choice(engine, destination), knight_destination=None)]
    )
    copies: list[GameEngine] = []

    def track_copy(value: _Copied) -> _Copied:
        if isinstance(value, GameEngine):
            copies.append(value)
        return deepcopy(value)

    monkeypatch.setattr("cle.sandbox.catan.deepcopy", track_copy)
    result = await sandbox.step()
    assert len(result.transitions) == 1
    assert engine.state.current_prompt == ActionPrompt.MOVE_ROBBER
    assert engine.state.board.robber_coordinate != destination
    move = Action(Color.RED, ActionType.MOVE_ROBBER, destination)
    red.choices.append(PlayerChoice(engine.state.playable_actions.index(move)))
    await sandbox.step()
    end: Any = Action(Color.RED, ActionType.END_TURN, None)
    red.choices.append(PlayerChoice(engine.state.playable_actions.index(end)))
    await sandbox.step()
    assert copies == []
    assert len(red.calls) == len(red.accepted) == 3
