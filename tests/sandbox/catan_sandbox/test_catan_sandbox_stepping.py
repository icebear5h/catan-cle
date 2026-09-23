"""Menu use, purity, retries, and snapshot restore."""
import pickle
from typing import Any

import pytest

from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.models.player import Color
from cle.players.baseline import FirstLegalPlayer, ScriptedPlayer
from cle.players.contracts import (
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import CatanSandbox, RetryPolicy

from .support import COLORS, InvalidThenValidPlayer, _sandbox


def test_settlement_action_descriptions_do_not_guess_opponent_nearness() -> None:
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.step(engine.state.playable_actions[0])
    engine.step(engine.state.playable_actions[0])
    observation = engine.observe(Color.BLUE)
    settlement_actions = tuple(
        action
        for action in observation.valid_actions
        if action.action_type == ActionType.BUILD_SETTLEMENT
    )
    red_settlement = observation.opponent_settlements[Color.RED][0]

    assert engine.state.current_color() == Color.BLUE
    assert settlement_actions
    assert any(abs(action.value - red_settlement) <= 5 for action in settlement_actions)
    descriptions = [
        CatanObservationFormatter()._format_single_action(action, observation)
        for action in settlement_actions
    ]
    assert all("Near:" not in description for description in descriptions)


def test_view_is_pure_and_only_exposes_legal_menu_to_current_player() -> None:
    sandbox, _ = _sandbox()

    red_before = sandbox.view(Color.RED)
    blue_before = sandbox.view(Color.BLUE)
    red_after = sandbox.view(Color.RED)

    assert pickle.dumps(red_before) == pickle.dumps(red_after)
    assert red_before.observation.board_map is not red_after.observation.board_map
    assert sandbox.revision == 0
    assert red_before.legal_actions
    assert blue_before.legal_actions == ()
    assert red_before.events == ()
    assert blue_before.events == ()


@pytest.mark.asyncio
async def test_step_applies_sole_roll_without_asking_player() -> None:
    class NoChoosePlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> None:
            raise AssertionError("forced ROLL must not ask the player")

        async def communicate(self, context: TalkContext) -> None:
            raise AssertionError("forced ROLL must not ask for pre-action speech")

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = NoChoosePlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)
    player_snapshot = red.snapshot()

    result = await sandbox.step()

    assert [action.action_type for action in engine.state.actions] == [ActionType.ROLL]
    assert result.transitions[0].requested_action == Action(
        Color.RED,
        ActionType.ROLL,
        None,
    )
    assert result.transitions[0].resolved_action.action_type == ActionType.ROLL
    assert result.contexts == ()
    assert result.attempts == ()
    assert red.snapshot() == player_snapshot
    assert engine.state.last_dice_roll is not None


@pytest.mark.asyncio
async def test_step_asks_player_when_knight_is_legal_before_roll() -> None:
    class RollChoosingPlayer(FirstLegalPlayer):
        choose_calls = 0

        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            self.choose_calls += 1
            roll_index = next(
                index
                for index, action in enumerate(context.legal_actions)
                if action.action_type == ActionType.ROLL
            )
            return PlayerAttempt(
                context_id=context.context_id,
                choice=PlayerChoice(action_index=roll_index),
            )

    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    engine.state.is_initial_build_phase = False
    engine.state.current_prompt = ActionPrompt.PLAY_TURN
    engine.state.player_state["P0_KNIGHT_IN_HAND"] = 1
    engine.state.player_state["P0_KNIGHT_OWNED_AT_START"] = True
    engine.state.playable_actions = generate_playable_actions(engine.state)
    red = RollChoosingPlayer(Color.RED)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox = CatanSandbox(engine, players)

    assert {action.action_type for action in engine.state.playable_actions} == {
        ActionType.PLAY_KNIGHT_CARD,
        ActionType.ROLL,
    }

    result = await sandbox.step()

    assert red.choose_calls == 1
    assert red.accepted_choices == 1
    assert len(result.contexts) == 1
    assert len(result.attempts) == 1
    assert result.transitions[0].requested_action.action_type == ActionType.ROLL


@pytest.mark.asyncio
async def test_step_uses_exact_engine_menu_and_acknowledges_player() -> None:
    red = ScriptedPlayer(Color.RED, choices=[1])
    sandbox, _ = _sandbox(red)
    expected = sandbox.view(Color.RED).legal_actions[1]

    result = await sandbox.step()

    assert result.transitions[0].requested_action == expected
    assert result.transitions[0].resolved_action == expected
    assert red.accepted_choices == 1
    assert sandbox.revision == 1


@pytest.mark.asyncio
async def test_snapshot_restore_recovers_engine_events_rng_and_players() -> None:
    sandbox, players = _sandbox()
    await sandbox.step()
    snapshot = sandbox.snapshot()
    events = sandbox.view(Color.RED).events

    await sandbox.step()
    assert sandbox.revision == 2

    sandbox.restore(snapshot)

    assert sandbox.revision == 1
    assert sandbox.view(Color.RED).events == events
    assert players[Color.RED].accepted_choices == 1
    assert players[Color.RED].event_cursor == 0


@pytest.mark.asyncio
async def test_step_retries_out_of_menu_choice_without_mutating_engine() -> None:
    red = InvalidThenValidPlayer(Color.RED)
    engine = GameEngine(COLORS, seed=7, shuffle_players=False)
    players = {color: FirstLegalPlayer(color) for color in COLORS}
    players[Color.RED] = red
    sandbox: Any = CatanSandbox(
        engine,
        players,
        retry_policy=RetryPolicy(max_decision_attempts=2),
    )

    result = await sandbox.step()

    assert result.after_revision == 1
    assert red.attempts == 2
    assert len(sandbox.decision_trace) == 1
    assert "outside" in sandbox.decision_trace[0].validation_error
    assert len(engine.state.actions) == 1


def test_sandbox_hot_path_has_no_threading_locks_or_duplicate_event_buffer() -> None:
    sandbox, _ = _sandbox()

    assert not hasattr(sandbox, "_lock")
    assert not hasattr(sandbox, "_step_lock")
    assert not hasattr(sandbox, "_events")
    assert not hasattr(sandbox, "_accepted_decisions")
