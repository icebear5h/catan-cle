"""Typed player/checkpoint compatibility at the legacy and sandbox boundaries."""

import pickle
from collections import deque
from dataclasses import dataclass, fields

import pytest

from cle.game_engine.events import EngineTransition
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.suite import load_context_suite
from cle.players.agent import AgentPlayer, AgentPlayerSnapshot
from cle.players.baseline import FirstLegalPlayer, HumanPlayer, ScriptedPlayer
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    SandboxPlayer,
    TalkContext,
    _restore_contract_slots,
)
from cle.players.data import AcceptanceResult, ContractSlot, PlayerSnapshot
from cle.sandbox.contracts import SandboxStepResult
from cle.sandbox.decision import build_decision_context

COLORS = (Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE)


@dataclass
class RecordedTransport:
    responses: deque[ModelResponse]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self.responses.popleft()


@dataclass(frozen=True)
class ContextResult:
    context: PlayerContext


@dataclass(frozen=True)
class TransitionResult(ContextResult):
    transitions: tuple[EngineTransition, ...]


def test_players_share_a_typed_snapshot_protocol_without_changing_pickle_paths() -> None:
    agent = AgentPlayer(Color.RED, RecordedTransport(deque()), session_id="contracts:RED")
    players: tuple[SandboxPlayer, ...] = (
        FirstLegalPlayer(Color.RED), ScriptedPlayer(Color.RED, (PlayerChoice(0), 1)),
        HumanPlayer(Color.RED), agent,
    )
    for player in players:
        saved = player.snapshot()
        restored: PlayerSnapshot = pickle.loads(pickle.dumps(saved))
        player.acknowledge_events(7)
        player.restore(restored)
        assert player.event_cursor == 0
        assert player.snapshot() == saved
    assert isinstance(agent.snapshot(), AgentPlayerSnapshot)
    assert AgentPlayerSnapshot.__module__ == "cle.players.agent"


def test_legacy_acceptance_retains_revision_fallback_and_duplicate_receipts() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    context = build_decision_context(engine)
    transition = engine.step(context.action_at(0))
    attempt = PlayerAttempt(context.context_id, PlayerChoice(0, game_plan="keep the accepted plan"))
    results: tuple[tuple[AcceptanceResult, int], ...] = (
        (ContextResult(context), 0),
        (TransitionResult(context, (transition,)), transition.after_revision),
        (SandboxStepResult((context,), (attempt,), (transition,)), transition.after_revision),
    )
    for result, expected_revision in results:
        player = AgentPlayer(
            Color.RED, RecordedTransport(deque()), session_id="legacy:RED", suite=load_context_suite(),
        )
        player.accept(attempt, result)
        saved = player.snapshot()
        player.accept(attempt, result)
        assert player.snapshot() == saved
        assert player.session.receipts[context.context_id].after_revision == expected_revision
        assert player.session.strategic_memory == "keep the accepted plan"


def test_legacy_speech_is_still_rejected_without_an_action_receipt() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    context = build_decision_context(engine)
    player = AgentPlayer(
        Color.RED, RecordedTransport(deque()), session_id="legacy:RED", suite=load_context_suite(),
    )
    before = player.snapshot()
    with pytest.raises(AttributeError, match="game_plan"):
        player.accept(PlayerAttempt(context.context_id, CommunicationChoice()), ContextResult(context))
    assert player.snapshot() == before


def test_historical_player_slots_keep_appended_defaults() -> None:
    engine = GameEngine(COLORS, seed=9, shuffle_players=False)
    context = build_decision_context(engine)
    transition = engine.step(context.action_at(0))
    talk = TalkContext(
        "talk:RED", Color.RED, COLORS, engine.project_events(Color.RED)[0],
        transition.after_revision - 1, engine.project_events(Color.RED), (),
    )
    cases: tuple[tuple[PlayerContext | TalkContext | CommunicationChoice, str], ...] = (
        (context, "speech_allowed"), (talk, "trigger_reason"), (CommunicationChoice(), "respondents"),
    )
    for original, appended_field in cases:
        state: list[ContractSlot] = [getattr(original, item.name) for item in fields(original)[:-1]]
        restored = type(original).__new__(type(original))
        _restore_contract_slots(restored, state)
        assert restored == original
        assert getattr(restored, appended_field) == getattr(original, appended_field)
        # CatanMap retains identity equality; a pickle roundtrip creates a new
        # board. Check the restored contract identity/default separately.
        roundtrip = pickle.loads(pickle.dumps(restored))
        assert type(roundtrip) is type(original)
        assert getattr(roundtrip, appended_field) == getattr(original, appended_field)
