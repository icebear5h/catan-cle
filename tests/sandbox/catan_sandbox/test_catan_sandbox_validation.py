"""Attempt shape validation and typed exhaustion."""
import pickle
from dataclasses import replace
from typing import Any

import pytest

from cle.game_engine.models.player import Color
from cle.harness.models import ModelResponse
from cle.players.baseline import FirstLegalPlayer
from cle.players.contracts import (
    CommitmentProposal,
    CommunicationChoice,
    CommunicationMode,
    PlayerAttempt,
    PlayerChoice,
    PlayerContext,
    TalkContext,
)
from cle.sandbox import RetryPolicy
from cle.sandbox.catan import PlayerResponseError, PostActionCommunicationError
from cle.sandbox.communication import CommunicationOpportunity, ReactionReason
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox

from .support import FixedTransport, NoCommunicationPolicy, _sandbox


@pytest.mark.asyncio
@pytest.mark.parametrize("returned", [None, {"action_index": 0}, PlayerAttempt("wrong-context", PlayerChoice(0))])
async def test_malformed_attempt_exhaustion_is_typed_and_does_not_mutate(returned: dict[str, int] | PlayerAttempt | None) -> None:
    calls = []

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            calls.append(feedback)
            return returned

    sandbox, players = _sandbox(InvalidPlayer(Color.RED))
    sandbox.retry_policy = RetryPolicy(2)
    sandbox.communication_policy = NoCommunicationPolicy()
    sandbox.decision_context()
    before = pickle.dumps(sandbox.snapshot())

    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()

    assert len(calls) == 2 and calls[0] is None and calls[1]
    assert len(caught.value.attempts) == 2
    assert all(isinstance(attempt, PlayerAttempt) and attempt.choice is None for attempt in caught.value.attempts)
    assert all(attempt.context_id.endswith(":0:RED") for attempt in sandbox.decision_trace)
    assert pickle.dumps(sandbox.snapshot()) == before
    assert players[Color.RED].accepted_choices == 0


@pytest.mark.asyncio
async def test_callback_exceptions_are_not_retried_as_malformed_returns() -> None:
    class BrokenPlayer(FirstLegalPlayer):
        calls = 0

        async def choose(self, context: PlayerContext, feedback: str | None=None) -> None:
            self.calls += 1
            raise TypeError("player implementation failed")

    sandbox, players = _sandbox(BrokenPlayer(Color.RED))
    with pytest.raises(TypeError, match="player implementation failed"):
        await sandbox.step()
    assert players[Color.RED].calls == 1
    assert sandbox.revision == 0
    assert sandbox.decision_trace == []


@pytest.mark.asyncio
async def test_attempt_shape_validation_precedes_copying_invalid_choice() -> None:
    class NotAChoice:
        def __deepcopy__(self: object, memo: dict[int, object]) -> None:
            raise AssertionError("Invalid player choices must not be copied")

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            return PlayerAttempt(context.context_id, NotAChoice())

    sandbox, _ = _sandbox(InvalidPlayer(Color.RED))
    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()
    assert "PlayerChoice" in caught.value.validation_error
    assert sandbox.revision == 0
    assert all(attempt.choice is None for attempt in caught.value.attempts)


@pytest.mark.asyncio
async def test_rejected_exception_attempts_do_not_alias_internal_trace() -> None:
    response: Any = ModelResponse(content="bad", native_reasoning_details=({"text": "original"},))

    class InvalidPlayer(FirstLegalPlayer):
        async def choose(self, context: PlayerContext, feedback: str | None=None) -> PlayerAttempt:
            return PlayerAttempt(context.context_id, PlayerChoice(999), model_response=response)

    sandbox, _ = _sandbox(InvalidPlayer(Color.RED))
    sandbox.retry_policy = RetryPolicy(1)
    with pytest.raises(PlayerResponseError) as caught:
        await sandbox.step()
    caught.value.attempts[0].model_response.native_reasoning_details[0]["text"] = "edited exception"
    response.native_reasoning_details[0]["text"] = "edited source"
    assert sandbox.decision_trace[0].model_response.native_reasoning_details == ({"text": "original"},)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [
    None,
    {"mode": "say"},
    CommunicationChoice(mode="say", text="Trade?", audience=(Color.RED,)),
    CommunicationChoice(mode=CommunicationMode.SAY, text={}, audience=(Color.RED,)),
    CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=[Color.RED]),
    CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,), commitment={}),
    *[
        CommunicationChoice(mode=CommunicationMode.SAY, text="Trade?", audience=(Color.RED,), commitment=proposal)
        for proposal in (
            CommitmentProposal("condition", "promise", "tomorrow"),
            CommitmentProposal("condition", "promise", True),
            CommitmentProposal("condition", "promise", -1),
            CommitmentProposal("condition", "promise", 1.5),
            CommitmentProposal("", "promise", 3),
            CommitmentProposal("condition", {}, 3),
        )
    ],
])
async def test_malformed_speech_has_rejected_admission_and_cannot_poison_engine(
    choice: CommunicationChoice,
) -> None:
    class BrokenSpeaker(FirstLegalPlayer):
        async def communicate(self, context: TalkContext) -> CommunicationChoice:
            return choice

    sandbox, _ = _sandbox()
    sandbox.register_player(BrokenSpeaker(Color.BLUE))
    with pytest.raises(PostActionCommunicationError) as caught:
        await sandbox.step()
    assert isinstance(caught.value.__cause__, ValueError)
    assert sandbox.revision == 1
    assert sandbox.game_engine.commitments == []
    assert sandbox.game_engine.project_messages(Color.RED) == ()
    rejected = sandbox.communication_trace[0]
    assert isinstance(rejected.choice, CommunicationChoice)
    assert rejected.accepted is False and rejected.validation_error
    assert sandbox.players[Color.BLUE].event_cursor == 0
    sandbox.communication_policy = NoCommunicationPolicy()
    await sandbox.step()
    assert sandbox.players[Color.RED].accepted_choices == 2


def test_talk_context_filters_cutoff_before_message_window_and_commitments() -> None:
    sandbox, _ = _sandbox()
    engine: Any = sandbox.game_engine
    engine.communication_limits = replace(engine.communication_limits, recent_message_window=2)
    for index in range(5):
        engine.append_message(
            speaker=Color.BLUE, text=f"private-{index}", audience=(Color.RED,),
            causation_id=f"talk:{index}", commitment=("condition", f"promise-{index}", 9),
        )
    cause = engine.project_events(Color.RED)[1]
    opportunity = CommunicationOpportunity(Color.RED, cause, 1, ReactionReason.TRADE, 0)
    context = sandbox._talk_context(opportunity)
    assert [event.payload["text"] for event in context.recent_messages] == ["private-0", "private-1"]
    assert [item.promise for item in context.active_commitments] == ["promise-0", "promise-1"]
    context.recent_messages[0].payload["text"] = "edited"
    context.active_commitments[0].promise = "edited"
    assert engine.project_events(Color.RED)[0].payload["text"] == "private-0"
    assert engine.commitments[0].promise == "promise-0"


@pytest.mark.asyncio
async def test_random_factory_ignores_transport_and_does_not_resolve_prompt_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Random mode must not load prompts or create a provider")

    monkeypatch.setattr("cle.sandbox.factory.materialize_live_prompt_suites", forbidden)
    monkeypatch.setattr("cle.sandbox.factory.create_text_transport", forbidden)
    transport = FixedTransport([])
    sandbox = create_live_sandbox(
        LiveSandboxConfig(mode="random", seed=7, palette="canonical_four"), transport=transport
    )
    await sandbox.step()
    assert all(type(player) is FirstLegalPlayer for player in sandbox.players.values())
    assert transport.requests == []
