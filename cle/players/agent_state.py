"""Synchronous provenance validation and committed agent-session updates."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from cle.game_engine.events import EngineTransition
from cle.harness.models import ChoiceReceipt, ModelMessage, ModelRequest, receipt_choice
from cle.players.contracts import CommunicationChoice, PlayerAttempt, PlayerContext, TalkContext
from cle.players.data import AcceptanceResult
from cle.players.notes import validate_notes
from cle.players.validation import validate_communication_choice

if TYPE_CHECKING:
    from cle.players.agent import AgentPlayer


def validate_context_policy(player: AgentPlayer) -> None:
    policy = player.suite.context.memory_mode
    talk_policy = player.communication_suite.memory_mode
    if player.suite.context.reactive_speech != player.communication_suite.reactive_speech:
        raise ValueError("Decision and communication reactive-speech contracts must match")
    if "fresh_notes" in (policy, talk_policy) and (
        policy != talk_policy
        or player.suite.context.max_notes_chars != player.communication_suite.max_notes_chars
    ):
        raise ValueError("Fresh action and communication suites must match memory mode and notes limit")
    if policy != player.session.context_policy:
        raise ValueError("Player-session context policy does not match the selected suite")


def input_next_sequence(player: AgentPlayer, cutoff: int | None, channel: str) -> int:
    cursor = (
        player.session.action_next_sequence
        if channel == "action" else player.session.talk_next_sequence
    )
    if type(cutoff) is not int or cutoff < -1:
        raise ValueError("Fresh contexts require an explicit integer visibility cutoff >= -1")
    if cutoff + 1 < cursor:
        raise ValueError(f"Stale {channel} context visibility cutoff")
    return cutoff + 1


def validate_context_update(
    player: AgentPlayer,
    request: ModelRequest | None,
    context: PlayerContext | TalkContext,
) -> None:
    """Check request provenance before the sandbox mutates authoritative state."""
    player._validate_context_policy()
    if isinstance(context, PlayerContext):
        channel, actor = "action", context.actor
    elif isinstance(context, TalkContext):
        channel, actor = "talk", context.player
    else:
        raise ValueError("Context updates require the original PlayerContext or TalkContext")
    if actor != player.color:
        raise ValueError("Context belongs to another player")
    if player.session.context_policy == "legacy":
        if isinstance(request, ModelRequest) and request.context_policy not in (None, "legacy"):
            raise ValueError("Model request context policy does not match this player")
        return
    if not isinstance(request, ModelRequest):
        raise ValueError("Fresh context updates require their original ModelRequest")
    if request.context_policy != player.session.context_policy:
        raise ValueError("Model request context policy does not match this player")
    if request.session_id != player.session.session_id:
        raise ValueError("Model request session identity does not match this player")
    if not isinstance(request.decision_id, str) or not request.decision_id:
        raise ValueError("Model request requires a context identity")
    if request.decision_id != context.context_id:
        raise ValueError("Model request context identity does not match its context")
    if request.channel != channel:
        raise ValueError(f"Model request channel must be {channel} for this context")
    if type(request.memory_revision) is not int or request.memory_revision < 0:
        raise ValueError("Model request memory_revision must be a non-negative integer")
    if request.memory_revision != player.session.memory_revision:
        raise ValueError("Stale model request memory revision")
    if type(request.input_next_sequence) is not int or request.input_next_sequence < 0:
        raise ValueError("Model request input_next_sequence must be a non-negative integer")
    next_sequence = player._input_next_sequence(context.visible_through_sequence, channel)
    if request.input_next_sequence != next_sequence:
        raise ValueError("Model request visibility cutoff does not match its context")


def accept(player: AgentPlayer, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
    choice = attempt.choice
    if choice is None:
        raise ValueError("Cannot accept a player attempt without a choice")
    fresh = player.session.context_policy == "fresh_notes"
    context: PlayerContext | None = getattr(result, "context", None)
    if fresh:
        if not isinstance(context, PlayerContext):
            raise ValueError("Fresh action acceptance requires the original result.context PlayerContext")
        if attempt.context_id != context.context_id:
            raise ValueError("Action attempt does not match its context")
    if attempt.context_id in player.session.receipts:
        return
    notes = None
    next_sequence = None
    if fresh:
        assert context is not None
        player.validate_context_update(attempt.model_request, context)
        if attempt.validation_error is not None:
            raise ValueError("Cannot accept an invalid player attempt")
        if choice.notes_update is not None:
            notes = validate_notes(choice.notes_update, player.suite.context.max_notes_chars)
        assert attempt.model_request is not None
        next_sequence = attempt.model_request.input_next_sequence
    if not fresh and attempt.model_request is not None and attempt.model_response is not None:
        player.session.messages.extend(
            (
                deepcopy(attempt.model_request.messages[-1]),
                ModelMessage(role="assistant", content=attempt.model_response.content),
            )
        )
    if not fresh:
        # Historical action suites require game_plan; retain their failure on a
        # speech choice rather than silently accepting an unsupported response.
        game_plan: str = getattr(choice, "game_plan")
        if game_plan:
            player.session.strategic_memory = game_plan
    if not fresh and context is not None and context.events:
        player.session.event_cursor = max(
            player.session.event_cursor,
            context.events[-1].sequence + 1,
        )
    after_revision: int | None = getattr(result, "after_revision", None)
    if after_revision is None:
        transitions: tuple[EngineTransition, ...] = getattr(result, "transitions", ())
        after_revision = transitions[-1].after_revision if transitions else 0
    receipt = ChoiceReceipt(choice=receipt_choice(choice), after_revision=after_revision)
    if fresh:
        assert next_sequence is not None
        if notes is not None:
            player.session.strategic_memory = notes
        player.session.action_next_sequence = next_sequence
        player.session.memory_revision += 1
        player.acknowledge_events(next_sequence)
    player.session.receipts[attempt.context_id] = receipt


def accept_communication(
    player: AgentPlayer, context: TalkContext, choice: CommunicationChoice,
) -> None:
    """Commit only admitted speech or explicit silence, never acquisition alone."""
    if not isinstance(context, TalkContext):
        raise ValueError("Communication acceptance requires its original TalkContext")
    if context.context_id in player.session.communication_receipts:
        return
    player._validate_context_policy()
    if player.session.context_policy == "legacy":
        return
    validate_communication_choice(
        choice,
        speaker=player.color,
        participants=context.participants,
        max_notes_chars=player.communication_suite.max_notes_chars,
    )
    request = choice.model_request
    player.validate_context_update(request, context)
    notes = (
        validate_notes(choice.notes_update, player.communication_suite.max_notes_chars)
        if choice.notes_update is not None else None
    )
    assert request is not None and request.input_next_sequence is not None
    if notes is not None:
        player.session.strategic_memory = notes
    player.session.talk_next_sequence = request.input_next_sequence
    player.session.memory_revision += 1
    player.session.communication_receipts.add(context.context_id)
    player.acknowledge_events(request.input_next_sequence)
