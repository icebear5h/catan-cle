"""Reactive speech budget and durable bounded opportunity queue."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cle.game_engine.events import GameEvent
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color
from cle.players.contracts import CommunicationChoice
from cle.sandbox import catan
from cle.sandbox.communication import CommunicationOpportunity, ReactionReason

if TYPE_CHECKING:
    from . import CatanSandbox


def _reactive(self: CatanSandbox, color: Color) -> bool:
    return bool(getattr(self.players[color], "reactive_speech", False))


def _start_speech_budget(self: CatanSandbox) -> None:
    if self._speech_calls_remaining is None:
        self._speech_calls_remaining = self.game_engine.communication_limits.max_messages_per_window


def _append_speech(
    self: CatanSandbox, speaker: Color, choice: CommunicationChoice, causation_id: str,
) -> GameEvent:
    proposal = choice.commitment
    return catan.deepcopy(self.game_engine.append_message(
        speaker=speaker, text=choice.text, audience=choice.audience,
        causation_id=causation_id,
        respondents=choice.respondents,
        commitment=(proposal.condition, proposal.promise, proposal.expires_turn) if proposal else None,
    ))


def _dice_total(event: GameEvent) -> int:
    """Total a roll event's dice pair, refusing every other payload shape.

    A malformed payload is a corrupt event, not a quiet miss, so it fails loudly
    here exactly as summing a non-iterable payload always has.
    """
    payload = event.public_payload
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        first, second = payload
        if isinstance(first, int) and isinstance(second, int):
            return first + second
    raise ValueError(
        f"{event.event_type} payload must be two integer dice, got "
        f"{type(payload).__name__}: {payload!r}"
    )


def _open_pre_robber_window(self: CatanSandbox) -> None:
    actions = self.game_engine.state.playable_actions
    if not actions or any(action.action_type != ActionType.MOVE_ROBBER for action in actions):
        return
    # Most recent cause distinguishes a seven from historical split Knights.
    cause = next((event for event in reversed(self.game_engine.events) if event.event_type in {
        ActionType.ROLL.value, ActionType.PLAY_KNIGHT_CARD.value, ActionType.MOVE_ROBBER.value,
    }), None)
    if cause is None or cause.event_type != ActionType.ROLL.value:
        return
    if _dice_total(cause) != 7:
        return
    if cause.sequence == self._pre_robber_sequence:
        return
    self._pre_robber_sequence = cause.sequence
    self._start_speech_budget()
    opportunities = []
    for color in self.game_engine.state.colors:
        if color != self.current_actor() and self._reactive(color):
            projected = catan.project_event(cause, color)
            assert projected is not None
            opportunities.append(CommunicationOpportunity(
                color, projected, self.revision - 1, ReactionReason.PRE_ROBBER, 0,
            ))
    self._pending_reactions += tuple(opportunities)


async def _run_reactive(self: CatanSandbox) -> tuple[GameEvent, ...]:
    """Durable bounded queue; successful replies are removed before another await."""
    emitted: list[GameEvent] = []
    limits = self.game_engine.communication_limits
    while self._pending_reactions:
        self._refresh_inference_policy()
        self._start_speech_budget()
        assert self._speech_calls_remaining is not None
        if self._speech_calls_remaining <= 0:
            self._pending_reactions = ()
            break
        opportunity = self._pending_reactions[0]
        player = self.players[opportunity.player]
        session = getattr(player, "session", None)
        received = max(getattr(session, "action_next_sequence", 0), getattr(session, "talk_next_sequence", 0))
        if opportunity.round >= limits.max_general_reaction_rounds or (
            opportunity.reason == ReactionReason.ADDRESSED_SPEECH and opportunity.cause.sequence < received
        ):
            self._pending_reactions = self._pending_reactions[1:]
            continue
        # A failed provider call consumes its slot too, so retries cannot reset a window.
        self._speech_calls_remaining -= 1
        messages = await self._run_communication((opportunity,), max_rounds=1)
        self._pending_reactions = self._pending_reactions[1:]
        emitted.extend(messages)
        self._pending_reactions += self.communication_policy.addressed(
            self.game_engine, messages, round_number=opportunity.round + 1,
        )
    return tuple(emitted)
