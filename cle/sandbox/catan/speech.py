"""Concurrent communication acquisition and ordered admission with receipts."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from cle.game_engine.events import GameEvent
from cle.game_engine.models.player import Color
from cle.players.contracts import CommunicationChoice, CommunicationMode, TalkContext
from cle.players.notes import MAX_NOTES_CHARS
from cle.sandbox import catan
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    ReactionReason,
)
from cle.sandbox.contracts import SandboxStepResult

from .support import PostActionCommunicationCancelled, PostActionCommunicationError

if TYPE_CHECKING:
    from . import CatanSandbox


async def _post_action_communication(
    self: CatanSandbox, result: SandboxStepResult, *, max_rounds: int,
) -> SandboxStepResult:
    messages: list[GameEvent] = []
    try:
        await self._run_communication(
            tuple(item for item in self.communication_policy.after_events(
                self.game_engine,
                tuple(event for transition in result.transitions for event in transition.events),
                round_number=0,
            ) if not self._reactive(item.player)),
            max_rounds=max_rounds,
            emitted=messages,
        )
    except asyncio.CancelledError as exc:
        raise PostActionCommunicationCancelled(
            catan.replace(result, messages=(*result.messages, *catan.deepcopy(messages)))
        ) from exc
    except Exception as exc:
        raise PostActionCommunicationError(
            catan.replace(result, messages=(*result.messages, *catan.deepcopy(messages)))
        ) from exc
    return catan.replace(result, messages=(*result.messages, *catan.deepcopy(messages)))


async def _run_communication(
    self: CatanSandbox,
    initial: tuple[CommunicationOpportunity, ...],
    *,
    max_rounds: int,
    emitted: list[GameEvent] | None = None,
) -> tuple[GameEvent, ...]:
    opportunities = catan.deepcopy(initial)
    if emitted is None:
        emitted = []
    round_number = 0
    limits = self.game_engine.communication_limits
    while opportunities and round_number < max_rounds:
        self._refresh_inference_policy()
        opportunities = tuple(item for item in opportunities if not self._reactive(item.player) or item.reason in {
            ReactionReason.PRE_ROBBER, ReactionReason.ADDRESSED_SPEECH,
        })
        if not opportunities:
            break
        revision = self.revision
        cutoff = max(item.visible_through_sequence for item in opportunities)
        by_player: dict[Color, CommunicationOpportunity] = {}
        for opportunity in opportunities:
            by_player.setdefault(
                opportunity.player,
                catan.replace(
                    opportunity,
                    visible_through_sequence=(
                        revision - 1
                        if getattr(self.players[opportunity.player], "context_policy", "legacy") == "fresh_notes"
                        else cutoff
                    ),
                ),
            )
        ordered = tuple(
            by_player[color] for color in self.game_engine.state.colors if color in by_player
        )
        contexts = tuple(self._talk_context(item) for item in ordered)
        # Retain completed replies even if a sibling fails before admission.
        acquired: dict[Color, CommunicationChoice] = {}
        invalid: dict[Color, str] = {}

        async def communicate(
            item: CommunicationOpportunity, context: TalkContext,
        ) -> CommunicationChoice:
            returned = await self.players[item.player].communicate(catan.deepcopy(context))
            try:
                catan.validate_communication_choice(
                    returned, speaker=item.player, participants=context.participants,
                    max_notes_chars=getattr(self.players[item.player], "max_notes_chars", MAX_NOTES_CHARS),
                )
                self._validate_context_update(self.players[item.player], returned.model_request, context)
            except ValueError as exc:
                invalid[item.player] = str(exc)
            try:
                choice = (
                    catan.deepcopy(returned)
                    if isinstance(returned, CommunicationChoice)
                    else CommunicationChoice()
                )
            except TypeError as exc:
                invalid[item.player] = f"Communication choice cannot be detached: {exc}"
                choice = CommunicationChoice()
            acquired[item.player] = choice
            return choice

        try:
            choices = await catan._gather_or_cancel(
                communicate(item, context) for item, context in zip(ordered, contexts)
            )
            self._check_revision(revision)
        except BaseException as exc:
            self.communication_trace.extend(
                CommunicationAdmission(
                    item,
                    acquired[item.player],
                    accepted=False,
                    validation_error=f"Communication withheld: {str(exc) or type(exc).__name__}",
                )
                for item in ordered
                if item.player in acquired
            )
            raise
        round_events = []
        failure: Exception | None = None
        withheld_error = None
        for opportunity, context, choice in zip(ordered, contexts, choices):
            validation_error = withheld_error
            if validation_error is None and opportunity.player not in invalid:
                try:
                    self._validate_context_update(self.players[opportunity.player], choice.model_request, context)
                except ValueError as exc:
                    invalid[opportunity.player] = str(exc)
            if validation_error is None and opportunity.player in invalid:
                validation_error = invalid[opportunity.player]
                failure = ValueError(validation_error)
                withheld_error = (
                    f"Communication withheld after {opportunity.player.value} "
                    f"rejection: {validation_error}"
                )
            if validation_error is None and choice.mode != CommunicationMode.SILENCE:
                if len(emitted) >= limits.max_messages_per_window:
                    validation_error = "Communication withheld: message limit reached"
                else:
                    commitment = (
                        (choice.commitment.condition, choice.commitment.promise, choice.commitment.expires_turn)
                        if choice.commitment is not None else None
                    )
                    try:
                        event = self.game_engine.append_message(
                            speaker=opportunity.player,
                            text=choice.text,
                            audience=choice.audience,
                            causation_id=opportunity.cause.causation_id,
                            commitment=commitment,
                            respondents=choice.respondents,
                        )
                    except Exception as exc:
                        failure = exc
                        validation_error = str(exc) or type(exc).__name__
                        withheld_error = (
                            f"Communication withheld after {opportunity.player.value} "
                            f"rejection: {validation_error}"
                        )
                    else:
                        round_events.append(catan.deepcopy(event))
                        emitted.append(catan.deepcopy(event))
                        if self._pending_decision_revision is not None or self._reactive(opportunity.player):
                            self._pending_decision_revision = self.revision
            self.communication_trace.append(CommunicationAdmission(
                opportunity, choice, accepted=validation_error is None, validation_error=validation_error,
            ))
            if validation_error is None:
                accept_communication = getattr(self.players[opportunity.player], "accept_communication", None)
                if accept_communication is not None:
                    accept_communication(catan.deepcopy(context), catan.deepcopy(choice))
                self.players[opportunity.player].acknowledge_events(opportunity.visible_through_sequence + 1)
        if failure is not None:
            raise failure
        round_number += 1
        opportunities = self.communication_policy.after_events(
            self.game_engine, tuple(round_events), round_number=round_number,
        )
    return tuple(emitted)


def _talk_context(self: CatanSandbox, opportunity: CommunicationOpportunity) -> TalkContext:
    color = opportunity.player
    cutoff = opportunity.visible_through_sequence
    visible = tuple(
        event for event in self.game_engine.project_events(color) if event.sequence <= cutoff
    )
    return catan.deepcopy(TalkContext(
        context_id=(
            f"{self.game_engine.id}:talk:{opportunity.cause.causation_id}:"
            f"{opportunity.round}:{color.value}"
            + (f":{opportunity.cause.sequence}" if self._reactive(color) else "")
        ),
        player=color,
        participants=self.game_engine.state.colors,
        cause=opportunity.cause,
        visible_through_sequence=opportunity.visible_through_sequence,
        game_events=tuple(event for event in visible if event.event_type != "MESSAGE_SENT"),
        recent_messages=tuple(
            event for event in visible if event.event_type == "MESSAGE_SENT"
        )[-self.game_engine.communication_limits.recent_message_window:],
        active_commitments=tuple(
            item for item in self.game_engine.active_commitments(color)
            if item.created_sequence <= cutoff and item.source_message_sequence <= cutoff
        ),
        observation=(
            self.game_engine.observe(color)
            if getattr(self.players[color], "context_policy", "legacy") == "fresh_notes"
            else None
        ),
        visible_messages=tuple(event for event in visible if event.event_type == "MESSAGE_SENT"),
        trigger_reason=opportunity.reason.value,
    ))
