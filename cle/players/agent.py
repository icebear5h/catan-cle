"""LLM-backed sandbox player with durable provider-independent continuity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from cle.harness.board_surface import BoardPresenter
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    load_communication_suite,
    parse_communication_response,
)
from cle.harness.context import (
    ContextAssembler,
    PlayerResponseParseError,
    PlayerResponseParser,
)
from cle.harness.models import (
    ChoiceReceipt,
    CompletionTransport,
    receipt_choice,
    ModelMessage,
    ModelRequest,
    PlayerSession,
    PlayerSessionSnapshot,
    PromptSource,
)
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import ContextSuite
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerContext,
    TalkContext,
)
from cle.game_engine.models.player import Color
from cle.players.notes import validate_notes
from cle.players.validation import validate_communication_choice


@dataclass(frozen=True, slots=True)
class AgentPlayerSnapshot:
    session: PlayerSessionSnapshot


class AgentPlayer:
    """A player implemented by an async model transport and editable suite."""

    def __init__(
        self,
        color: Color,
        transport: CompletionTransport,
        *,
        session_id: str,
        suite: ContextSuite | None = None,
        communication_suite: CommunicationSuite | None = None,
        board_presenter: BoardPresenter | None = None,
        prompt_sources: tuple[PromptSource, ...] = (),
    ) -> None:
        self.color = color
        self.transport = transport
        self.prompt_sources = prompt_sources
        if suite is None:
            shared = load_shared_prompt_suite()
            suite = shared.decision_suite()
            if communication_suite is None:
                communication_suite = shared.communication_suite()
        self.suite = suite
        self.communication_suite = communication_suite or load_communication_suite()
        self.session = PlayerSession(
            color=color,
            session_id=session_id,
            context_policy=self.suite.context.memory_mode,
        )
        self._validate_context_policy()
        self._assembler = ContextAssembler(
            self.suite,
            board_presenter=board_presenter,
        )
        self._parser = PlayerResponseParser(self.suite)

    @property
    def event_cursor(self) -> int:
        return self.session.event_cursor

    @property
    def context_policy(self) -> str:
        return self.session.context_policy

    @property
    def max_notes_chars(self) -> int:
        return self.suite.context.max_notes_chars

    @property
    def reactive_speech(self) -> bool:
        return self.suite.context.reactive_speech

    def _validate_context_policy(self) -> None:
        policy = self.suite.context.memory_mode
        talk_policy = self.communication_suite.memory_mode
        if self.suite.context.reactive_speech != self.communication_suite.reactive_speech:
            raise ValueError("Decision and communication reactive-speech contracts must match")
        if "fresh_notes" in (policy, talk_policy) and (
            policy != talk_policy
            or self.suite.context.max_notes_chars != self.communication_suite.max_notes_chars
        ):
            raise ValueError("Fresh action and communication suites must match memory mode and notes limit")
        if policy != self.session.context_policy:
            raise ValueError("Player-session context policy does not match the selected suite")

    def _input_next_sequence(self, cutoff: int | None, channel: str) -> int:
        cursor = (
            self.session.action_next_sequence
            if channel == "action" else self.session.talk_next_sequence
        )
        if type(cutoff) is not int or cutoff < -1:
            raise ValueError("Fresh contexts require an explicit integer visibility cutoff >= -1")
        if cutoff + 1 < cursor:
            raise ValueError(f"Stale {channel} context visibility cutoff")
        return cutoff + 1

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        if context.actor != self.color:
            raise ValueError(
                f"Player {self.color} cannot answer a context for {context.actor}"
            )
        self._validate_context_policy()
        accepted = self.session.receipts.get(context.context_id)
        if accepted is not None:
            return PlayerAttempt(context_id=context.context_id, choice=deepcopy(accepted.choice))

        context = deepcopy(context)
        fresh = self.session.context_policy == "fresh_notes"
        if fresh:
            next_sequence = self._input_next_sequence(context.visible_through_sequence, "action")
            cursor = self.session.action_next_sequence
            messages = tuple(
                event for event in context.visible_messages
                if cursor <= event.sequence < next_sequence
            )
            context = replace(
                context,
                events=tuple(
                    event for event in context.events
                    if cursor <= event.sequence < next_sequence
                ),
                recent_messages=messages,
                visible_messages=messages,
            )
        model_request = self._assembler.assemble(context, self.session, feedback)
        model_request = replace(model_request, prompt_sources=self.prompt_sources)
        if fresh:
            model_request = replace(
                model_request,
                memory_revision=self.session.memory_revision,
                input_next_sequence=next_sequence,
                context_policy="fresh_notes",
                channel="action",
            )
        model_response = deepcopy(await self.transport.complete(deepcopy(model_request)))
        try:
            choice = self._parser.parse(context, model_response)
        except PlayerResponseParseError as exc:
            return PlayerAttempt(
                context_id=context.context_id,
                choice=None,
                validation_error=str(exc),
                model_request=model_request,
                model_response=model_response,
            )
        return PlayerAttempt(
            context_id=context.context_id,
            choice=choice,
            model_request=model_request,
            model_response=model_response,
        )

    async def communicate(self, context: Any) -> CommunicationChoice:
        if not isinstance(context, TalkContext):
            raise TypeError("AgentPlayer communication requires TalkContext")
        if context.player != self.color:
            raise ValueError(f"Player {self.color} cannot answer a context for {context.player}")
        self._validate_context_policy()
        context = deepcopy(context)
        fresh = self.session.context_policy == "fresh_notes"
        if fresh:
            next_sequence = self._input_next_sequence(context.visible_through_sequence, "talk")
            if context.observation is None:
                raise ValueError("Fresh communication requires a current observation")
            if context.cause.sequence >= next_sequence and not (
                context.cause.event_type == "PRE_ACTION"
                and context.cause.sequence == next_sequence
            ):
                raise ValueError("Communication trigger exceeds the visibility cutoff")
            cursor = self.session.talk_next_sequence
            messages = tuple(
                event for event in context.visible_messages
                if cursor <= event.sequence < next_sequence
            )
            context = replace(
                context,
                game_events=tuple(
                    event for event in context.game_events
                    if cursor <= event.sequence < next_sequence
                ),
                recent_messages=messages,
                visible_messages=messages,
            )
        request = build_communication_request(
            context,
            self.session.session_id,
            self.communication_suite,
            notes=self.session.strategic_memory,
            board_presenter=self._assembler.board_presenter,
        )
        request = replace(request, prompt_sources=self.prompt_sources)
        if fresh:
            request = replace(
                request,
                memory_revision=self.session.memory_revision,
                input_next_sequence=next_sequence,
                context_policy="fresh_notes",
                channel="talk",
            )
        response = deepcopy(await self.transport.complete(deepcopy(request)))
        choice = parse_communication_response(
            response,
            speaker=self.color,
            participants=context.participants,
            suite=self.communication_suite,
            instruction=next(
                (
                    component.template for component in request.components
                    if component.id == "environment.response_schema"
                ),
                "",
            ),
        )
        return replace(
            choice,
            model_request=request,
            model_response=response,
        )

    def validate_context_update(
        self,
        request: ModelRequest | None,
        context: PlayerContext | TalkContext,
    ) -> None:
        """Check request provenance before the sandbox mutates authoritative state."""
        self._validate_context_policy()
        if isinstance(context, PlayerContext):
            channel, actor = "action", context.actor
        elif isinstance(context, TalkContext):
            channel, actor = "talk", context.player
        else:
            raise ValueError("Context updates require the original PlayerContext or TalkContext")
        if actor != self.color:
            raise ValueError("Context belongs to another player")
        if self.session.context_policy == "legacy":
            if isinstance(request, ModelRequest) and request.context_policy not in (None, "legacy"):
                raise ValueError("Model request context policy does not match this player")
            return
        if not isinstance(request, ModelRequest):
            raise ValueError("Fresh context updates require their original ModelRequest")
        if request.context_policy != self.session.context_policy:
            raise ValueError("Model request context policy does not match this player")
        if request.session_id != self.session.session_id:
            raise ValueError("Model request session identity does not match this player")
        if not isinstance(request.decision_id, str) or not request.decision_id:
            raise ValueError("Model request requires a context identity")
        if request.decision_id != context.context_id:
            raise ValueError("Model request context identity does not match its context")
        if request.channel != channel:
            raise ValueError(f"Model request channel must be {channel} for this context")
        if type(request.memory_revision) is not int or request.memory_revision < 0:
            raise ValueError("Model request memory_revision must be a non-negative integer")
        if request.memory_revision != self.session.memory_revision:
            raise ValueError("Stale model request memory revision")
        if type(request.input_next_sequence) is not int or request.input_next_sequence < 0:
            raise ValueError("Model request input_next_sequence must be a non-negative integer")
        next_sequence = self._input_next_sequence(context.visible_through_sequence, channel)
        if request.input_next_sequence != next_sequence:
            raise ValueError("Model request visibility cutoff does not match its context")

    def accept(self, attempt: PlayerAttempt, result: Any) -> None:
        choice = attempt.choice
        if choice is None:
            raise ValueError("Cannot accept a player attempt without a choice")
        fresh = self.session.context_policy == "fresh_notes"
        context = getattr(result, "context", None)
        if fresh and not isinstance(context, PlayerContext):
            raise ValueError("Fresh action acceptance requires the original result.context PlayerContext")
        if fresh and attempt.context_id != context.context_id:
            raise ValueError("Action attempt does not match its context")
        if attempt.context_id in self.session.receipts:
            return
        notes = None
        if fresh:
            self.validate_context_update(attempt.model_request, context)
            if attempt.validation_error is not None:
                raise ValueError("Cannot accept an invalid player attempt")
            if choice.notes_update is not None:
                notes = validate_notes(choice.notes_update, self.suite.context.max_notes_chars)
        if not fresh and attempt.model_request is not None and attempt.model_response is not None:
            self.session.messages.extend(
                (
                    deepcopy(attempt.model_request.messages[-1]),
                    ModelMessage(
                        role="assistant",
                        content=attempt.model_response.content,
                    ),
                )
            )
        if not fresh and choice.game_plan:
            self.session.strategic_memory = choice.game_plan
        if not fresh and context is not None and context.events:
            self.session.event_cursor = max(
                self.session.event_cursor,
                context.events[-1].sequence + 1,
            )
        after_revision = getattr(result, "after_revision", None)
        if after_revision is None:
            transitions = getattr(result, "transitions", ())
            after_revision = transitions[-1].after_revision if transitions else 0
        receipt = ChoiceReceipt(
            choice=receipt_choice(choice),
            after_revision=after_revision,
        )
        if fresh:
            if notes is not None:
                self.session.strategic_memory = notes
            self.session.action_next_sequence = attempt.model_request.input_next_sequence
            self.session.memory_revision += 1
            self.acknowledge_events(attempt.model_request.input_next_sequence)
        self.session.receipts[attempt.context_id] = receipt

    def accept_communication(self, context: TalkContext, choice: CommunicationChoice) -> None:
        """Commit only admitted speech or explicit silence, never acquisition alone."""
        if not isinstance(context, TalkContext):
            raise ValueError("Communication acceptance requires its original TalkContext")
        if context.context_id in self.session.communication_receipts:
            return
        self._validate_context_policy()
        if self.session.context_policy == "legacy":
            return
        validate_communication_choice(
            choice,
            speaker=self.color,
            participants=context.participants,
            max_notes_chars=self.communication_suite.max_notes_chars,
        )
        request = choice.model_request
        self.validate_context_update(request, context)
        notes = (
            validate_notes(choice.notes_update, self.communication_suite.max_notes_chars)
            if choice.notes_update is not None else None
        )
        if notes is not None:
            self.session.strategic_memory = notes
        self.session.talk_next_sequence = request.input_next_sequence
        self.session.memory_revision += 1
        self.session.communication_receipts.add(context.context_id)
        self.acknowledge_events(request.input_next_sequence)

    def acknowledge_events(self, next_sequence: int) -> None:
        self.session.event_cursor = max(self.session.event_cursor, next_sequence)

    def status(self) -> dict[str, Any]:
        return {
            "kind": "agent",
            "color": self.color.value,
            "suite": f"{self.suite.id}@{self.suite.version}",
            "session_id": self.session.session_id,
            "messages": len(self.session.messages),
            "event_cursor": self.session.event_cursor,
            "accepted_choices": len(self.session.receipts),
            "context_policy": self.session.context_policy,
            "action_next_sequence": self.session.action_next_sequence,
            "talk_next_sequence": self.session.talk_next_sequence,
            "memory_revision": self.session.memory_revision,
            "accepted_communications": len(self.session.communication_receipts),
        }

    def snapshot(self) -> AgentPlayerSnapshot:
        return AgentPlayerSnapshot(session=self.session.snapshot())

    def validate_restore(self, snapshot: AgentPlayerSnapshot) -> None:
        """Preflight a full restore without touching this player's live session."""
        self._validate_context_policy()
        if not isinstance(snapshot, AgentPlayerSnapshot):
            raise ValueError("Agent restore requires an AgentPlayerSnapshot")
        candidate = PlayerSession(
            color=self.color,
            session_id=self.session.session_id,
            context_policy=self.session.context_policy,
        )
        candidate.restore(snapshot.session, max_notes_chars=self.suite.context.max_notes_chars)

    def restore(self, snapshot: AgentPlayerSnapshot) -> None:
        self.validate_restore(snapshot)
        self.session.restore(snapshot.session, max_notes_chars=self.suite.context.max_notes_chars)
