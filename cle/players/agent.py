"""LLM-backed sandbox player with durable provider-independent continuity."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace

from cle.game_engine.models.player import Color
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
    CompletionTransport,
    ModelRequest,
    PlayerSession,
    PlayerSessionSnapshot,
    PromptSource,
)
from cle.harness.shared_suite import load_shared_prompt_suite
from cle.harness.suite import ContextSuite
from cle.players import agent_state
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerContext,
    TalkContext,
)
from cle.players.data import AcceptanceResult, PlayerSnapshot, PlayerStatus


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
        # A shared decision suite must never pair with legacy communication_v5.
        if suite is None or (communication_suite is None and suite.context.mode == "shared"):
            shared = load_shared_prompt_suite()
            bundled = shared.decision_suite()
            if suite is not None and suite != bundled:
                raise ValueError(
                    "A shared decision suite outside the built-in bundle needs its "
                    "bundle's communication_suite"
                )
            suite = bundled
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
        agent_state.validate_context_policy(self)

    def _input_next_sequence(self, cutoff: int | None, channel: str) -> int:
        return agent_state.input_next_sequence(self, cutoff, channel)

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

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
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
        agent_state.validate_context_update(self, request, context)

    def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
        agent_state.accept(self, attempt, result)

    def accept_communication(self, context: TalkContext, choice: CommunicationChoice) -> None:
        """Commit only admitted speech or explicit silence, never acquisition alone."""
        agent_state.accept_communication(self, context, choice)

    def acknowledge_events(self, next_sequence: int) -> None:
        self.session.event_cursor = max(self.session.event_cursor, next_sequence)

    def status(self) -> PlayerStatus:
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

    def validate_restore(self, snapshot: PlayerSnapshot) -> None:
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

    def restore(self, snapshot: PlayerSnapshot) -> None:
        self.validate_restore(snapshot)
        assert isinstance(snapshot, AgentPlayerSnapshot)
        self.session.restore(snapshot.session, max_notes_chars=self.suite.context.max_notes_chars)
