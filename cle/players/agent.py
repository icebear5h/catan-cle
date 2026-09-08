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
    ModelMessage,
    PlayerSession,
    PlayerSessionSnapshot,
)
from cle.harness.suite import ContextSuite, load_context_suite
from cle.players.contracts import (
    CommunicationChoice,
    PlayerAttempt,
    PlayerContext,
    TalkContext,
)
from cle.game_engine.models.player import Color


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
    ) -> None:
        self.color = color
        self.transport = transport
        self.suite = suite or load_context_suite()
        self.communication_suite = communication_suite or load_communication_suite()
        self.session = PlayerSession(color=color, session_id=session_id)
        self._assembler = ContextAssembler(
            self.suite,
            board_presenter=board_presenter,
        )
        self._parser = PlayerResponseParser(self.suite)

    @property
    def event_cursor(self) -> int:
        return self.session.event_cursor

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        if context.actor != self.color:
            raise ValueError(
                f"Player {self.color} cannot answer a context for {context.actor}"
            )
        accepted = self.session.receipts.get(context.context_id)
        if accepted is not None:
            return PlayerAttempt(context_id=context.context_id, choice=deepcopy(accepted.choice))

        context = deepcopy(context)
        model_request = self._assembler.assemble(context, self.session, feedback)
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
        request = build_communication_request(
            context,
            self.session.session_id,
            self.communication_suite,
        )
        response = deepcopy(await self.transport.complete(deepcopy(request)))
        choice = parse_communication_response(
            response,
            speaker=self.color,
            participants=context.participants,
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

    def accept(self, attempt: PlayerAttempt, result: Any) -> None:
        choice = attempt.choice
        if choice is None:
            raise ValueError("Cannot accept a player attempt without a choice")
        if attempt.context_id in self.session.receipts:
            return
        if attempt.model_request is not None and attempt.model_response is not None:
            self.session.messages.extend(
                (
                    deepcopy(attempt.model_request.messages[-1]),
                    ModelMessage(
                        role="assistant",
                        content=attempt.model_response.content,
                    ),
                )
            )
        if choice.game_plan:
            self.session.strategic_memory = choice.game_plan
        context = getattr(result, "context", None)
        if context is not None and context.events:
            self.session.event_cursor = max(
                self.session.event_cursor,
                context.events[-1].sequence + 1,
            )
        after_revision = getattr(result, "after_revision", None)
        if after_revision is None:
            transitions = getattr(result, "transitions", ())
            after_revision = transitions[-1].after_revision if transitions else 0
        self.session.receipts[attempt.context_id] = ChoiceReceipt(
            choice=deepcopy(choice),
            after_revision=after_revision,
        )

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
        }

    def snapshot(self) -> AgentPlayerSnapshot:
        return AgentPlayerSnapshot(session=self.session.snapshot())

    def restore(self, snapshot: AgentPlayerSnapshot) -> None:
        self.session.restore(snapshot.session)
