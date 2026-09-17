"""Provider-independent messages, responses, and player sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from cle.harness.board_surface import BoardPresentation
from cle.players.contracts import CommunicationChoice, PlayerChoice
from cle.players.notes import MAX_NOTES_CHARS, validate_notes
from cle.game_engine.models.player import Color


@dataclass(frozen=True)
class ModelMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class PromptComponent:
    """One authored prompt string plus its authoritative rendered value."""

    id: str
    channel: Literal["system", "environment"]
    template: str
    value: str
    rendered: str
    variables: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class PromptSource:
    kind: str
    id: str
    version: str
    sha256: str
    source: str


@dataclass(frozen=True)
class ModelRequest:
    decision_id: str
    session_id: str
    messages: tuple[ModelMessage, ...]
    components: tuple[PromptComponent, ...] = ()
    board_presentation: BoardPresentation | None = None
    memory_revision: int | None = None
    input_next_sequence: int | None = None
    context_policy: str | None = None
    channel: str | None = None
    prompt_sources: tuple[PromptSource, ...] = ()
    trigger_reason: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    content: str
    model: str | None = None
    usage: tuple[tuple[str, Any], ...] = ()
    latency_ms: int | None = None
    finish_reason: str | None = None
    native_reasoning: str = ""
    native_reasoning_details: tuple[Any, ...] = ()
    reasoning_request: tuple[tuple[str, Any], ...] = ()
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    provider_native_finish_reason: str | None = None
    provider_request_payload: Any = None
    provider_response_payload: Any = None


class CompletionTransport(Protocol):
    """Shared asynchronous completion transport used by agent players."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return one completion for a fully assembled request."""


@dataclass(frozen=True)
class ChoiceReceipt:
    choice: PlayerChoice | CommunicationChoice
    after_revision: int


# Provider diagnostics on a PlayerChoice. They are recorded once per call in
# the trace store's model_calls table; a receipt never reads them back.
_RECEIPT_TRACE_FIELDS: dict[str, Any] = {
    "raw_response": "",
    "native_reasoning": "",
    "native_reasoning_details": (),
    "reasoning_request": (),
    "usage": (),
}


def receipt_choice(
    choice: PlayerChoice | CommunicationChoice,
) -> PlayerChoice | CommunicationChoice:
    """Return a detached copy of a choice holding only what redelivery needs.

    Receipts exist so an already-accepted context can be answered again with
    the same decision. Every receipt lives in the session for the rest of the
    game and is re-pickled into each step's snapshot, so carrying the model's
    reasoning trace here made snapshots grow quadratically with game length.
    """
    if isinstance(choice, PlayerChoice):
        choice = replace(choice, **_RECEIPT_TRACE_FIELDS)
    return deepcopy(choice)


@dataclass(frozen=True)
class PlayerSessionSnapshot:
    color: Color
    session_id: str
    messages: tuple[ModelMessage, ...]
    strategic_memory: str
    event_cursor: int
    receipts: tuple[tuple[str, ChoiceReceipt], ...]
    action_next_sequence: int = 0
    talk_next_sequence: int = 0
    memory_revision: int = 0
    context_policy: str = "legacy"
    communication_receipts: tuple[str, ...] = ()


@dataclass
class PlayerSession:
    """Durable provider-independent continuity for one game player."""

    color: Color
    session_id: str
    messages: list[ModelMessage] = field(default_factory=list)
    strategic_memory: str = ""
    event_cursor: int = 0
    receipts: dict[str, ChoiceReceipt] = field(default_factory=dict)
    action_next_sequence: int = 0
    talk_next_sequence: int = 0
    memory_revision: int = 0
    context_policy: str = "legacy"
    communication_receipts: set[str] = field(default_factory=set)

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Old sessions have only the mixed reaction cursor, not channel coverage.
        self.__dict__.update(
            action_next_sequence=0,
            talk_next_sequence=0,
            memory_revision=0,
            context_policy="legacy",
            communication_receipts=set(),
        )
        self.__dict__.update(state)

    def snapshot(self) -> PlayerSessionSnapshot:
        return PlayerSessionSnapshot(
            color=self.color,
            session_id=self.session_id,
            messages=deepcopy(tuple(self.messages)),
            strategic_memory=self.strategic_memory,
            event_cursor=self.event_cursor,
            receipts=deepcopy(tuple(self.receipts.items())),
            action_next_sequence=self.action_next_sequence,
            talk_next_sequence=self.talk_next_sequence,
            memory_revision=self.memory_revision,
            context_policy=self.context_policy,
            communication_receipts=tuple(sorted(self.communication_receipts)),
        )

    def restore(
        self,
        snapshot: PlayerSessionSnapshot,
        *,
        max_notes_chars: int = MAX_NOTES_CHARS,
    ) -> None:
        if not isinstance(snapshot, PlayerSessionSnapshot):
            raise ValueError("Player-session restore requires a PlayerSessionSnapshot")
        if snapshot.color != self.color or snapshot.session_id != self.session_id:
            raise ValueError("Player-session snapshot identity does not match this player")
        if snapshot.context_policy not in ("legacy", "fresh_notes") or snapshot.context_policy != self.context_policy:
            raise ValueError("Player-session snapshot context policy does not match this player")
        for name in ("event_cursor", "action_next_sequence", "talk_next_sequence", "memory_revision"):
            value = getattr(snapshot, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"Player-session snapshot {name} must be a non-negative integer")
        if not isinstance(snapshot.strategic_memory, str):
            raise ValueError("Player-session snapshot strategic_memory must be a string")
        if snapshot.context_policy == "fresh_notes":
            validate_notes(snapshot.strategic_memory, max_notes_chars)
        if not isinstance(snapshot.messages, tuple) or any(
            not isinstance(message, ModelMessage)
            or message.role not in ("system", "user", "assistant")
            or not isinstance(message.content, str)
            for message in snapshot.messages
        ):
            raise ValueError("Player-session snapshot messages must contain ModelMessages")
        if not isinstance(snapshot.receipts, tuple) or any(
            not isinstance(item, tuple) or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], ChoiceReceipt)
            or not isinstance(item[1].choice, (PlayerChoice, CommunicationChoice))
            or type(item[1].after_revision) is not int or item[1].after_revision < 0
            for item in snapshot.receipts
        ):
            raise ValueError("Player-session snapshot receipts must contain (context ID, ChoiceReceipt) pairs")
        if not isinstance(snapshot.communication_receipts, tuple) or any(
            not isinstance(context_id, str) for context_id in snapshot.communication_receipts
        ):
            raise ValueError("Player-session snapshot communication_receipts must contain context IDs")
        # Stage every conversion and deep copy before changing any live field.
        messages = deepcopy(list(snapshot.messages))
        receipts = deepcopy(dict(snapshot.receipts))
        communication_receipts = set(snapshot.communication_receipts)
        if len(receipts) != len(snapshot.receipts) or len(communication_receipts) != len(snapshot.communication_receipts):
            raise ValueError("Player-session snapshot contains duplicate receipt context IDs")
        self.messages = messages
        self.strategic_memory = snapshot.strategic_memory
        self.event_cursor = snapshot.event_cursor
        self.receipts = receipts
        self.action_next_sequence = snapshot.action_next_sequence
        self.talk_next_sequence = snapshot.talk_next_sequence
        self.memory_revision = snapshot.memory_revision
        self.communication_receipts = communication_receipts
