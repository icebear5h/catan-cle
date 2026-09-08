"""Provider-independent messages, responses, and player sessions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from cle.harness.board_surface import BoardPresentation
from cle.players.contracts import PlayerChoice
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
class ModelRequest:
    decision_id: str
    session_id: str
    messages: tuple[ModelMessage, ...]
    components: tuple[PromptComponent, ...] = ()
    board_presentation: BoardPresentation | None = None


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
    choice: PlayerChoice
    after_revision: int


@dataclass(frozen=True)
class PlayerSessionSnapshot:
    color: Color
    session_id: str
    messages: tuple[ModelMessage, ...]
    strategic_memory: str
    event_cursor: int
    receipts: tuple[tuple[str, ChoiceReceipt], ...]


@dataclass
class PlayerSession:
    """Durable provider-independent continuity for one game player."""

    color: Color
    session_id: str
    messages: list[ModelMessage] = field(default_factory=list)
    strategic_memory: str = ""
    event_cursor: int = 0
    receipts: dict[str, ChoiceReceipt] = field(default_factory=dict)

    def snapshot(self) -> PlayerSessionSnapshot:
        return PlayerSessionSnapshot(
            color=self.color,
            session_id=self.session_id,
            messages=tuple(self.messages),
            strategic_memory=self.strategic_memory,
            event_cursor=self.event_cursor,
            receipts=deepcopy(tuple(self.receipts.items())),
        )

    def restore(self, snapshot: PlayerSessionSnapshot) -> None:
        if snapshot.color != self.color or snapshot.session_id != self.session_id:
            raise ValueError("Player-session snapshot identity does not match this player")
        self.messages = list(snapshot.messages)
        self.strategic_memory = snapshot.strategic_memory
        self.event_cursor = snapshot.event_cursor
        self.receipts = deepcopy(dict(snapshot.receipts))
