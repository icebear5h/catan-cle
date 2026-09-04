"""Player-facing contracts for sandbox orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.trading import TradeOffer


@dataclass(frozen=True, slots=True)
class PlayerContext:
    """Complete perspective-safe facts for one exact player choice."""

    context_id: str
    actor: Color
    turn_number: int
    phase: str
    observation: PlayerObservation
    events: tuple[PlayerEvent, ...]
    legal_actions: tuple[Action, ...]
    prompt_key: str

    def action_at(self, index: int) -> Action:
        if index < 0 or index >= len(self.legal_actions):
            raise IndexError(
                f"Action index {index} is outside 0..{len(self.legal_actions) - 1}"
            )
        return self.legal_actions[index]


@dataclass(frozen=True, slots=True)
class PlayerChoice:
    """A player's selected index plus optional durable plans and trace data."""

    action_index: int
    trade_offer: TradeOffer | None = None
    game_plan: str = ""
    # Snapshot compatibility only; new parsers never populate or render this field.
    rationale: str = ""
    raw_response: str = ""
    model: str | None = None
    usage: tuple[tuple[str, Any], ...] = ()
    latency_ms: int | None = None
    parse_warning: str | None = None
    native_reasoning: str = ""
    native_reasoning_details: tuple[Any, ...] = ()
    reasoning_request: tuple[tuple[str, Any], ...] = ()
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    provider_native_finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PlayerAttempt:
    """One typed attempt returned by a player implementation."""

    context_id: str
    choice: PlayerChoice | None
    validation_error: str | None = None
    model_request: Any = None
    model_response: Any = None


@dataclass(frozen=True, slots=True)
class CommitmentProposal:
    condition: str
    promise: str
    expires_turn: int


@dataclass(frozen=True, slots=True)
class TalkContext:
    context_id: str
    player: Color
    participants: tuple[Color, ...]
    cause: PlayerEvent
    visible_through_sequence: int
    game_events: tuple[PlayerEvent, ...]
    recent_messages: tuple[PlayerEvent, ...]
    active_commitments: tuple[Any, ...] = ()


class CommunicationMode(str, Enum):
    SILENCE = "silence"
    SAY = "say"


@dataclass(frozen=True, slots=True)
class CommunicationChoice:
    mode: CommunicationMode = CommunicationMode.SILENCE
    text: str = ""
    audience: tuple[Color, ...] = ()
    intent: str | None = None
    commitment: CommitmentProposal | None = None
    model_request: Any = None
    model_response: Any = None


@runtime_checkable
class SandboxPlayer(Protocol):
    color: Color
    event_cursor: int

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        """Return one typed choice attempt."""

    async def communicate(self, context: Any) -> CommunicationChoice:
        """Return speech or silence for a bounded communication opportunity."""

    def accept(self, attempt: PlayerAttempt, result: Any) -> None:
        """Record an attempt only after the engine accepted its action."""

    def acknowledge_events(self, next_sequence: int) -> None:
        """Advance this player's reaction cursor after processing a cutoff."""

    def status(self) -> Mapping[str, Any]:
        """Return non-secret diagnostics."""

    def snapshot(self) -> Any:
        """Return restorable private player state."""

    def restore(self, snapshot: Any) -> None:
        """Restore private player state."""
