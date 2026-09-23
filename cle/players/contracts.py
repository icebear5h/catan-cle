"""Player-facing contracts for sandbox orchestration."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Mapping, Protocol, runtime_checkable

from cle.game_engine.communication import SocialCommitment
from cle.game_engine.events import PlayerEvent
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.game_engine.trading import TradeOffer
from cle.players.data import (
    AcceptanceResult,
    ActionCall,
    ContractSlot,
    JsonValue,
    PlayerSnapshot,
    RestoredContract,
)

if TYPE_CHECKING:
    from cle.harness.models import ModelRequest, ModelResponse


def _restore_contract_slots(self: RestoredContract, state: list[ContractSlot]) -> None:
    """Fill appended defaults when loading historical frozen/slotted pickles."""
    items = fields(self)
    if len(state) > len(items) or any(
        item.default is MISSING for item in items[len(state):]
    ):
        raise ValueError(f"Invalid stored {type(self).__name__} field count")
    values = [*state, *(item.default for item in items[len(state):])]
    for item, value in zip(items, values, strict=True):
        object.__setattr__(self, item.name, value)


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
    recent_messages: tuple[PlayerEvent, ...] = ()
    active_commitments: tuple[SocialCommitment, ...] = ()
    discard_count: int = 0
    visible_through_sequence: int | None = None
    visible_messages: tuple[PlayerEvent, ...] = ()
    speech_allowed: bool = False

    __setstate__ = _restore_contract_slots

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
    usage: tuple[tuple[str, JsonValue], ...] = ()
    latency_ms: int | None = None
    parse_warning: str | None = None
    native_reasoning: str = ""
    native_reasoning_details: tuple[JsonValue, ...] = ()
    reasoning_request: tuple[tuple[str, JsonValue], ...] = ()
    provider_response_id: str | None = None
    provider_request_id: str | None = None
    provider_native_finish_reason: str | None = None
    discard_cards: tuple[str, ...] | None = None
    knight_destination: tuple[int, int, int] | None = None
    notes_update: str | None = None

    # Proposer authorization, not responder willingness. None is a normal probe.
    confirm_if_accepted_by: tuple[Color, ...] | str | None = None

    # Complete admitted semantic envelope; only its first action uses action_index.
    batch_actions: tuple[ActionCall, ...] = ()

    __setstate__ = _restore_contract_slots


@dataclass(frozen=True, slots=True)
class PlayerAttempt:
    """One typed attempt returned by a player implementation."""

    context_id: str
    choice: PlayerChoice | CommunicationChoice | None
    validation_error: str | None = None
    model_request: ModelRequest | None = None
    model_response: ModelResponse | None = None


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
    active_commitments: tuple[SocialCommitment, ...] = ()
    observation: PlayerObservation | None = None
    visible_messages: tuple[PlayerEvent, ...] = ()
    trigger_reason: str | None = None

    __setstate__ = _restore_contract_slots


class CommunicationMode(str, Enum):
    SILENCE = "silence"
    SAY = "say"


@dataclass(frozen=True, slots=True)
class CommunicationChoice:
    mode: CommunicationMode = CommunicationMode.SILENCE
    text: str = ""
    audience: tuple[Color, ...] = ()
    commitment: CommitmentProposal | None = None
    model_request: ModelRequest | None = None
    model_response: ModelResponse | None = None
    notes_update: str | None = None
    validation_error: str | None = None
    # None retains historical audience semantics; reactive speech is always public.
    respondents: tuple[Color, ...] | None = None

    __setstate__ = _restore_contract_slots


@runtime_checkable
class SandboxPlayer(Protocol):
    color: Color

    @property
    def event_cursor(self) -> int:
        """Next unread event; advanced through acknowledge_events, not assignment."""

    async def choose(
        self,
        context: PlayerContext,
        feedback: str | None = None,
    ) -> PlayerAttempt:
        """Return one typed choice attempt."""

    async def communicate(self, context: TalkContext) -> CommunicationChoice:
        """Return speech or silence for a bounded communication opportunity."""

    def accept(self, attempt: PlayerAttempt, result: AcceptanceResult) -> None:
        """Record an attempt only after the engine accepted its action."""

    def acknowledge_events(self, next_sequence: int) -> None:
        """Advance this player's reaction cursor after processing a cutoff."""

    def status(self) -> Mapping[str, str | int]:
        """Return non-secret diagnostics."""

    def snapshot(self) -> PlayerSnapshot:
        """Return restorable private player state."""

    def restore(self, snapshot: PlayerSnapshot) -> None:
        """Restore private player state."""
