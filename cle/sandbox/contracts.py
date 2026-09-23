"""Typed results and snapshots for sandbox orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from cle.game_engine.events import EngineTransition, GameEngineSnapshot, GameEvent, PlayerEvent
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation
from cle.players.contracts import PlayerAttempt, PlayerContext, _restore_contract_slots
from cle.players.data import PlayerSnapshot
from cle.sandbox.action_batches import AutomaticBatchAction, PendingActionBatch
from cle.sandbox.communication import CommunicationOpportunity
from cle.sandbox.trade_preauthorization import AutomaticTradeAction, TradePreauthorization


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_decision_attempts: int = 3

    def __post_init__(self) -> None:
        if self.max_decision_attempts < 1:
            raise ValueError("max_decision_attempts must be positive")


@dataclass(frozen=True, slots=True)
class SandboxStepResult:
    """One completed single-player or barrier sandbox step."""

    contexts: tuple[PlayerContext, ...]
    attempts: tuple[PlayerAttempt, ...]
    transitions: tuple[EngineTransition, ...]
    messages: tuple[GameEvent, ...] = ()
    automatic_action: AutomaticTradeAction | AutomaticBatchAction | None = None

    __setstate__ = _restore_contract_slots

    @property
    def context(self) -> PlayerContext | None:
        return self.contexts[0] if len(self.contexts) == 1 else None

    @property
    def before_revision(self) -> int:
        return self.transitions[0].before_revision

    @property
    def after_revision(self) -> int:
        return self.transitions[-1].after_revision if self.transitions else self.messages[-1].sequence + 1

    @property
    def winner(self) -> Color | None:
        return self.transitions[-1].winner


@dataclass(frozen=True, slots=True)
class SandboxView:
    revision: int
    observer: Color
    current_actor: Color
    turn_number: int
    phase: str
    observation: PlayerObservation
    events: tuple[PlayerEvent, ...]
    legal_actions: tuple[Action, ...]
    winner: Color | None


@dataclass(frozen=True, slots=True)
class SandboxSnapshot:
    engine: GameEngineSnapshot
    player_states: tuple[tuple[Color, PlayerSnapshot], ...]
    pending_decision_revision: int | None = None
    speech_used: bool = False
    speech_calls_remaining: int | None = None
    pending_reactions: tuple[CommunicationOpportunity, ...] = ()
    pre_robber_sequence: int | None = None
    trade_preauthorization: TradePreauthorization | None = None
    pending_action_batch: PendingActionBatch | None = None

    __setstate__ = _restore_contract_slots
