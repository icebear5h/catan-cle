"""Typed results and snapshots for sandbox orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cle.players.contracts import PlayerAttempt, PlayerContext
from cle.game_engine.events import EngineTransition, GameEngineSnapshot, GameEvent, PlayerEvent
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation


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

    @property
    def context(self) -> PlayerContext | None:
        return self.contexts[0] if len(self.contexts) == 1 else None

    @property
    def before_revision(self) -> int:
        return self.transitions[0].before_revision

    @property
    def after_revision(self) -> int:
        return self.transitions[-1].after_revision

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
    player_states: tuple[tuple[Color, Any], ...]
