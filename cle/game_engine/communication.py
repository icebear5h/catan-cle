"""Bounded social communication state recorded in the canonical game log."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cle.game_engine.models.player import Color


@dataclass(frozen=True, slots=True)
class CommunicationLimits:
    recent_message_window: int = 12
    max_general_reaction_rounds: int = 2
    max_trade_reaction_rounds: int = 3
    max_messages_per_player_round: int = 1
    max_messages_per_window: int = 12

    def __post_init__(self) -> None:
        if min(
            self.recent_message_window,
            self.max_general_reaction_rounds,
            self.max_trade_reaction_rounds,
            self.max_messages_per_player_round,
            self.max_messages_per_window,
        ) < 1:
            raise ValueError("Communication limits must be positive")


class CommitmentStatus(str, Enum):
    ACTIVE = "active"
    FULFILLED = "fulfilled"
    VIOLATED = "violated"
    EXPIRED = "expired"


@dataclass(slots=True)
class SocialCommitment:
    id: str
    proposer: Color
    audience: tuple[Color, ...]
    condition: str
    promise: str
    created_sequence: int
    expires_turn: int
    source_message_sequence: int
    status: CommitmentStatus = CommitmentStatus.ACTIVE

    @property
    def active(self) -> bool:
        return self.status == CommitmentStatus.ACTIVE
