"""Deterministic broad communication triggers for sandbox orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cle.game_engine.events import GameEvent, PlayerEvent, project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import ActionType
from cle.game_engine.models.player import Color


class ReactionReason(str, Enum):
    PRE_ACTION = "pre_action"
    TRADE = "trade"
    ROBBER = "robber"
    MAJOR_BUILD = "major_build"
    TURN_CHANGE = "turn_change"


@dataclass(frozen=True, slots=True)
class CommunicationOpportunity:
    player: Color
    cause: PlayerEvent
    visible_through_sequence: int
    reason: ReactionReason
    round: int


class CommunicationPolicy:
    """Shared stateless trigger policy; silence remains the common response."""

    _TRADE_TYPES = {
        ActionType.OFFER_TRADE.value,
        ActionType.ACCEPT_TRADE.value,
        ActionType.REJECT_TRADE.value,
        ActionType.COUNTER_OFFER.value,
        ActionType.CONFIRM_TRADE.value,
        ActionType.CANCEL_TRADE.value,
    }
    _ROBBER_TYPES = {
        ActionType.ROLL.value,
        ActionType.MOVE_ROBBER.value,
        ActionType.STEAL.value,
        ActionType.PLAY_KNIGHT_CARD.value,
    }
    _BUILD_TYPES = {
        ActionType.BUILD_SETTLEMENT.value,
        ActionType.BUILD_CITY.value,
        ActionType.BUILD_ROAD.value,
    }

    def pre_action(self, engine: GameEngine) -> tuple[CommunicationOpportunity, ...]:
        observation = engine.observe(engine.state.current_color())
        if observation.current_phase != "main_game":
            return ()
        actor = engine.state.current_color()
        cause = PlayerEvent(
            sequence=engine.revision,
            causation_id=f"pre-action:{engine.revision}",
            actor=actor,
            event_type="PRE_ACTION",
            payload=None,
        )
        return (
            CommunicationOpportunity(
                player=actor,
                cause=cause,
                visible_through_sequence=engine.revision - 1,
                reason=ReactionReason.PRE_ACTION,
                round=0,
            ),
        )

    def after_events(
        self,
        engine: GameEngine,
        events: tuple[GameEvent, ...],
        *,
        round_number: int,
    ) -> tuple[CommunicationOpportunity, ...]:
        opportunities = []
        for event in events:
            reason = self._reason(event)
            if reason is None:
                continue
            recipients = self._recipients(engine, event)
            for color in recipients:
                projected = project_event(event, color)
                if projected is None:
                    continue
                opportunities.append(
                    CommunicationOpportunity(
                        player=color,
                        cause=projected,
                        visible_through_sequence=event.sequence,
                        reason=reason,
                        round=round_number,
                    )
                )
        return tuple(opportunities)

    def _reason(self, event: GameEvent) -> ReactionReason | None:
        if event.event_type in self._TRADE_TYPES:
            return ReactionReason.TRADE
        if event.event_type in self._ROBBER_TYPES:
            if event.event_type != ActionType.ROLL.value:
                return ReactionReason.ROBBER
            dice = event.public_payload
            if isinstance(dice, tuple) and sum(dice) == 7:
                return ReactionReason.ROBBER
            return None
        if event.event_type in self._BUILD_TYPES:
            return ReactionReason.MAJOR_BUILD
        if event.event_type == ActionType.END_TURN.value:
            return ReactionReason.TURN_CHANGE
        return None

    @staticmethod
    def _recipients(engine: GameEngine, event: GameEvent) -> tuple[Color, ...]:
        return tuple(color for color in engine.state.colors if color != event.actor)
