"""
Contains GameEngine class which is a thin-wrapper around the GameState class.
"""

from __future__ import annotations

import copy
import random
import sys
import uuid
from collections.abc import Sequence

from cle.game_engine.communication import (
    CommitmentStatus,
    CommunicationLimits,
    SocialCommitment,
)
from cle.game_engine.events import (
    EngineTransition,
    GameEngineSnapshot,
    GameEvent,
    HistoryEntry,
    PlayerEvent,
    PrivateOverlays,
    event_from_action,
    project_event,
)
from cle.game_engine.game.messaging import append_message
from cle.game_engine.game.persistence import copy_engine, restore_engine, snapshot_engine
from cle.game_engine.game.validation import can_fund_offer, is_valid_action, is_valid_trade
from cle.game_engine.models.enums import Action
from cle.game_engine.models.map import CatanMap
from cle.game_engine.models.player import Color
from cle.game_engine.observation import PlayerObservation, observe_state
from cle.game_engine.state import GameState, apply_action, assert_forced_action_is_explicit
from cle.game_engine.state_functions import player_key
from cle.game_engine.trading import TradeLimits

__all__ = ["GameEngine", "can_fund_offer", "is_valid_action", "is_valid_trade"]


class GameEngine:
    """Deterministic Catan rules and mutable materialized game state."""

    def __init__(
        self,
        colors: Sequence[Color],
        seed: int | None = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: CatanMap | None = None,
        initialize: bool = True,
        shuffle_players: bool = True,
        capture_history: bool = False,
        trade_limits: TradeLimits | None = None,
        communication_limits: CommunicationLimits | None = None,
    ) -> None:
        """Create a rules engine for the supplied participant colors."""
        if initialize:
            self.seed = (
                seed
                if seed is not None
                else random.SystemRandom().randrange(sys.maxsize)
            )
            self.rng = random.Random(self.seed)

            self.id = str(uuid.uuid4())
            self.vps_to_win = vps_to_win
            self.state = GameState(
                colors,
                catan_map,
                discard_limit=discard_limit,
                shuffle_players=shuffle_players,
                rng=self.rng,
                trade_limits=trade_limits,
            )
            self.capture_history = capture_history
            self.events: list[GameEvent] = []
            self.communication_limits = communication_limits or CommunicationLimits()
            self.commitments: list[SocialCommitment] = []
            self.history: list[HistoryEntry] = []

    def step(
        self,
        action: Action,
        validate_action: bool = True,
        force: bool = False,
    ) -> EngineTransition:
        """Strictly apply one already-selected legal game action."""
        if not isinstance(action, Action):
            raise ValueError("Expected one typed engine Action")
        action = copy.deepcopy(action)
        if not force and self.winning_color() is not None:
            raise ValueError("Cannot step a terminal game")
        if force:
            assert_forced_action_is_explicit(action)
            validate_action = False

        if validate_action and not is_valid_action(self.state, action):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.state.playable_actions}"
            )

        before_revision = self.revision
        history_entry: HistoryEntry | None = None
        if self.capture_history:
            history_entry = (
                copy.deepcopy(self.state),
                copy.deepcopy(action),
                before_revision,
                tuple(copy.deepcopy(self.commitments)),
            )

        trade_offers = (
            copy.deepcopy(self.state.trade_window.offers)
            if self.state.trade_window is not None else {}
        )
        resolved_action = apply_action(self.state, action, force=force)
        if history_entry is not None:
            self.history.append(history_entry)
        action_event = event_from_action(resolved_action, before_revision, trade_offers=trade_offers)
        event = self.publish_event(
            action_event.event_type,
            action_event.actor,
            action_event.public_payload,
            private_overlays=action_event.private_overlays,
            visible_to=action_event.visible_to,
            causation_id=action_event.causation_id,
        )
        for commitment in self.commitments:
            if commitment.active and commitment.expires_turn <= self.state.num_turns:
                commitment.status = CommitmentStatus.EXPIRED
        return EngineTransition(
            before_revision=before_revision,
            after_revision=self.revision,
            requested_action=copy.deepcopy(action),
            resolved_action=copy.deepcopy(resolved_action),
            events=(event,),
            winner=self.winning_color(),
        )

    def undo(self) -> Action | None:
        """Undo the last action, restoring the previous game state.

        Returns:
            Action: The action that was undone, or None if no history.
        """
        if not self.capture_history or not self.history:
            return None

        prev_state, action, event_count, commitments = self.history.pop()
        self.state = prev_state
        self.rng = self.state.rng
        self.commitments = list(copy.deepcopy(commitments))
        del self.events[event_count:]
        return action

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self.capture_history and len(self.history) > 0

    @property
    def revision(self) -> int:
        return len(self.events)

    def observe(self, color: Color) -> PlayerObservation:
        observation = observe_state(self.state, color)
        if self.winning_color() is not None:
            observation.valid_actions = []
        return observation

    def is_action_valid(self, action: Action) -> bool:
        return self.winning_color() is None and is_valid_action(self.state, action)

    def publish_event(
        self,
        event_type: str,
        actor: Color,
        public_payload: object,
        *,
        private_overlays: PrivateOverlays = (),
        visible_to: tuple[Color, ...] | None = None,
        causation_id: str | None = None,
    ) -> GameEvent:
        """Publish one already-resolved fact without applying a gameplay action."""
        if actor not in self.state.colors:
            raise ValueError("Event actor must be a participant")
        if not isinstance(event_type, str) or not event_type:
            raise ValueError("Event type must be a nonempty string")
        sequence = self.revision
        event = copy.deepcopy(GameEvent(
            sequence=sequence,
            causation_id=causation_id if causation_id is not None else f"action:{sequence}",
            actor=actor,
            event_type=event_type,
            public_payload=public_payload,
            private_overlays=private_overlays,
            visible_to=visible_to,
        ))
        self.events.append(event)
        return copy.deepcopy(event)

    def project_events(self, color: Color) -> tuple[PlayerEvent, ...]:
        if color not in self.state.colors:
            raise ValueError(f"Color {color} is not a participant")
        projected = (project_event(event, color) for event in self.events)
        return tuple(event for event in projected if event is not None)

    def project_game_events(self, color: Color) -> tuple[PlayerEvent, ...]:
        return tuple(
            event
            for event in self.project_events(color)
            if event.event_type != "MESSAGE_SENT"
        )

    def project_messages(
        self,
        color: Color,
        limit: int | None = None,
    ) -> tuple[PlayerEvent, ...]:
        messages = tuple(
            event
            for event in self.project_events(color)
            if event.event_type == "MESSAGE_SENT"
        )
        maximum = self.communication_limits.recent_message_window if limit is None else limit
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("Message limit must be a non-negative integer")
        return messages[-maximum:] if maximum else ()

    def append_message(
        self,
        *,
        speaker: Color,
        text: str,
        audience: tuple[Color, ...],
        causation_id: str,
        commitment: tuple[str, str, int] | None = None,
        respondents: tuple[Color, ...] | None = None,
    ) -> GameEvent:
        return append_message(
            self,
            speaker=speaker,
            text=text,
            audience=audience,
            causation_id=causation_id,
            commitment=commitment,
            respondents=respondents,
        )

    def active_commitments(self, color: Color) -> tuple[SocialCommitment, ...]:
        return tuple(
            copy.deepcopy(commitment)
            for commitment in self.commitments
            if commitment.active
            and (commitment.proposer == color or color in commitment.audience)
        )

    def snapshot(self) -> GameEngineSnapshot:
        return snapshot_engine(self)

    def restore(self, snapshot: GameEngineSnapshot) -> None:
        restore_engine(self, snapshot)

    def winning_color(self) -> Color | None:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        color = self.state.colors[self.state.current_turn_index]
        key = player_key(self.state, color)
        if self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= self.vps_to_win:
            return color
        return None

    def copy(self) -> GameEngine:
        """Creates a copy of this GameEngine, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            GameEngine: GameEngine copy.
        """
        return copy_engine(self, GameEngine(colors=[], initialize=False))
