"""
Contains GameEngine class which is a thin-wrapper around the GameState class.
"""

import copy
import uuid
import random
import sys
from typing import Sequence, Union, Optional

from cle.game_engine.communication import (
    CommitmentStatus,
    CommunicationLimits,
    SocialCommitment,
)
from cle.game_engine.events import (
    EngineTransition,
    GameEngineSnapshot,
    GameEvent,
    PlayerEvent,
    event_from_action,
    project_event,
)
from cle.game_engine.models.actions import trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.state import GameState, apply_action, assert_forced_action_is_explicit
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    player_key,
    player_has_rolled,
)
from cle.game_engine.models.map import CatanMap
from cle.game_engine.observation import PlayerObservation, observe_state
from cle.game_engine.trading import TradeLimits, TradeOffer
from cle.game_engine.models.player import Color


def is_valid_action(state, action):
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE/COUNTER_OFFER in the right time."""
    if action.action_type == ActionType.OFFER_TRADE:
        return (
            state.current_color() == action.color
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and player_has_rolled(state, action.color)
            and is_valid_trade(action.value)
            and action.value.parent_offer_id is None
            and action.value.offered_by == action.color
            and can_fund_offer(state, action.value)
        )

    if action.action_type in {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    } and action in trade_response_actions(state, action.color):
        return True

    if action.action_type == ActionType.COUNTER_OFFER:
        window = state.trade_window
        if window is None or not isinstance(action.value, TradeOffer):
            return False
        parent = window.offers.get(action.value.parent_offer_id)
        valid_prompt = state.current_prompt == ActionPrompt.PLAY_TURN
        return (
            parent is not None
            and parent.active
            and parent.parent_offer_id is None
            and action.value.offered_by == action.color
            and action.color != parent.offered_by
            and action.color in parent.audience
            and can_fund_offer(state, action.value)
            and valid_prompt
            and is_valid_trade(action.value)
        )

    return action in state.playable_actions


def is_valid_trade(action_value):
    """Return whether an action carries one canonical typed offer."""
    return isinstance(action_value, TradeOffer)


def can_fund_offer(state, offer: TradeOffer) -> bool:
    hand = get_player_freqdeck(state, offer.offered_by)
    return (
        all(held >= given for held, given in zip(hand, offer.give))
        and sum(hand) - sum(offer.give) >= offer.give_any
    )


class GameEngine:
    """Deterministic Catan rules and mutable materialized game state."""

    def __init__(
        self,
        colors: Sequence[Color],
        seed: Optional[int] = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        catan_map: Optional[CatanMap] = None,
        initialize: bool = True,
        shuffle_players: bool = True,
        capture_history: bool = False,
        trade_limits: TradeLimits | None = None,
        communication_limits: CommunicationLimits | None = None,
    ):
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
            self.history: list[
                tuple[GameState, Action, int, tuple[SocialCommitment, ...]]
            ] = []

    def step(
        self,
        action: Action,
        validate_action: bool = True,
        force: bool = False,
    ) -> EngineTransition:
        """Strictly apply one already-selected legal game action."""
        if force:
            assert_forced_action_is_explicit(action)
            validate_action = False

        if validate_action and not is_valid_action(self.state, action):
            raise ValueError(
                f"{action} not playable right now. playable_actions={self.state.playable_actions}"
            )

        before_revision = self.revision
        if self.capture_history:
            self.history.append(
                (
                    self.state.copy(),
                    action,
                    before_revision,
                    tuple(copy.deepcopy(self.commitments)),
                )
            )

        resolved_action = apply_action(self.state, action, force=force)
        event = event_from_action(resolved_action, before_revision)
        self.events.append(event)
        for commitment in self.commitments:
            if commitment.active and commitment.expires_turn <= self.state.num_turns:
                commitment.status = CommitmentStatus.EXPIRED
        return EngineTransition(
            before_revision=before_revision,
            after_revision=self.revision,
            requested_action=action,
            resolved_action=resolved_action,
            events=(event,),
            winner=self.winning_color(),
        )

    def undo(self) -> Optional[Action]:
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
        return observe_state(self.state, color)

    def is_action_valid(self, action: Action) -> bool:
        return is_valid_action(self.state, action)

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
        maximum = limit or self.communication_limits.recent_message_window
        return messages[-maximum:]

    def append_message(
        self,
        *,
        speaker: Color,
        text: str,
        audience: tuple[Color, ...],
        intent: str | None,
        causation_id: str,
        commitment: tuple[str, str, int] | None = None,
    ) -> GameEvent:
        if speaker not in self.state.colors:
            raise ValueError(f"Speaker {speaker} is not a participant")
        recipients = tuple(dict.fromkeys((speaker, *audience)))
        if any(color not in self.state.colors for color in recipients):
            raise ValueError("Message audience contains a non-participant")
        sequence = self.revision
        payload = {
            "speaker": speaker,
            "text": text,
            "audience": audience,
            "intent": intent,
        }
        is_public = set(recipients) == set(self.state.colors)
        event = GameEvent(
            sequence=sequence,
            causation_id=causation_id,
            actor=speaker,
            event_type="MESSAGE_SENT",
            public_payload=payload if is_public else None,
            private_overlays=(
                ()
                if is_public
                else tuple((color, payload) for color in recipients)
            ),
            visible_to=None if is_public else recipients,
        )
        self.events.append(event)
        if commitment is not None:
            condition, promise, expires_turn = commitment
            self.commitments.append(
                SocialCommitment(
                    id=f"commitment:{sequence}",
                    proposer=speaker,
                    audience=audience,
                    condition=condition,
                    promise=promise,
                    created_sequence=sequence,
                    expires_turn=expires_turn,
                    source_message_sequence=sequence,
                )
            )
        return event

    def active_commitments(self, color: Color) -> tuple[SocialCommitment, ...]:
        return tuple(
            commitment
            for commitment in self.commitments
            if commitment.active
            and (commitment.proposer == color or color in commitment.audience)
        )

    def snapshot(self) -> GameEngineSnapshot:
        return GameEngineSnapshot(
            engine_id=self.id,
            seed=self.seed,
            vps_to_win=self.vps_to_win,
            state=self.state.copy(),
            events=tuple(self.events),
            capture_history=self.capture_history,
            communication_limits=self.communication_limits,
            commitments=tuple(copy.deepcopy(self.commitments)),
            history=tuple(
                (
                    state.copy(),
                    action,
                    event_count,
                    tuple(copy.deepcopy(commitments)),
                )
                for state, action, event_count, commitments in self.history
            ),
        )

    def restore(self, snapshot: GameEngineSnapshot) -> None:
        self.id = snapshot.engine_id
        self.seed = snapshot.seed
        self.vps_to_win = snapshot.vps_to_win
        self.state = snapshot.state.copy()
        self.rng = self.state.rng
        self.events = list(snapshot.events)
        self.capture_history = snapshot.capture_history
        self.communication_limits = snapshot.communication_limits
        self.commitments = list(copy.deepcopy(snapshot.commitments))
        self.history = [
            (
                state.copy(),
                action,
                event_count,
                tuple(copy.deepcopy(commitments)),
            )
            for state, action, event_count, commitments in snapshot.history
        ]

    def winning_color(self) -> Union[Color, None]:
        """Gets winning color

        Returns:
            Union[Color, None]: Might be None if game truncated by TURNS_LIMIT
        """
        result = None
        for color in self.state.colors:
            key = player_key(self.state, color)
            if (
                self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"]
                >= self.vps_to_win
            ):
                result = color

        return result

    def copy(self) -> "GameEngine":
        """Creates a copy of this GameEngine, that can be modified without
        repercusions on this one (useful for simulations).

        Returns:
            GameEngine: GameEngine copy.
        """
        game_copy = GameEngine(colors=[], initialize=False)
        game_copy.seed = self.seed
        game_copy.id = self.id
        game_copy.vps_to_win = self.vps_to_win
        game_copy.state = self.state.copy()
        game_copy.rng = game_copy.state.rng
        game_copy.events = list(self.events)
        game_copy.communication_limits = self.communication_limits
        game_copy.commitments = list(copy.deepcopy(self.commitments))
        game_copy.capture_history = self.capture_history
        game_copy.history = (
            [
                (
                    state.copy(),
                    action,
                    event_count,
                    tuple(copy.deepcopy(commitments)),
                )
                for state, action, event_count, commitments in self.history
            ]
            if self.capture_history
            else []
        )
        return game_copy
