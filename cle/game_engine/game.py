"""
Contains GameEngine class which is a thin-wrapper around the GameState class.
"""

import copy
import uuid
import random
import sys
from typing import Any, Sequence, Union, Optional

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
from cle.game_engine.models.actions import generate_playable_actions, trade_response_actions
from cle.game_engine.models.enums import Action, ActionPrompt, ActionType
from cle.game_engine.state import (
    GameState,
    apply_action,
    assert_forced_action_is_explicit,
    new_trade_window,
    validate_discard,
)
from cle.game_engine.state_functions import (
    get_player_freqdeck,
    maintain_longest_road,
    player_key,
    player_has_rolled,
)
from cle.game_engine.models.map import CatanMap
from cle.game_engine.observation import PlayerObservation, observe_state
from cle.game_engine.trading import TradeLimits, TradeOffer, TradeWindowStatus
from cle.game_engine.models.player import Color


def is_valid_action(state, action):
    """True if its a valid action right now. An action is valid
    if its in playable_actions or if its a OFFER_TRADE/COUNTER_OFFER in the right time."""
    if not isinstance(action, Action) or action.color not in state.colors:
        return False
    if action.action_type == ActionType.DISCARD and action.value is not None:
        try:
            validate_discard(state, action)
        except ValueError:
            return False
        return True
    if action.action_type in {ActionType.OFFER_TRADE, ActionType.COUNTER_OFFER}:
        if not (
            is_valid_trade(action.value)
            and action.value.id is None
            and state.current_prompt == ActionPrompt.PLAY_TURN
            and action.value.offered_by == action.color
        ):
            return False
        window = state.trade_window
        if action.action_type == ActionType.OFFER_TRADE:
            if not (
                state.current_color() == action.color
                and player_has_rolled(state, action.color)
                and action.value.parent_offer_id is None
            ):
                return False
            if window is None or window.status == TradeWindowStatus.CLOSED:
                # Match ensure_trade_window without installing a window during validation.
                window = new_trade_window(state)
        elif window is None or action.value.parent_offer_id is None:
            return False
        try:
            window.validate_offer(action.value)
        except ValueError:
            return False
        return can_fund_offer(state, action.value)

    if action.action_type in {
        ActionType.ACCEPT_TRADE,
        ActionType.REJECT_TRADE,
    } and action in trade_response_actions(state, action.color):
        return True

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
        history_entry = None
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
        public_payload: Any,
        *,
        private_overlays: tuple[tuple[Color, Any], ...] = (),
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
        if speaker not in self.state.colors:
            raise ValueError(f"Speaker {speaker} is not a participant")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text must be a nonempty string")
        if not isinstance(audience, (tuple, list)):
            raise ValueError("Message audience must be a sequence of participants")
        if not isinstance(causation_id, str) or not causation_id:
            raise ValueError("Message causation ID must be a nonempty string")
        audience = tuple(audience)
        if any(not isinstance(color, Color) for color in audience):
            raise ValueError("Message audience must contain participant colors")
        recipients = tuple(dict.fromkeys((speaker, *audience)))
        if any(color not in self.state.colors for color in recipients):
            raise ValueError("Message audience contains a non-participant")
        if respondents is not None and (
            not isinstance(respondents, tuple) or len(set(respondents)) != len(respondents)
            or any(not isinstance(color, Color) or color == speaker or color not in audience for color in respondents)
            or set(recipients) != set(self.state.colors)
        ):
            raise ValueError("Respondents require public speech and distinct eligible other players")
        sequence = self.revision
        proposed_commitment = None
        if commitment is not None:
            if not isinstance(commitment, (tuple, list)) or len(commitment) != 3:
                raise ValueError("Commitment must contain condition, promise, and expiry")
            condition, promise, expires_turn = commitment
            if not all(isinstance(value, str) and value.strip() for value in (condition, promise)):
                raise ValueError("Commitment condition and promise must be nonempty strings")
            if type(expires_turn) is not int or expires_turn < 0:
                raise ValueError("Commitment expiry must be a non-negative integer")
            proposed_commitment = SocialCommitment(
                id=f"commitment:{sequence}",
                proposer=speaker,
                audience=audience,
                condition=condition,
                promise=promise,
                created_sequence=sequence,
                expires_turn=expires_turn,
                source_message_sequence=sequence,
            )
        payload = {
            "speaker": speaker,
            "text": text,
            "audience": audience,
        }
        if respondents is not None:
            payload["respondents"] = respondents
        is_public = set(recipients) == set(self.state.colors)
        event = self.publish_event(
            "MESSAGE_SENT",
            speaker,
            payload if is_public else None,
            causation_id=causation_id,
            private_overlays=(
                ()
                if is_public
                else tuple((color, payload) for color in recipients)
            ),
            visible_to=None if is_public else recipients,
        )
        if proposed_commitment is not None:
            self.commitments.append(proposed_commitment)
        return event

    def active_commitments(self, color: Color) -> tuple[SocialCommitment, ...]:
        return tuple(
            copy.deepcopy(commitment)
            for commitment in self.commitments
            if commitment.active
            and (commitment.proposer == color or color in commitment.audience)
        )

    def snapshot(self) -> GameEngineSnapshot:
        return GameEngineSnapshot(
            engine_id=self.id,
            seed=self.seed,
            vps_to_win=self.vps_to_win,
            state=copy.deepcopy(self.state),
            events=tuple(copy.deepcopy(self.events)),
            capture_history=self.capture_history,
            communication_limits=self.communication_limits,
            commitments=tuple(copy.deepcopy(self.commitments)),
            history=tuple(
                (
                    copy.deepcopy(state),
                    copy.deepcopy(action),
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
        self.state = copy.deepcopy(snapshot.state)
        # Pre-fix saves can retain enemy-crossing candidates and road awards.
        # Probe only derived board fields, keeping the saved holder as the tie input.
        board = self.state.board
        rebuilt = copy.copy(board)
        road_result = rebuilt.recompute_road_state()
        road_changed = (
            rebuilt.road_color != board.road_color
            or rebuilt.road_length != board.road_length
        )
        for color in self.state.colors:
            key = player_key(self.state, color)
            length = rebuilt.road_lengths.get(color, 0)
            if (
                length != board.road_lengths.get(color, 0)
                or length != self.state.player_state[f"{key}_LONGEST_ROAD_LENGTH"]
                or (color == rebuilt.road_color) != self.state.player_state[f"{key}_HAS_ROAD"]
                or {frozenset(nodes) for nodes in rebuilt.connected_components.get(color, [])}
                != {frozenset(nodes) for nodes in board.connected_components.get(color, [])}
            ):
                road_changed = True
                break
        if (
            road_changed
            or any(
                set(edges) != set(rebuilt.buildable_edges(color))
                for color, edges in board.buildable_edges_cache.items()
            )
            or any(
                action.action_type == ActionType.BUILD_ROAD
                and action.value not in rebuilt.buildable_edges(action.color)
                for action in self.state.playable_actions
            )
        ):
            for color, edges in board.buildable_edges_cache.items():
                if set(edges) == set(rebuilt.buildable_edges(color)):
                    rebuilt.buildable_edges_cache[color] = edges
            self.state.board = rebuilt
            maintain_longest_road(self.state, *road_result)
            playable_actions = generate_playable_actions(self.state)
            # Keep saved menu indices only for an exact multiset match; values can be mutable.
            unmatched = self.state.playable_actions.copy()
            for action in playable_actions:
                try:
                    unmatched.remove(action)
                except ValueError:
                    break
            else:
                if not unmatched:
                    playable_actions = self.state.playable_actions
            self.state.playable_actions = playable_actions
        self.rng = self.state.rng
        self.events = list(copy.deepcopy(snapshot.events))
        self.capture_history = snapshot.capture_history
        self.communication_limits = snapshot.communication_limits
        self.commitments = list(copy.deepcopy(snapshot.commitments))
        self.history = [
            (
                copy.deepcopy(state),
                copy.deepcopy(action),
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
        color = self.state.colors[self.state.current_turn_index]
        key = player_key(self.state, color)
        if self.state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] >= self.vps_to_win:
            return color
        return None

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
        game_copy.state = copy.deepcopy(self.state)
        game_copy.rng = game_copy.state.rng
        game_copy.events = copy.deepcopy(self.events)
        game_copy.communication_limits = self.communication_limits
        game_copy.commitments = list(copy.deepcopy(self.commitments))
        game_copy.capture_history = self.capture_history
        game_copy.history = (
            [
                (
                    copy.deepcopy(state),
                    copy.deepcopy(action),
                    event_count,
                    tuple(copy.deepcopy(commitments)),
                )
                for state, action, event_count, commitments in self.history
            ]
            if self.capture_history
            else []
        )
        return game_copy
