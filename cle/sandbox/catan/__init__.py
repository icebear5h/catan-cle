"""Lightweight asynchronous orchestration for one Catan game engine."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import copy as copy
from copy import deepcopy as deepcopy
from dataclasses import replace as replace

from cle.game_engine.events import project_event as project_event
from cle.game_engine.game import GameEngine
from cle.game_engine.game import is_valid_action as is_valid_action
from cle.game_engine.models.actions import generate_playable_actions as generate_playable_actions
from cle.game_engine.models.actions import trade_response_actions as trade_response_actions
from cle.game_engine.models.enums import Action
from cle.game_engine.models.player import Color
from cle.game_engine.state import GameState
from cle.game_engine.state import ensure_trade_window as ensure_trade_window
from cle.game_engine.state_functions import player_num_resource_cards as player_num_resource_cards
from cle.harness.action_tools import legal_tool_names as legal_tool_names
from cle.harness.action_tools import parse_tool_choice as parse_tool_choice
from cle.players.contracts import PlayerAttempt, PlayerContext, SandboxPlayer
from cle.players.validation import action_from_choice as action_from_choice
from cle.players.validation import choice_followup_action as choice_followup_action
from cle.players.validation import validate_communication_choice as validate_communication_choice
from cle.players.validation import validate_player_attempt as validate_player_attempt
from cle.sandbox.action_batches import PendingActionBatch
from cle.sandbox.communication import (
    CommunicationAdmission,
    CommunicationOpportunity,
    CommunicationPolicy,
)
from cle.sandbox.contracts import RetryPolicy, SandboxSnapshot, SandboxStepResult, SandboxView
from cle.sandbox.decision import build_decision_context as build_decision_context
from cle.sandbox.trade_preauthorization import TradePreauthorization

from . import barriers, checkpoints, choices, continuations, reactions, speech, stepping
from .support import (
    MissingPlayerError as MissingPlayerError,
)
from .support import (
    PlayerResponseError as PlayerResponseError,
)
from .support import (
    PostActionCommunicationCancelled as PostActionCommunicationCancelled,
)
from .support import (
    PostActionCommunicationError as PostActionCommunicationError,
)
from .support import (
    SandboxError as SandboxError,
)
from .support import (
    TerminalSandboxError as TerminalSandboxError,
)
from .support import (
    _gather_or_cancel as _gather_or_cancel,
)

# Extracted helpers resolve the original imported functions through this module
# at call time, preserving public-path wrappers and monkeypatches.


class CatanSandbox:
    """Compose one game engine with one independently stateful player per color."""

    def __init__(
        self,
        game_engine: GameEngine,
        players: Mapping[Color, SandboxPlayer],
        *,
        retry_policy: RetryPolicy | None = None,
        communication_policy: CommunicationPolicy | None = None,
        refresh_players: Callable[[CatanSandbox], None] | None = None,
    ) -> None:
        self.game_engine = game_engine
        self.players = dict(players)
        self.retry_policy = retry_policy or RetryPolicy()
        self.communication_policy = communication_policy or CommunicationPolicy()
        self._refresh_players = refresh_players
        self.decision_trace: list[PlayerAttempt] = []
        self.communication_trace: list[CommunicationAdmission] = []
        self._step_state: GameState | None = None
        self._pending_decision_revision: int | None = None
        self._speech_used = False
        self._speech_calls_remaining: int | None = None
        self._pending_reactions: tuple[CommunicationOpportunity, ...] = ()
        self._pre_robber_sequence: int | None = None
        self._trade_preauthorization: TradePreauthorization | None = None
        self._pending_action_batch: PendingActionBatch | None = None
        self._validate_players()

    @classmethod
    def create(
        cls,
        colors: Sequence[Color],
        players: Mapping[Color, SandboxPlayer],
        *,
        seed: int | None = None,
        discard_limit: int = 7,
        vps_to_win: int = 10,
        shuffle_players: bool = True,
        retry_policy: RetryPolicy | None = None,
        communication_policy: CommunicationPolicy | None = None,
    ) -> CatanSandbox:
        engine = GameEngine(
            colors,
            seed=seed,
            discard_limit=discard_limit,
            vps_to_win=vps_to_win,
            shuffle_players=shuffle_players,
        )
        return cls(
            engine,
            players,
            retry_policy=retry_policy,
            communication_policy=communication_policy,
        )

    @property
    def revision(self) -> int:
        return self.game_engine.revision

    def current_actor(self) -> Color:
        current_color: Callable[[], Color] = self.game_engine.state.current_color
        return current_color()

    def register_player(self, player: SandboxPlayer) -> None:
        if self._step_state is not None:
            raise SandboxError("Cannot replace a player while a step is in flight")
        if player.color not in self.game_engine.state.colors:
            raise ValueError(f"Player color {player.color} is not in this game")
        self.players[player.color] = player
        self._validate_players()

    def view(self, observer: Color | None = None) -> SandboxView:
        observer = observer or self.current_actor()
        observation = self.game_engine.observe(observer)
        current_actor = self.current_actor()
        return deepcopy(SandboxView(
            revision=self.revision,
            observer=observer,
            current_actor=current_actor,
            turn_number=self.game_engine.state.num_turns,
            phase=observation.current_phase,
            observation=observation,
            events=self.game_engine.project_events(observer),
            legal_actions=(
                tuple(self.game_engine.state.playable_actions)
                if observer == current_actor and self.game_engine.winning_color() is None
                else ()
            ),
            winner=self.game_engine.winning_color(),
        ))

    async def step(self) -> SandboxStepResult:
        """Obtain one valid player action, apply it strictly, and acknowledge it."""
        if self._step_state is not None:
            raise SandboxError("A sandbox already has a step in flight")
        self._step_state = self.game_engine.state
        try:
            return await self._step()
        finally:
            self._step_state = None

    def decision_context(
        self,
        actor: Color | None = None,
        advertised_actions: tuple[Action, ...] | None = None,
    ) -> PlayerContext:
        """Build the shared perspective-safe context for one exact decision."""
        return choices.decision_context(self, actor, advertised_actions)

    def snapshot(self) -> SandboxSnapshot:
        return checkpoints.snapshot(self)

    def restore(self, snapshot: SandboxSnapshot) -> None:
        checkpoints.restore(self, snapshot)

    def _refresh_inference_policy(self) -> None:
        # Only call with no outstanding acquisition/admission. Concurrent speech
        # and trade batches, including retries, retain their original players.
        if self._refresh_players is not None:
            self._refresh_players(self)

    def _check_revision(self, revision: int) -> None:
        if self.revision != revision or self.game_engine.state is not self._step_state:
            raise SandboxError(
                f"Stale context: game state changed; expected revision {revision}, "
                f"found {self.revision}"
            )

    def _validate_players(self) -> None:
        engine_colors = set(self.game_engine.state.colors)
        player_colors = set(self.players)
        if player_colors != engine_colors:
            missing = engine_colors - player_colors
            extra = player_colors - engine_colors
            raise ValueError(
                f"Sandbox players must exactly match engine colors; "
                f"missing={sorted(color.value for color in missing)}, "
                f"extra={sorted(color.value for color in extra)}"
            )
        mismatched = [color for color, player in self.players.items() if player.color != color]
        if mismatched:
            raise ValueError(f"Player mapping keys do not match players: {mismatched}")

    # Bind typed helpers as ordinary methods, retaining instance override seams.
    _step = stepping._step
    _step_barrier = barriers._step_barrier
    _discard_barrier_contexts = barriers._discard_barrier_contexts
    _trade_barrier_contexts = barriers._trade_barrier_contexts
    _post_action_communication = speech._post_action_communication
    _run_communication = speech._run_communication
    _talk_context = speech._talk_context
    _validate_context_update = staticmethod(choices._validate_context_update)
    _prompt_player = choices._prompt_player
    _retry_feedback = staticmethod(choices._retry_feedback)
    _get_action_from_player = choices._get_action_from_player
    _is_action_valid = choices._is_action_valid
    _batch_boundary = continuations._batch_boundary
    _pause_action_batch = continuations._pause_action_batch
    _advance_action_batch = continuations._advance_action_batch
    _resolve_action_batch = continuations._resolve_action_batch
    _resolve_trade_preauthorization = continuations._resolve_trade_preauthorization
    _reactive = reactions._reactive
    _start_speech_budget = reactions._start_speech_budget
    _append_speech = reactions._append_speech
    _open_pre_robber_window = reactions._open_pre_robber_window
    _run_reactive = reactions._run_reactive
