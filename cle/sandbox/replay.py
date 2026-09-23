"""Replay sandbox facade over the deterministic replay runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from copy import deepcopy
from typing import TypedDict

from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color
from cle.players.contracts import PlayerContext, SandboxPlayer
from cle.replay.contracts import ReplayOutcome, ReplayRuntimeState
from cle.replay.runtime.navigation import (
    replay_goto_divergence_logic,
    replay_goto_fast_logic,
    replay_goto_sequential_logic,
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox.contracts import SandboxView
from cle.sandbox.decision import build_decision_context


class ReplayCursorIdentity(TypedDict):
    """The exact replay cursor a frozen decision context was built from."""

    game: GameEngine
    game_id: object
    replay_index: int
    replay_revision: int


class ReplaySandbox:
    """Own replay stepping/navigation around one mutable runtime state object.

    The runtime object is the same structural `ReplayRuntimeState` the replay
    package defines, so existing `ServerState` instances and headless eval states
    drive one executor. Flask and Socket.IO are not dependencies of this class.
    """

    def __init__(
        self,
        runtime_state: ReplayRuntimeState,
        game_engine: GameEngine | None = None,
        *,
        players: Mapping[Color, SandboxPlayer] | None = None,
    ) -> None:
        self._state = runtime_state
        self._game_engine = game_engine or runtime_state.current_game
        self._players = dict(players or {})

    @property
    def game_engine(self) -> GameEngine:
        # A state with no loaded game still constructs; it fails on first use.
        assert self._game_engine is not None
        return self._game_engine

    def replace_game_engine(self, engine: GameEngine) -> None:
        self._game_engine = engine

    @property
    def players(self) -> Mapping[Color, SandboxPlayer]:
        return dict(self._players)

    @property
    def revision(self) -> int:
        revision: int = getattr(self._state, "replay_revision", 0)
        return revision

    @property
    def replay_index(self) -> int:
        return self._state.replay_index

    @property
    def source_ongoing(self) -> bool:
        replay_data = getattr(self._state, "replay_data", None) or {}
        return 0 <= self.replay_index < len(replay_data.get("parsed_actions", ()))

    def view(self, observer: Color | None = None) -> SandboxView:
        """Return a pure player projection at the replay revision."""
        engine = self.game_engine
        observer = observer or engine.state.current_color()
        observation = engine.observe(observer)
        current_actor = engine.state.current_color()
        source_ongoing = self.source_ongoing
        winner = None if source_ongoing else engine.winning_color()
        legal_actions = (
            tuple(deepcopy(engine.state.playable_actions))
            if observer == current_actor and winner is None
            else ()
        )
        observation.valid_actions = list(legal_actions)
        return SandboxView(
            revision=self.revision,
            observer=observer,
            current_actor=current_actor,
            turn_number=engine.state.num_turns,
            phase=observation.current_phase,
            observation=observation,
            events=engine.project_events(observer),
            legal_actions=legal_actions,
            winner=winner,
        )

    def step(
        self,
        broadcast: Callable[[], None] | None = None,
        *,
        allow_lookahead: bool = True,
    ) -> ReplayOutcome:
        """Advance exactly one recorded replay event."""
        response: ReplayOutcome = replay_step_logic(
            self._state,
            broadcast or _no_op,
            allow_lookahead=allow_lookahead,
        )
        return response

    def undo(self, broadcast: Callable[[], None] | None = None) -> ReplayOutcome:
        response: ReplayOutcome = replay_undo_logic(self._state, broadcast or _no_op)
        return response

    def goto_fast(
        self,
        target_step: int,
        broadcast: Callable[[], None] | None = None,
    ) -> ReplayOutcome:
        callback = broadcast or _no_op
        response: ReplayOutcome = replay_goto_fast_logic(
            self._state,
            target_step,
            lambda: self.step(callback),
            callback,
        )
        return response

    def goto_sequential(
        self,
        target_step: int,
        broadcast: Callable[[], None] | None = None,
    ) -> ReplayOutcome:
        callback = broadcast or _no_op
        response: ReplayOutcome = replay_goto_sequential_logic(
            self._state,
            target_step,
            lambda: self.step(callback),
            callback,
        )
        return response

    def goto_divergence(
        self,
        max_steps: int,
        broadcast: Callable[[], None] | None = None,
    ) -> ReplayOutcome:
        callback = broadcast or _no_op
        response: ReplayOutcome = replay_goto_divergence_logic(
            self._state,
            max_steps,
            lambda: self.step(callback),
            callback,
        )
        return response

    def decision_context(self) -> tuple[PlayerContext, ReplayCursorIdentity]:
        """Freeze one general player context plus its replay cursor identity."""
        mutation_lock = getattr(self._state, "replay_mutation_lock", nullcontext())
        with mutation_lock:
            game = self.game_engine
            replay_data = self._state.replay_data or {}
            replay_index = self._state.replay_index
            replay_revision = self.revision
            context = build_decision_context(
                game,
                context_revision=replay_revision,
                allow_terminal=self.source_ongoing,
            )
            identity: ReplayCursorIdentity = {
                "game": game,
                "game_id": replay_data.get("game_id"),
                "replay_index": replay_index,
                "replay_revision": replay_revision,
            }
        return context, identity

    def is_stale(self, identity: ReplayCursorIdentity) -> bool:
        mutation_lock = getattr(self._state, "replay_mutation_lock", nullcontext())
        with mutation_lock:
            replay_data = self._state.replay_data
            current_game_id = replay_data.get("game_id") if replay_data else None
            return (
                self.game_engine is not identity["game"]
                or self._state.replay_index != identity["replay_index"]
                or self.revision != identity["replay_revision"]
                or current_game_id != identity["game_id"]
            )


def _no_op() -> None:
    return None
