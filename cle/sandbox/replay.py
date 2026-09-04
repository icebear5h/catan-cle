"""Replay sandbox facade over the deterministic replay runtime."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Mapping

from cle.replay.runtime.navigation import (
    replay_goto_divergence_logic,
    replay_goto_fast_logic,
    replay_goto_sequential_logic,
    replay_undo_logic,
)
from cle.replay.runtime.step_executor import replay_step_logic
from cle.players.contracts import PlayerContext, SandboxPlayer
from cle.sandbox.contracts import SandboxView
from cle.sandbox.decision import build_decision_context
from cle.game_engine.game import GameEngine
from cle.game_engine.models.player import Color


class ReplaySandbox:
    """Own replay stepping/navigation around one mutable runtime state object.

    The runtime object is deliberately duck-typed during migration so existing
    `ServerState` instances and headless eval states can use the same core replay
    executor. Flask and Socket.IO are not dependencies of this class.
    """

    def __init__(
        self,
        runtime_state: Any,
        game_engine: GameEngine | None = None,
        *,
        players: Mapping[Color, SandboxPlayer] | None = None,
    ) -> None:
        self._state = runtime_state
        self._game_engine = game_engine or runtime_state.current_game
        self._players = dict(players or {})

    @property
    def game_engine(self) -> GameEngine:
        return self._game_engine

    def replace_game_engine(self, engine: GameEngine) -> None:
        self._game_engine = engine

    @property
    def players(self) -> Mapping[Color, SandboxPlayer]:
        return dict(self._players)

    @property
    def revision(self) -> int:
        return getattr(self._state, "replay_revision", 0)

    @property
    def replay_index(self) -> int:
        return self._state.replay_index

    def view(self, observer: Color | None = None):
        """Return a pure player projection at the replay revision."""
        engine = self.game_engine
        observer = observer or engine.state.current_color()
        observation = engine.observe(observer)
        current_actor = engine.state.current_color()
        return SandboxView(
            revision=self.revision,
            observer=observer,
            current_actor=current_actor,
            turn_number=engine.state.num_turns,
            phase=observation.current_phase,
            observation=observation,
            events=engine.project_events(observer),
            legal_actions=(
                tuple(engine.state.playable_actions)
                if observer == current_actor and engine.winning_color() is None
                else ()
            ),
            winner=engine.winning_color(),
        )

    def step(
        self,
        broadcast: Callable[[], None] | None = None,
        *,
        allow_lookahead: bool = True,
    ):
        """Advance exactly one recorded replay event."""
        return replay_step_logic(
            self._state,
            broadcast or _no_op,
            allow_lookahead=allow_lookahead,
        )

    def undo(self, broadcast: Callable[[], None] | None = None):
        return replay_undo_logic(self._state, broadcast or _no_op)

    def goto_fast(
        self,
        target_step: int,
        broadcast: Callable[[], None] | None = None,
    ):
        callback = broadcast or _no_op
        return replay_goto_fast_logic(
            self._state,
            target_step,
            lambda: self.step(callback),
            callback,
        )

    def goto_sequential(
        self,
        target_step: int,
        broadcast: Callable[[], None] | None = None,
    ):
        callback = broadcast or _no_op
        return replay_goto_sequential_logic(
            self._state,
            target_step,
            lambda: self.step(callback),
            callback,
        )

    def goto_divergence(
        self,
        max_steps: int,
        broadcast: Callable[[], None] | None = None,
    ):
        callback = broadcast or _no_op
        return replay_goto_divergence_logic(
            self._state,
            max_steps,
            lambda: self.step(callback),
            callback,
        )

    def decision_context(self) -> tuple[PlayerContext, dict[str, Any]]:
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
            )
            identity = {
                "game": game,
                "game_id": replay_data.get("game_id"),
                "replay_index": replay_index,
                "replay_revision": replay_revision,
            }
        return context, identity

    def is_stale(self, identity: Mapping[str, Any]) -> bool:
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
