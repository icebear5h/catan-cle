"""Runtime accessors keeping replay mechanics independent of viewer state shape."""

from __future__ import annotations

from cle.game_engine.game import GameEngine
from cle.replay.contracts import (
    GameEngineHolder,
    GameEngineReplacer,
    ReplayRuntimeState,
)


def get_game_engine(state: ReplayRuntimeState) -> GameEngine | None:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is not None and isinstance(sandbox, GameEngineHolder):
        return sandbox.game_engine
    return state.current_game


def set_game_engine(state: ReplayRuntimeState, engine: GameEngine) -> None:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is not None and isinstance(sandbox, GameEngineReplacer):
        sandbox.replace_game_engine(engine)
    else:
        state.current_game = engine
