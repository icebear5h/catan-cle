"""Runtime accessors keeping replay mechanics independent of viewer state shape."""

from __future__ import annotations


def get_game_engine(state):
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is not None and hasattr(sandbox, "game_engine"):
        return sandbox.game_engine
    return state.current_game


def set_game_engine(state, engine) -> None:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is not None and hasattr(sandbox, "replace_game_engine"):
        sandbox.replace_game_engine(engine)
    else:
        state.current_game = engine
