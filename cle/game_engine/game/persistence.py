"""Snapshot, restore, and copy of a live engine's materialized state."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from cle.game_engine.events import GameEngineSnapshot
from cle.game_engine.game.recovery import copy_history, repair_restored_road_state

if TYPE_CHECKING:
    from cle.game_engine.game import GameEngine


def snapshot_engine(engine: GameEngine) -> GameEngineSnapshot:
    return GameEngineSnapshot(
        engine_id=engine.id,
        seed=engine.seed,
        vps_to_win=engine.vps_to_win,
        state=copy.deepcopy(engine.state),
        events=tuple(copy.deepcopy(engine.events)),
        capture_history=engine.capture_history,
        communication_limits=engine.communication_limits,
        commitments=tuple(copy.deepcopy(engine.commitments)),
        history=tuple(copy_history(engine.history)),
    )


def restore_engine(engine: GameEngine, snapshot: GameEngineSnapshot) -> None:
    engine.id = snapshot.engine_id
    engine.seed = snapshot.seed
    engine.vps_to_win = snapshot.vps_to_win
    engine.state = copy.deepcopy(snapshot.state)
    repair_restored_road_state(engine.state)
    engine.rng = engine.state.rng
    engine.events = list(copy.deepcopy(snapshot.events))
    engine.capture_history = snapshot.capture_history
    engine.communication_limits = snapshot.communication_limits
    engine.commitments = list(copy.deepcopy(snapshot.commitments))
    engine.history = copy_history(snapshot.history)


def copy_engine(engine: GameEngine, game_copy: GameEngine) -> GameEngine:
    """Fill one uninitialized engine with deep copies of another's state."""
    game_copy.seed = engine.seed
    game_copy.id = engine.id
    game_copy.vps_to_win = engine.vps_to_win
    game_copy.state = copy.deepcopy(engine.state)
    game_copy.rng = game_copy.state.rng
    game_copy.events = copy.deepcopy(engine.events)
    game_copy.communication_limits = engine.communication_limits
    game_copy.commitments = list(copy.deepcopy(engine.commitments))
    game_copy.capture_history = engine.capture_history
    game_copy.history = (
        copy_history(engine.history) if engine.capture_history else []
    )
    return game_copy
