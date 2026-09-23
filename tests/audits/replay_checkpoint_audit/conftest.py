"""Shared fixtures for replay checkpoint reuse, rollback, and continuation evidence."""

import pytest

from cle.game_engine.game import GameEngine
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.state import ServerState

from .support import COLORS


@pytest.fixture
def replay() -> tuple[ServerState, ReplaySandbox]:
    engine = GameEngine(COLORS, seed=4, shuffle_players=False, capture_history=True)
    state = ServerState()
    sandbox = ReplaySandbox(state, engine)
    state.current_sandbox = sandbox
    state.replay_mode = state.game_running = True
    state.corner_to_node_map = {"_audit": engine.state.playable_actions[0].value}
    state.replay_data = {
        "game_id": "checkpoint-audit",
        "events": [{}],
        "parsed_actions": [{
            "index": 0, "type": "BUILD_SETTLEMENT", "player": 1,
            "colonist_corner": "audit",
        }],
        "colonist_color_to_engine_idx": {"1": 0, "2": 1},
        "end_game_state": {},
    }
    return state, sandbox
