"""Blind and strict replay phases never index an upcoming parsed action."""
import contextlib
import io
from threading import Event, Thread
from typing import Any

from cle.game_engine.game import GameEngine
from cle.game_engine.models.actions import generate_playable_actions
from cle.game_engine.models.enums import ActionPrompt
from cle.game_engine.models.player import Color
from cle.replay.runtime.step_executor import replay_step_logic
from cle.sandbox import CatanSandbox
from cle.sandbox.replay import ReplaySandbox
from playground.game_viewer.app import app
from playground.game_viewer.commentary.contextualizer import (
    CausalCommentarySession,
)
from playground.game_viewer.routes.health import _get_state_snapshot
from playground.game_viewer.state import ServerState

from .support import (
    _BlindFutureTrapActions,
    _FutureTrapActions,
)


def live_sandbox(state: ServerState) -> CatanSandbox | ReplaySandbox:
    """The viewer's current sandbox, which every caller here has already created."""
    sandbox = state.current_sandbox
    assert sandbox is not None
    return sandbox



def test_blind_begin_and_commit_never_index_upcoming_parsed_action(paired_state: ServerState) -> None:
    original_actions = paired_state.replay_data["parsed_actions"]
    paired_state.replay_data["parsed_actions"] = _BlindFutureTrapActions()
    session = CausalCommentarySession(paired_state)

    context = session.begin()
    token = session.commit(context.context_id, {"blind": True})

    assert token.value
    paired_state.replay_data["parsed_actions"] = original_actions


def test_same_session_rebuilds_corner_index_after_replay_replacement(paired_state: ServerState) -> None:
    session = CausalCommentarySession(paired_state)
    first_context = session.begin()
    first_index = session._corner_index
    session.abandon(first_context.context_id)

    old_game_id = paired_state.replay_data["game_id"]
    paired_state.replay_data["game_id"] = "replacement-game"
    second_context = session.begin()

    assert second_context.game_id == "replacement-game"
    assert session._corner_index is not first_index
    paired_state.replay_data["game_id"] = old_game_id


def test_state_snapshot_waits_for_replay_mutation_lock(paired_state: ServerState) -> None:
    entered = Event()
    finished = Event()
    result: Any = {}

    def read_snapshot() -> None:
        with app.app_context():
            entered.set()
            with paired_state.replay_mutation_lock:
                result["response"] = _get_state_snapshot(paired_state)
            finished.set()

    with paired_state.replay_mutation_lock:
        thread = Thread(target=read_snapshot)
        thread.start()
        assert entered.wait(timeout=1)
        assert finished.wait(timeout=0.05) is False
        paired_state.replay_index = 1
        paired_state.replay_revision += 1

    assert finished.wait(timeout=1)
    thread.join(timeout=1)
    assert result["response"].get_json()["replay"]["event_index"] == 1


def test_live_step_is_rejected_during_replay(paired_state: ServerState) -> None:
    client = app.test_client()

    step_response = client.post("/api/step")

    assert step_response.status_code == 409
    assert paired_state.replay_index == 0
    assert live_sandbox(paired_state).game_engine.state.actions == []


def test_strict_replay_step_never_reads_a_future_parsed_row() -> None:
    players = [
        Color.RED,
        Color.BLUE,
        Color.WHITE,
        Color.ORANGE,
    ]
    game = GameEngine(players, shuffle_players=False)
    game.state.is_initial_build_phase = False
    game.state.current_prompt = ActionPrompt.MOVE_ROBBER
    game.state.playable_actions = generate_playable_actions(game.state)
    state: Any = ServerState()
    state.current_game = game
    state.current_players = players
    state.replay_mode = True
    state.game_running = True
    state.replay_data = {
        "game_id": "future-trap",
        "parsed_actions": _FutureTrapActions(
            {"index": 0, "type": "END_TURN", "player": 1},
            {"index": 1, "type": "MOVE_ROBBER", "player": 1},
        ),
        "events": [{"input": {"deltaS": 0}}],
        "total_events": 2,
        "colonist_color_to_engine_idx": {"1": 0},
        "end_game_state": {},
    }

    with contextlib.redirect_stdout(io.StringIO()):
        result = replay_step_logic(state, lambda: None, allow_lookahead=False)

    assert not isinstance(result, tuple)
    assert state.replay_index == 1
