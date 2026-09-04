"""Health, state, and reset endpoints."""

import json

from flask import Blueprint, jsonify, current_app

from cle.game_engine.json import GameEncoder
from ..live.game_logging import get_player_resources, get_player_dev_cards
from ..replay.model_traces import build_paired_model_trace_window
from ..replay.narrator_reasoning import build_paired_narrator_reasoning_window
from ..replay.transcript import build_paired_transcript_window

health_bp = Blueprint('health', __name__)


def _get_state():
    return current_app.config['SERVER_STATE']


@health_bp.route('/api/health')
def health():
    """Health check."""
    return jsonify({"status": "ok", "message": "Game viewer server running"})


@health_bp.route('/api/reset', methods=['POST'])
def reset_game():
    """Reset/clear the current game."""
    state = _get_state()
    socketio = current_app.config['SOCKETIO']

    with state.replay_mutation_lock:
        trace_store = getattr(state, "live_trace_store", None)
        if trace_store is not None and state.live_trace_game_id is not None:
            trace_store.mark_game_status(
                state.live_trace_game_id,
                status="reset",
            )
        state.reset()
        socketio.emit('game_state', {
            "game": None,
            "running": False,
            "live_trace_game_id": None,
            "live_inference": None,
            "last_live_step_error": None,
            "game_log": [],
            "all_player_resources": None,
            "replay_mode": False,
            "replay": None,
        })

    return jsonify({"status": "reset", "message": "Game cleared"})


def _get_state_snapshot(state):
    sandbox = state.current_sandbox
    if sandbox is None:
        return jsonify({"error": "No game"}), 404
    engine = sandbox.game_engine

    all_resources = get_player_resources(engine.state)
    all_dev_cards = get_player_dev_cards(engine.state)

    result = {
        "game": json.loads(json.dumps(engine, cls=GameEncoder)),
        "running": state.game_running,
        "live_trace_game_id": (
            None if state.replay_mode else state.live_trace_game_id
        ),
        "game_log": state.game_log[-50:],
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
        "live_inference": (
            None if state.replay_mode else getattr(state, "live_inference", None)
        ),
        "last_live_step_error": (
            None
            if state.replay_mode
            else getattr(state, "last_live_step_error", None)
        ),
        "replay_mode": state.replay_mode,
    }

    if state.replay_mode and state.replay_data:
        total_events = state.replay_data.get("total_events", 0)
        result["replay"] = {
            "game_id": state.replay_data.get("game_id"),
            "event_index": state.replay_index,
            "total_events": total_events,
            "colonist_players": state.replay_data.get("colonist_players", []),
            "play_order": state.replay_data.get("play_order", []),
            "progress": f"{state.replay_index}/{total_events}",
        }
        transcript_window = build_paired_transcript_window(
            state.replay_data, state.replay_index
        )
        if transcript_window is not None:
            result["replay"]["paired_transcript"] = transcript_window
        model_trace_window = build_paired_model_trace_window(
            state.replay_data, state.replay_index
        )
        if model_trace_window is not None:
            result["replay"]["paired_model_trace"] = model_trace_window
        narrator_reasoning_window = build_paired_narrator_reasoning_window(
            state.replay_data, state.replay_index
        )
        if narrator_reasoning_window is not None:
            result["replay"]["paired_narrator_reasoning"] = (
                narrator_reasoning_window
            )

    return jsonify(result)


@health_bp.route('/api/state')
def get_state():
    """Get one cursor-consistent game/replay snapshot."""
    state = _get_state()
    with state.replay_mutation_lock:
        return _get_state_snapshot(state)
