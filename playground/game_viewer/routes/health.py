"""Health, state, and reset endpoints."""

import json
import time

from flask import Blueprint, jsonify, request, current_app

from engine.json import GameEncoder
from ..live.game_logging import get_player_resources, get_player_dev_cards

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

    state.reset()

    socketio.emit('game_state', {
        "game": None,
        "running": False,
        "llm_thinking": [],
        "game_log": [],
        "all_player_resources": None
    })

    return jsonify({"status": "reset", "message": "Game cleared"})


@health_bp.route('/api/state')
def get_state():
    """Get current game state."""
    state = _get_state()

    if not state.current_game:
        return jsonify({"error": "No game"}), 404

    all_resources = get_player_resources(state.current_game.state)
    all_dev_cards = get_player_dev_cards(state.current_game.state)

    result = {
        "game": json.loads(json.dumps(state.current_game, cls=GameEncoder)),
        "running": state.game_running,
        "llm_thinking": state.llm_thinking[-10:],
        "game_log": state.game_log[-50:],
        "all_player_resources": all_resources,
        "all_player_dev_cards": all_dev_cards,
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

    return jsonify(result)
