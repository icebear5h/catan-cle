"""Inject arbitrary game state for VLM benchmark screenshots."""

from flask import Blueprint, jsonify, request, current_app

inject_bp = Blueprint('inject', __name__)


@inject_bp.route('/api/inject-state', methods=['POST'])
def inject_state():
    """Accept a Game object's JSON serialization and broadcast it to the frontend.

    Expects POST body:
        {"game": <serialized game JSON from GameEncoder>}

    Or if "raw" is True, expects the full game_state event payload directly.
    """
    socketio = current_app.config['SOCKETIO']
    data = request.json

    if not data or 'game' not in data:
        return jsonify({"error": "Missing 'game' key in request body"}), 400

    # Build minimal game_state event that the frontend can render
    state_data = {
        "game": data["game"],
        "running": False,
        "game_log": [],
        "all_player_resources": data.get("all_player_resources", {}),
        "all_player_dev_cards": data.get("all_player_dev_cards", {}),
        "player_hands": data.get("player_hands", {}),
        "player_types": data.get("player_types", {}),
        "trade_state": None,
        "replay_mode": False,
        "last_dice_roll": None,
    }

    socketio.emit('game_state', state_data)

    return jsonify({"status": "ok"})


