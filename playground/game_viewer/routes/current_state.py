"""Endpoints for fetching current game state and observations."""

import pickle
import base64

from flask import Blueprint, jsonify, current_app

from cle.env.observation_formatter import CatanObservationFormatter, create_observation_from_state

current_state_bp = Blueprint('current_state', __name__)


@current_state_bp.route('/api/current-game-state', methods=['GET'])
def current_game_state():
    """Return the current game as a base64-encoded pickle.

    The notebook can unpickle this to get a full Game object
    without creating (and injecting) its own.
    """
    state = current_app.config['SERVER_STATE']

    if state.current_game is None:
        return jsonify({"error": "No game is currently running"}), 404

    pickled = base64.b64encode(pickle.dumps(state.current_game)).decode('ascii')
    return jsonify({"game_pickle": pickled})


@current_state_bp.route('/api/current-player-observation', methods=['GET'])
def current_player_observation():
    """Return the current player's formatted text observation."""
    state = current_app.config['SERVER_STATE']

    if state.current_game is None:
        return jsonify({"error": "No game is currently running"}), 404

    game = state.current_game
    current_player = game.state.current_player()
    formatter = CatanObservationFormatter()
    observation = create_observation_from_state(game.state, current_player.color)
    formatted = formatter.format(observation)

    color_name = (
        current_player.color.value
        if hasattr(current_player.color, "value")
        else str(current_player.color)
    )

    return jsonify({
        "color": color_name,
        "observation": formatted.raw_str,
    })
