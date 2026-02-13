"""Replay playback endpoints."""

import json
import time
import traceback
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app

from engine.game import Game
from engine.models.player import SimplePlayer, Color

from ..colonist.coordinates import create_map_from_colonist
from ..colonist.event_parser import parse_colonist_events_to_actions
from ..replay.step_executor import replay_step_logic
from ..replay.navigation import (
    replay_undo_logic, replay_goto_fast_logic,
    replay_goto_sequential_logic, replay_goto_divergence_logic,
)
from .websocket import broadcast_game_state

replay_bp = Blueprint('replay', __name__)


def _get_state():
    return current_app.config['SERVER_STATE']


def _broadcast():
    state = _get_state()
    socketio = current_app.config['SOCKETIO']
    broadcast_game_state(socketio, state)


@replay_bp.route('/api/load-replay', methods=['POST'])
def load_replay():
    """Load a Colonist replay for playback - runs through our engine."""
    state = _get_state()

    data = request.json or {}
    game_id = data.get('game_id', '').strip()

    if not game_id:
        return jsonify({"error": "game_id required"}), 400

    replay_dir = Path(__file__).parent.parent.parent.parent / "data_pipeline" / "bootstrapping" / "data" / "raw_replays"
    replay_file = None

    for pattern in [f"{game_id}.json", f"{game_id}_*.json", f"*{game_id}*.json"]:
        matches = list(replay_dir.glob(pattern))
        if matches:
            replay_file = matches[0]
            break

    if not replay_file:
        return jsonify({"error": f"Replay {game_id} not found in {replay_dir}"}), 404

    try:
        with open(replay_file) as f:
            raw_data = json.load(f)

        if "data" in raw_data:
            events = raw_data["data"].get("eventHistory", {}).get("events", [])
        elif "events" in raw_data:
            events = raw_data["events"]
        else:
            events = raw_data.get("eventHistory", {}).get("events", [])

        if not events:
            return jsonify({"error": "No events found in replay"}), 400

        initial_state = None
        if "data" in raw_data:
            initial_state = raw_data["data"].get("eventHistory", {}).get("initialState")
        elif "eventHistory" in raw_data:
            initial_state = raw_data["eventHistory"].get("initialState")

        tile_hex_states = {}
        if initial_state:
            tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
            print(f"Loaded {len(tile_hex_states)} tile hex states for robber matching")

        parsed_actions = parse_colonist_events_to_actions(events, tile_hex_states)

        COLONIST_COLOR_NAMES = {
            1: "red", 2: "blue", 4: "green", 5: "black", 9: "white", 11: "mystic_blue",
        }

        COLONIST_TO_ENGINE_COLOR = {
            1: Color.RED, 2: Color.BLUE, 4: Color.GREEN, 5: Color.BLACK,
            9: Color.WHITE, 11: Color.MYSTIC_BLUE,
        }

        FALLBACK_ENGINE_COLORS = [Color.ORANGE, Color.BRONZE, Color.SILVER, Color.GOLD, Color.PINK, Color.MYSTIC_BLUE]

        colonist_players = []
        play_order_indices = []
        play_order_colors = []
        player_states = []

        if "data" in raw_data:
            player_states = raw_data["data"].get("playerUserStates", [])
            play_order_colors = raw_data["data"].get("playOrder", [])

            color_to_player_idx = {}
            for idx, p in enumerate(player_states):
                color_id = p.get("selectedColor")
                color_to_player_idx[color_id] = idx
                colonist_players.append({
                    "username": p.get("username"),
                    "color": COLONIST_COLOR_NAMES.get(color_id, f"color_{color_id}"),
                    "userId": str(p.get("userId")),
                })

            for color_id in play_order_colors:
                player_idx = color_to_player_idx.get(color_id)
                if player_idx is not None:
                    play_order_indices.append(player_idx)

        catan_map = None
        if initial_state:
            catan_map = create_map_from_colonist(initial_state)

        if play_order_colors:
            players = []
            fallback_idx = 0
            for color_id in play_order_colors:
                if color_id in COLONIST_TO_ENGINE_COLOR:
                    engine_color = COLONIST_TO_ENGINE_COLOR[color_id]
                else:
                    engine_color = FALLBACK_ENGINE_COLORS[fallback_idx % len(FALLBACK_ENGINE_COLORS)]
                    fallback_idx += 1
                    print(f"Warning: Unknown Colonist color ID {color_id}, using fallback {engine_color}")
                players.append(SimplePlayer(engine_color))
            print(f"Created players with colors: {[p.color for p in players]}")
        else:
            players = [
                SimplePlayer(Color.RED), SimplePlayer(Color.BLUE),
                SimplePlayer(Color.WHITE), SimplePlayer(Color.ORANGE),
            ]

        state.current_game = Game(players, catan_map=catan_map, shuffle_players=False)
        state.current_players = players

        state.replay_data = {
            "game_id": game_id,
            "events": events,
            "parsed_actions": parsed_actions,
            "total_events": len(parsed_actions),
            "file": str(replay_file),
            "colonist_players": colonist_players,
            "play_order": play_order_indices,
            "colonist_color_to_engine_idx": {str(color_id): idx for idx, color_id in enumerate(play_order_colors)},
            "tile_hex_states": tile_hex_states,
        }
        state.replay_index = 0
        state.replay_actions_per_step = []
        state.first_divergence_step = {}
        state.replay_mode = True
        state.game_running = True
        state.game_log = [{
            "type": "general",
            "timestamp": time.time(),
            "message": f"Loaded Colonist replay {game_id} ({len(parsed_actions)} actions)"
        }]

        _broadcast()

        return jsonify({
            "status": "loaded",
            "game_id": game_id,
            "total_events": len(parsed_actions),
            "file": str(replay_file),
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@replay_bp.route('/api/replay-step', methods=['POST'])
def replay_step():
    """Step through the loaded replay."""
    state = _get_state()
    result = replay_step_logic(state, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-undo', methods=['POST'])
def replay_undo():
    """Undo the last replay step."""
    state = _get_state()
    result = replay_undo_logic(state, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-fast', methods=['POST'])
def replay_goto_fast():
    """Jump to a specific replay step using fast skip logic."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    result = replay_goto_fast_logic(state, target_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-sequential', methods=['POST'])
def replay_goto_sequential():
    """Jump to a specific replay step using sequential stepping."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    result = replay_goto_sequential_logic(state, target_step, replay_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-divergence', methods=['POST'])
def replay_goto_divergence():
    """Step sequentially until divergence is detected."""
    state = _get_state()
    data = request.get_json() or {}
    max_steps = data.get("max_steps", 500)
    result = replay_goto_divergence_logic(state, max_steps, replay_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)
