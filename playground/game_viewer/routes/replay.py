"""Replay playback endpoints."""

import json
import time
import traceback
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app
from httpx import HTTPError

from cle.harness.reasoning import validate_native_reasoning_request
from cle.harness.validation import (
    HarnessValidationError,
    validate_game_plan,
    validate_model_id,
)
from cle.players.baseline import ScriptedPlayer
from cle.replay.colonist.coordinates import create_map_from_colonist
from cle.replay.colonist.event_parser import parse_colonist_events_to_actions
from cle.replay.runtime.revision import bump_replay_revision
from cle.sandbox.replay import ReplaySandbox
from game_engine.game import GameEngine
from game_engine.models.player import Color

from ..async_runtime import sandbox_async_runtime
from ..replay.decision_preview import generate_decision_preview
from ..replay.model_traces import load_paired_model_traces
from ..replay.narrator_reasoning import load_paired_narrator_reasoning
from ..replay.transcript import get_curated_replay_path, load_paired_transcript
from .websocket import broadcast_game_state

replay_bp = Blueprint('replay', __name__)


def _get_state():
    return current_app.config['SERVER_STATE']


def _broadcast():
    state = _get_state()
    socketio = current_app.config['SOCKETIO']
    broadcast_game_state(socketio, state)


def _load_replay_transaction(state):
    data = request.json or {}
    game_id = data.get('game_id', '').strip()

    if not game_id:
        return jsonify({"error": "game_id required"}), 400

    replay_dir = Path(__file__).parent.parent.parent.parent / "data_pipeline" / "bootstrapping" / "data" / "raw_replays"
    curated_replay = get_curated_replay_path(game_id)
    replay_file = curated_replay if curated_replay and curated_replay.exists() else None

    if replay_file is None:
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
            initial_state = (
                raw_data["data"].get("eventHistory", {}).get("initialState")
                or raw_data["data"].get("initialState")
            )
        elif "eventHistory" in raw_data:
            initial_state = raw_data["eventHistory"].get("initialState") or raw_data.get("initialState")

        tile_hex_states = {}
        if initial_state:
            tile_hex_states = initial_state.get("mapState", {}).get("tileHexStates", {})
            print(f"Loaded {len(tile_hex_states)} tile hex states for robber matching")

        replay_player_ids = (
            raw_data["data"].get("playOrder", [])
            if "data" in raw_data
            else raw_data.get("playOrder", [])
        )
        parsed_actions = parse_colonist_events_to_actions(
            events,
            tile_hex_states,
            player_ids=replay_player_ids,
        )
        paired_transcript = load_paired_transcript(game_id, events, parsed_actions)
        archived_player_perspective = (
            raw_data.get("data", {}).get("playerPerspective")
            if "data" in raw_data
            else raw_data.get("playerPerspective")
        )
        paired_model_traces = load_paired_model_traces(
            game_id,
            narrator=(paired_transcript or {}).get("narrator"),
            archived_player_perspective=archived_player_perspective,
        )
        paired_narrator_reasoning = load_paired_narrator_reasoning(
            game_id,
            paired_transcript,
        )

        COLONIST_COLOR_NAMES = {
            1: "red", 2: "blue", 3: "orange", 4: "green", 5: "black",
            6: "bronze", 7: "silver", 8: "gold", 9: "white",
            10: "pink", 11: "mystic_blue",
        }

        COLONIST_TO_ENGINE_COLOR = {
            1: Color.RED, 2: Color.BLUE, 3: Color.ORANGE, 4: Color.GREEN,
            5: Color.BLACK, 6: Color.BRONZE, 7: Color.SILVER,
            8: Color.GOLD, 9: Color.WHITE, 10: Color.PINK,
            11: Color.MYSTIC_BLUE,
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
            colors = []
            fallback_idx = 0
            for color_id in play_order_colors:
                if color_id in COLONIST_TO_ENGINE_COLOR:
                    engine_color = COLONIST_TO_ENGINE_COLOR[color_id]
                else:
                    engine_color = FALLBACK_ENGINE_COLORS[fallback_idx % len(FALLBACK_ENGINE_COLORS)]
                    fallback_idx += 1
                    print(f"Warning: Unknown Colonist color ID {color_id}, using fallback {engine_color}")
                colors.append(engine_color)
            print(f"Created players with colors: {colors}")
        else:
            colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]

        engine = GameEngine(
            colors,
            catan_map=catan_map,
            shuffle_players=False,
            capture_history=True,
        )

        state.replay_data = {
            "game_id": game_id,
            "events": events,
            "parsed_actions": parsed_actions,
            "total_events": len(parsed_actions),
            "file": str(replay_file),
            "initial_state": initial_state or {},
            "end_game_state": raw_data.get("data", {}).get("eventHistory", {}).get("endGameState", {})
            if "data" in raw_data
            else raw_data.get("eventHistory", {}).get("endGameState", {}),
            "game_settings": raw_data.get("data", {}).get("gameSettings", {})
            if "data" in raw_data
            else raw_data.get("gameSettings", {}),
            "colonist_players": colonist_players,
            "play_order": play_order_indices,
            "colonist_color_to_engine_idx": {str(color_id): idx for idx, color_id in enumerate(play_order_colors)},
            "tile_hex_states": tile_hex_states,
            "player_perspective": archived_player_perspective,
            "paired_transcript": paired_transcript,
            "paired_model_traces": paired_model_traces,
            "paired_narrator_reasoning": paired_narrator_reasoning,
        }
        state.replay_index = 0
        state.replay_actions_per_step = []
        state.first_divergence_step = {}
        state.replay_semantic_issues = []
        state.replay_final_state_synced = False
        state.replay_pending_dev_card = None
        state.replay_step_checkpoints = []
        state.replay_trade_ledger = {}
        state.current_sandbox = ReplaySandbox(
            state,
            engine,
            players={color: ScriptedPlayer(color) for color in engine.state.colors},
        )
        bump_replay_revision(state)
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
            "has_paired_transcript": paired_transcript is not None,
            "has_paired_model_traces": paired_model_traces is not None,
            "has_paired_narrator_reasoning": paired_narrator_reasoning is not None,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"Invalid JSON: {e}"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@replay_bp.route('/api/load-replay', methods=['POST'])
def load_replay():
    """Load a Colonist replay for playback - runs through our engine."""
    state = _get_state()
    with state.replay_mutation_lock:
        return _load_replay_transaction(state)


@replay_bp.route('/api/replay-step', methods=['POST'])
def replay_step():
    """Step through the loaded replay."""
    state = _get_state()
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.step(_broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-undo', methods=['POST'])
def replay_undo():
    """Undo the last replay step."""
    state = _get_state()
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.undo(_broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-fast', methods=['POST'])
def replay_goto_fast():
    """Jump to a specific replay step using fast skip logic."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_fast(target_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-sequential', methods=['POST'])
def replay_goto_sequential():
    """Jump to a specific replay step using sequential stepping."""
    state = _get_state()
    data = request.get_json() or {}
    target_step = data.get("step", 0)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_sequential(target_step, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-goto-divergence', methods=['POST'])
def replay_goto_divergence():
    """Step sequentially until divergence is detected."""
    state = _get_state()
    data = request.get_json() or {}
    max_steps = data.get("max_steps", 500)
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return jsonify({"error": "No replay sandbox loaded"}), 400
    result = sandbox.goto_divergence(max_steps, _broadcast)
    if isinstance(result, tuple):
        return jsonify(result[0]), result[1]
    return jsonify(result)


@replay_bp.route('/api/replay-llm-response', methods=['POST'])
def replay_llm_response():
    """Generate a non-mutating LLM decision for the current replay position."""
    state = _get_state()
    if not state.replay_mode or not state.replay_data or not state.current_sandbox:
        return jsonify({"error": "No replay loaded"}), 400
    sandbox = state.current_sandbox

    data = request.get_json(silent=True) or {}
    try:
        model = validate_model_id(data.get("model"))
        game_plan = validate_game_plan(data.get("game_plan"))
        reasoning_request = validate_native_reasoning_request(data.get("reasoning"))
        temperature = float(data.get("temperature", 0.2))
        max_tokens = int(data.get("max_tokens", 8_192))
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
    except (HarnessValidationError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400

    if not state.replay_llm_lock.acquire(blocking=False):
        return jsonify({"error": "A replay LLM response is already being generated"}), 429

    try:
        result = sandbox_async_runtime.run(
            generate_decision_preview(
                sandbox,
                model=model,
                game_plan=game_plan,
                reasoning_request=reasoning_request,
                temperature=temperature,
                max_tokens=max_tokens,
                transport_factory=current_app.config.get(
                    "REPLAY_COMPLETION_TRANSPORT_FACTORY"
                ),
            )
        )
    except ValueError as exc:
        status = 503 if "API_KEY" in str(exc) else 409
        return jsonify({"error": str(exc)}), status
    except HTTPError as exc:
        return jsonify({"error": f"OpenRouter request failed: {exc}"}), 502
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Replay LLM response failed: {exc}"}), 500
    finally:
        state.replay_llm_lock.release()

    return jsonify(result)
