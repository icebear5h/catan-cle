"""Live game endpoints: start, step, auto-play, stop."""

import io
import os
import time
from pathlib import Path

from flask import Blueprint, jsonify, request, current_app, send_file

from engine.game import Game
from engine.models.player import SimplePlayer, Color
from cle.agents.llm_player import LLMPlayer
from engine.state_functions import get_visible_victory_points

from ..live.game_logging import analyze_action, post_analyze_action
from ..live.runner import run_game_auto
from .websocket import broadcast_game_state

_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "screenshots"

live_game_bp = Blueprint('live_game', __name__)


def _get_state():
    return current_app.config['SERVER_STATE']


def _broadcast():
    state = _get_state()
    socketio = current_app.config['SOCKETIO']
    broadcast_game_state(socketio, state)


@live_game_bp.route('/api/start-game', methods=['POST'])
def start_game():
    """Start a new game with configured players."""
    state = _get_state()

    state.auto_play_running = False
    state.llm_processing = False
    state.replay_data = None
    state.replay_index = 0
    state.replay_mode = False
    state.replay_actions_per_step = []
    state.replay_step_checkpoints = []
    state.replay_trade_ledger = {}

    data = request.json or {}
    # Support both new `mode` param and legacy `use_llm` boolean
    mode = data.get('mode')
    if mode is None:
        use_llm = data.get('use_llm', False)
        mode = 'llm' if use_llm else 'random'

    if mode in ('llm', 'llm_vs_random'):
        has_vlm = os.getenv("NOVITA_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        has_groq = os.getenv("GROQ_API_KEY")
        if not has_vlm and not has_groq:
            return jsonify({
                "error": "No API key set",
                "message": "Set NOVITA_API_KEY (vision) or OPENROUTER_API_KEY (vision) or GROQ_API_KEY (text-only)"
            }), 400

    if mode == 'llm':
        players = [
            LLMPlayer(Color.RED), LLMPlayer(Color.BLUE),
            LLMPlayer(Color.WHITE), LLMPlayer(Color.ORANGE),
        ]
    elif mode == 'llm_vs_random':
        players = [
            LLMPlayer(Color.RED),
            SimplePlayer(Color.BLUE), SimplePlayer(Color.WHITE),
            SimplePlayer(Color.ORANGE),
        ]
    else:
        players = [
            SimplePlayer(Color.RED), SimplePlayer(Color.BLUE),
            SimplePlayer(Color.WHITE), SimplePlayer(Color.ORANGE),
        ]

    state.current_game = Game(players)
    state.current_players = players
    state.game_running = True
    state.llm_thinking = []
    state.game_log = []

    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"Game started with {len(players)} players"
    })

    _broadcast()

    player_types = {}
    for player in players:
        color_str = player.color.name if hasattr(player.color, 'name') else str(player.color)
        if isinstance(player, LLMPlayer):
            player_types[color_str] = "LLM"
        else:
            player_types[color_str] = "Random"

    return jsonify({
        "status": "started",
        "players": [str(p.color) for p in players],
        "player_types": player_types,
        "mode": mode
    })


@live_game_bp.route('/api/step', methods=['POST'])
def step_game():
    """Execute one game step."""
    state = _get_state()
    game = state.current_game

    if not game or not state.game_running:
        return jsonify({"error": "No game running"}), 400

    if state.llm_processing:
        return jsonify({"error": "LLM is still thinking, please wait"}), 429

    current_player = game.state.current_player()

    print(f"\n{'*'*80}")
    print(f"STEP: Turn {game.state.num_turns} | Player: {current_player.color} | Type: {'LLM' if isinstance(current_player, LLMPlayer) else 'Random'}")
    print(f"{'*'*80}")

    print(f"\nGAME STATE SNAPSHOT:")
    print(f"  Phase: {'Initial Placement' if game.state.is_initial_build_phase else 'Main Game'}")
    print(f"  Current player: {current_player.color}")
    print(f"  Playable actions: {len(game.state.playable_actions)}")

    print(f"\n  Victory Points:")
    for i, color in enumerate(game.state.colors):
        vp = get_visible_victory_points(game.state, color)
        player_type = "LLM" if i < len(state.current_players) and isinstance(state.current_players[i], LLMPlayer) else "Random"
        print(f"    {color}: {vp} VP ({player_type})")

    decision_info = {
        "color": str(current_player.color),
        "is_llm": isinstance(current_player, LLMPlayer),
        "timestamp": time.time()
    }

    if not game.state.playable_actions:
        return jsonify({"error": "No valid actions"}), 400

    decision_info["available_actions"] = [str(a) for a in game.state.playable_actions]

    if isinstance(current_player, LLMPlayer):
        state.llm_processing = True

    try:
        action = current_player.decide(game, game.state.playable_actions)

        action_description = f"{current_player.color} performed: {action}"
        for i, player in enumerate(state.current_players):
            if isinstance(player, LLMPlayer) and player.color != current_player.color:
                player.log_event(action_description)
    except Exception as e:
        if isinstance(current_player, LLMPlayer):
            state.llm_processing = False

        print(f"\n{'='*80}")
        print(f"ERROR: Exception during player decision")
        print(f"{'='*80}")
        print(f"Player: {current_player.color}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")

        return jsonify({
            "error": "Player decision failed",
            "details": str(e),
            "player": str(current_player.color)
        }), 500
    finally:
        if isinstance(current_player, LLMPlayer):
            state.llm_processing = False

    decision_info["action"] = str(action)

    if isinstance(current_player, LLMPlayer):
        if hasattr(current_player, 'last_reasoning') and current_player.last_reasoning:
            decision_info["reasoning"] = current_player.last_reasoning
        if hasattr(current_player, 'last_game_plan') and current_player.last_game_plan:
            decision_info["game_plan"] = current_player.last_game_plan
        if hasattr(current_player, 'last_observation') and current_player.last_observation:
            decision_info["observation"] = current_player.last_observation
        if hasattr(current_player, 'strategic_notes') and current_player.strategic_notes:
            decision_info["strategic_notes"] = current_player.strategic_notes
        if hasattr(current_player, 'last_screenshot') and current_player.last_screenshot:
            decision_info["has_screenshot"] = True

    pre_state = analyze_action(state, action, game.state)

    try:
        print(f"\nEXECUTING ACTION:")
        print(f"  {action}")
        exec_start = time.time()
        game.execute(action)
        exec_time = time.time() - exec_start
        print(f"  Execution time: {exec_time:.3f}s")
        print(f"  Success!")
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR: Exception during action execution")
        print(f"{'='*80}")
        print(f"Action: {action}")
        print(f"Error: {str(e)}")
        print(f"{'='*80}\n")

        return jsonify({
            "error": "Action execution failed",
            "details": str(e),
            "action": str(action)
        }), 500

    post_analyze_action(state, pre_state, game.state)

    print(f"\nSTATE AFTER ACTION:")
    print(f"  Turn: {game.state.num_turns}")
    print(f"  Current player: {game.state.current_player().color}")
    print(f"  Next actions available: {len(game.state.playable_actions)}")

    winner = game.winning_color()
    if winner:
        state.game_running = False
        decision_info["game_over"] = True
        decision_info["winner"] = str(winner)

    state.llm_thinking.append(decision_info)

    _broadcast()

    print(f"{'*'*80}")
    print(f"STEP COMPLETE")
    print(f"{'*'*80}\n")

    return jsonify({
        "status": "ok",
        "action": str(action),
        "game_over": winner is not None,
        "winner": str(winner) if winner else None
    })


@live_game_bp.route('/api/auto-play', methods=['POST'])
def auto_play():
    """Play the game automatically until completion."""
    state = _get_state()
    socketio = current_app.config['SOCKETIO']

    if not state.current_game or not state.game_running:
        return jsonify({"error": "No game running"}), 400

    data = request.json or {}
    delay = data.get('delay', 0.5)

    state.auto_play_running = True

    socketio.start_background_task(run_game_auto, state, delay, _broadcast)

    return jsonify({"status": "running"})


@live_game_bp.route('/api/stop-auto-play', methods=['POST'])
def stop_auto_play():
    """Stop the auto-play."""
    state = _get_state()
    state.auto_play_running = False
    return jsonify({"status": "stopped"})


@live_game_bp.route('/api/latest-screenshot')
def latest_screenshot():
    """Serve the latest board screenshot taken by the LLM player."""
    state = _get_state()
    if state.current_players:
        for p in state.current_players:
            if isinstance(p, LLMPlayer) and p.last_screenshot:
                return send_file(
                    io.BytesIO(p.last_screenshot),
                    mimetype='image/png',
                )
    # Fallback: try the file on disk
    latest_path = _SCREENSHOT_DIR / "latest.png"
    if latest_path.exists():
        return send_file(str(latest_path), mimetype='image/png')
    return jsonify({"error": "No screenshot available"}), 404
