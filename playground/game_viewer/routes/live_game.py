"""Thin live-game adapter: create one sandbox and advance one full step."""

import time
from dataclasses import asdict
from typing import Any, Mapping

from flask import Blueprint, current_app, jsonify, request

from cle.harness.reasoning import validate_native_reasoning_request
from cle.sandbox.factory import LiveSandboxConfig, create_live_sandbox
from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.trading import TradeLimits

from ..async_runtime import sandbox_async_runtime
from ..live.game_logging import analyze_action, post_analyze_action
from ..live.reasoning_trace import build_live_reasoning_traces
from ..state import bump_replay_revision
from .websocket import build_game_state_snapshot

live_game_bp = Blueprint('live_game', __name__)


def _get_state():
    return current_app.config['SERVER_STATE']


def _player_is_agent(sandbox, color):
    player = sandbox.players.get(color)
    return bool(player and player.status().get("kind") == "agent")


def _config_from_stored_payload(payload: Mapping[str, Any]) -> LiveSandboxConfig:
    seed = payload.get("seed")
    return LiveSandboxConfig(
        mode=payload.get("mode", "random"),
        model=payload.get("model"),
        temperature=float(payload.get("temperature", 0.3)),
        max_tokens=int(payload.get("max_tokens", 8_192)),
        reasoning=payload.get("reasoning"),
        seed=int(seed) if seed is not None else None,
        shuffle_players=bool(payload.get("shuffle_players", True)),
        context_suite_path=payload.get("context_suite_path"),
        max_decision_attempts=int(payload.get("max_decision_attempts", 3)),
        trade_limits=TradeLimits(**(payload.get("trade_limits") or {})),
        communication_limits=CommunicationLimits(
            **(payload.get("communication_limits") or {})
        ),
    )


def _prepare_live_state(state) -> None:
    state.step_processing = False
    state.replay_data = None
    state.replay_index = 0
    state.replay_mode = False
    state.replay_actions_per_step = []
    state.replay_step_checkpoints = []
    state.replay_trade_ledger = {}


def _start_game_transaction(state):
    _prepare_live_state(state)

    data = request.json or {}
    mode = data.get("mode", "random")

    try:
        reasoning = (
            {"enabled": False}
            if mode == "random"
            else validate_native_reasoning_request(data.get("reasoning"))
        )
        seed = data.get("seed")
        trade_limits = TradeLimits(**(data.get("trade_limits") or {}))
        communication_limits = CommunicationLimits(
            **(data.get("communication_limits") or {})
        )
        config = LiveSandboxConfig(
            mode=mode,
            model=data.get("model"),
            temperature=float(data.get("temperature", 0.3)),
            max_tokens=int(data.get("max_tokens", 8_192)),
            reasoning=reasoning,
            seed=int(seed) if seed is not None else None,
            shuffle_players=bool(data.get("shuffle_players", True)),
            max_decision_attempts=int(data.get("max_decision_attempts", 3)),
            trade_limits=trade_limits,
            communication_limits=communication_limits,
        )
        sandbox = create_live_sandbox(config)
        trace_store = getattr(state, "live_trace_store", None)
        trace_game_id = str(sandbox.game_engine.id)
        if trace_store is not None:
            trace_store.start_game(
                trace_game_id,
                config=config,
                snapshot=sandbox.snapshot(),
                display_name=data.get("name"),
            )
            trace_display_name = trace_store.get_game(trace_game_id)[
                "display_name"
            ]
            if state.live_trace_game_id is not None:
                trace_store.mark_game_status(
                    state.live_trace_game_id,
                    status="replaced",
                )
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({"error": "Invalid live-game configuration", "message": str(exc)}), 400

    state.current_sandbox = sandbox
    state.game_running = True
    state.game_log = []
    state.live_trace_game_id = trace_game_id if trace_store is not None else None

    state.game_log.append({
        "type": "general",
        "timestamp": time.time(),
        "message": f"Game started with {len(sandbox.game_engine.state.colors)} players"
    })

    state_snapshot = build_game_state_snapshot(state)

    player_types = {
        color.name: "LLM" if _player_is_agent(sandbox, color) else "Random"
        for color in sandbox.game_engine.state.colors
    }

    bump_replay_revision(state)
    return jsonify({
        "status": "started",
        "players": [str(color) for color in sandbox.game_engine.state.colors],
        "player_types": player_types,
        "mode": mode,
        "trade_limits": asdict(config.trade_limits),
        "communication_limits": asdict(config.communication_limits),
        "reasoning_request": dict(reasoning),
        "trace_game_id": state.live_trace_game_id,
        "trace_display_name": (
            trace_display_name if trace_store is not None else None
        ),
        "trace_database": (
            str(trace_store.path) if trace_store is not None else None
        ),
        "state": state_snapshot,
    })


@live_game_bp.route('/api/start-game', methods=['POST'])
def start_game():
    """Start a new game and always return a JSON response."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _start_game_transaction(state)
    except Exception as exc:
        current_app.logger.exception("Unhandled live sandbox start failure")
        return jsonify({
            "error": "Sandbox start failed",
            "details": str(exc),
        }), 500


def _step_game_transaction(state):
    sandbox = getattr(state, "current_sandbox", None)

    if state.replay_mode:
        return jsonify({"error": "Use replay controls while a replay is loaded"}), 409
    if sandbox is None or not state.game_running:
        return jsonify({"error": "No live sandbox running"}), 400
    if state.step_processing:
        return jsonify({"error": "A sandbox step is already in progress"}), 429

    actor = sandbox.current_actor()
    pre_game_state = sandbox.game_engine.state.copy()
    rejected_attempt_cursor = len(sandbox.decision_trace)
    communication_cursor = len(sandbox.communication_trace)

    state.step_processing = True
    try:
        result = sandbox_async_runtime.run(sandbox.step())
    except Exception as exc:
        return jsonify({
            "error": "Sandbox step failed",
            "details": str(exc),
            "player": str(actor),
        }), 500
    finally:
        state.step_processing = False

    transition = result.transitions[-1]
    pre_state = analyze_action(state, transition.requested_action, pre_game_state)
    post_analyze_action(state, pre_state, sandbox.game_engine.state)

    winner = result.winner
    if winner is not None:
        state.game_running = False

    reasoning_traces = build_live_reasoning_traces(sandbox, result)
    state_snapshot = build_game_state_snapshot(state)
    trace_step_index = None
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is not None and state.live_trace_game_id is not None:
        trace_step_index = trace_store.record_step(
            state.live_trace_game_id,
            result=result,
            rejected_attempts=sandbox.decision_trace[rejected_attempt_cursor:],
            communication_attempts=sandbox.communication_trace[communication_cursor:],
            public_state=state_snapshot,
            snapshot=sandbox.snapshot(),
        )
    return jsonify({
        "status": "ok",
        "action": str(transition.requested_action),
        "actions": [str(item.requested_action) for item in result.transitions],
        "game_over": winner is not None,
        "winner": str(winner) if winner else None,
        "reasoning_traces": reasoning_traces,
        "trace_game_id": state.live_trace_game_id,
        "trace_step_index": trace_step_index,
        "state": state_snapshot,
    })


@live_game_bp.route('/api/step', methods=['POST'])
def step_game():
    """Execute one live-game step and always return a JSON response."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _step_game_transaction(state)
    except Exception as exc:
        state.step_processing = False
        current_app.logger.exception("Unhandled live sandbox step failure")
        return jsonify({
            "error": "Sandbox step failed",
            "details": str(exc),
        }), 500


@live_game_bp.route('/api/live-traces', methods=['GET'])
def list_live_traces():
    """List locally persisted live games without loading snapshot blobs."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400
    return jsonify({
        "database": str(trace_store.path),
        "games": trace_store.list_games(limit=limit),
    })


@live_game_bp.route('/api/live-traces/<game_id>', methods=['GET'])
def get_live_trace(game_id):
    """Return one locally persisted game with steps and model calls."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    trace = trace_store.get_game(game_id)
    if trace is None:
        return jsonify({"error": "Live trace not found"}), 404
    return jsonify(trace)


@live_game_bp.route('/api/live-traces/<game_id>', methods=['PATCH'])
def rename_live_trace(game_id):
    """Set or clear the user-facing name for one saved live game."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "name" not in data:
        return jsonify({"error": "A name field is required"}), 400
    try:
        renamed = trace_store.rename_game(game_id, data["name"])
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    if not renamed:
        return jsonify({"error": "Live trace not found"}), 404
    trace = trace_store.get_game(game_id)
    return jsonify({
        "status": "renamed",
        "game_id": game_id,
        "display_name": trace["display_name"],
    })


def _load_live_trace_transaction(state, game_id):
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    try:
        resume_point = trace_store.load_resume_point(game_id)
    except KeyError:
        return jsonify({"error": "Live trace not found"}), 404

    try:
        config = _config_from_stored_payload(resume_point.config)
        sandbox = create_live_sandbox(config, snapshot=resume_point.snapshot)
    except (OSError, TypeError, ValueError) as exc:
        return jsonify({
            "error": "Saved live game cannot be restored",
            "details": str(exc),
        }), 409

    previous_game_id = state.live_trace_game_id
    if previous_game_id is not None and previous_game_id != game_id:
        trace_store.mark_game_status(previous_game_id, status="replaced")

    _prepare_live_state(state)
    state.current_sandbox = sandbox
    winner = sandbox.game_engine.winning_color()
    state.game_running = winner is None
    prior_public_state = resume_point.public_state or {}
    stored_game_log = prior_public_state.get("game_log", [])
    state.game_log = stored_game_log if isinstance(stored_game_log, list) else []
    state.live_trace_game_id = game_id
    trace_store.mark_game_status(
        game_id,
        status="running" if winner is None else "completed",
        winner=winner.value if winner is not None else None,
    )
    bump_replay_revision(state)
    return jsonify({
        "status": "loaded",
        "trace_game_id": game_id,
        "trace_display_name": resume_point.display_name,
        "trace_database": str(trace_store.path),
        "loaded_step_index": resume_point.step_index,
        "mode": config.mode,
        "state": build_game_state_snapshot(state),
    })


@live_game_bp.route(
    '/api/live-traces/<game_id>/steps/<int:step_index>',
    methods=['GET'],
)
def get_live_trace_step(game_id, step_index):
    """Return one browse-only public checkpoint and its model calls."""
    state = _get_state()
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    checkpoint = trace_store.get_step(game_id, step_index)
    if checkpoint is None:
        return jsonify({"error": "Live trace step not found"}), 404
    return jsonify(checkpoint)


@live_game_bp.route('/api/live-traces/<game_id>/load', methods=['POST'])
def load_live_trace(game_id):
    """Restore the latest durable checkpoint and continue the same trace."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _load_live_trace_transaction(state, game_id)
    except Exception as exc:
        state.step_processing = False
        current_app.logger.exception("Unhandled saved live-game load failure")
        return jsonify({
            "error": "Saved live game load failed",
            "details": str(exc),
        }), 500
