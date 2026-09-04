"""Thin live-game adapter: create one sandbox and advance one full step."""

import time
from dataclasses import asdict, replace
from typing import Any, Mapping

from flask import Blueprint, current_app, jsonify, request

from cle.harness.reasoning import (
    native_reasoning_request,
    reasoning_token_count,
    validate_native_reasoning_request,
)
from cle.sandbox.catan import PlayerResponseError
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_REASONING_EFFORT,
    LivePromptSuiteSource,
    LiveSandboxConfig,
    create_live_sandbox,
    materialize_live_prompt_suites,
    resolve_live_model,
)
from cle.game_engine.communication import CommunicationLimits
from cle.game_engine.trading import TradeLimits

from ..async_runtime import sandbox_async_runtime
from ..live.game_logging import (
    analyze_transitions,
    format_action_for_display,
    normalize_public_state_game_log,
)
from ..live.reasoning_trace import build_live_reasoning_traces
from ..state import bump_replay_revision
from .websocket import broadcast_game_state, build_game_state_snapshot

live_game_bp = Blueprint("live_game", __name__)


def _get_state():
    return current_app.config["SERVER_STATE"]


def _player_is_agent(sandbox, color):
    player = sandbox.players.get(color)
    return bool(player and player.status().get("kind") == "agent")


def _live_inference_payload(config: LiveSandboxConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "reasoning": validate_native_reasoning_request(config.reasoning),
        "max_tokens": config.max_tokens,
        "max_decision_attempts": config.max_decision_attempts,
    }


def _stored_suite_source(value: Any) -> LivePromptSuiteSource | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Stored prompt suite metadata must be a mapping")
    required = {"id", "version", "sha256", "source"}
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"Stored prompt suite metadata is missing: {sorted(missing)}"
        )
    return LivePromptSuiteSource(
        id=str(value["id"]),
        version=str(value["version"]),
        sha256=str(value["sha256"]),
        source=str(value["source"]),
    )


def _optional_max_tokens(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_tokens must be a positive integer or null")
    parsed = int(value)
    if parsed < 1:
        raise ValueError("max_tokens must be a positive integer or null")
    return parsed


def _config_from_stored_payload(payload: Mapping[str, Any]) -> LiveSandboxConfig:
    seed = payload.get("seed")
    mode = payload.get("mode", "random")
    return LiveSandboxConfig(
        mode=mode,
        model=(
            None
            if mode == "random"
            else resolve_live_model(payload.get("model"))
        ),
        temperature=float(payload.get("temperature", 0.3)),
        max_tokens=_optional_max_tokens(payload.get("max_tokens")),
        reasoning=payload.get("reasoning"),
        seed=int(seed) if seed is not None else None,
        shuffle_players=bool(payload.get("shuffle_players", True)),
        palette=payload.get("palette", "random_all"),
        board_surface=payload.get("board_surface", "legacy_semantic"),
        context_suite_path=payload.get("context_suite_path"),
        communication_suite_path=payload.get("communication_suite_path"),
        decision_suite=_stored_suite_source(payload.get("decision_suite")),
        communication_suite=_stored_suite_source(
            payload.get("communication_suite")
        ),
        max_decision_attempts=int(
            payload.get(
                "max_decision_attempts",
                DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
            )
        ),
        trade_limits=TradeLimits(**(payload.get("trade_limits") or {})),
        communication_limits=CommunicationLimits(**(payload.get("communication_limits") or {})),
    )


def _prepare_live_state(state) -> None:
    state.step_processing = False
    state.live_inference = None
    state.last_live_step_error = None
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
            else validate_native_reasoning_request(
                data.get("reasoning")
                or native_reasoning_request(DEFAULT_LIVE_REASONING_EFFORT)
            )
        )
        model = (
            None
            if mode == "random"
            else resolve_live_model(data.get("model"))
        )
        seed = data.get("seed")
        trade_limits = TradeLimits(**(data.get("trade_limits") or {}))
        communication_limits = CommunicationLimits(**(data.get("communication_limits") or {}))
        config = materialize_live_prompt_suites(LiveSandboxConfig(
            mode=mode,
            model=model,
            temperature=float(data.get("temperature", 0.3)),
            max_tokens=_optional_max_tokens(data.get("max_tokens")),
            reasoning=reasoning,
            seed=int(seed) if seed is not None else None,
            shuffle_players=bool(data.get("shuffle_players", True)),
            palette=data.get("palette", "random_all"),
            board_surface=data.get("board_surface", "indexed_tile_rows"),
            max_decision_attempts=int(
                data.get(
                    "max_decision_attempts",
                    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
                )
            ),
            trade_limits=trade_limits,
            communication_limits=communication_limits,
        ))
        sandbox = create_live_sandbox(config)
        trace_store = getattr(state, "live_trace_store", None)
        trace_game_id = str(sandbox.game_engine.id)
        realized_colors = [color.value for color in sandbox.game_engine.state.colors]
        if trace_store is not None:
            trace_config = {
                **asdict(config),
                "realized_colors": realized_colors,
            }
            trace_store.start_game(
                trace_game_id,
                config=trace_config,
                snapshot=sandbox.snapshot(),
                display_name=data.get("name"),
            )
            trace_display_name = trace_store.get_game(trace_game_id)["display_name"]
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
    state.live_inference = _live_inference_payload(config)

    state.game_log.append(
        {
            "type": "general",
            "timestamp": time.time(),
            "message": f"Game started with {len(sandbox.game_engine.state.colors)} players",
        }
    )

    state_snapshot = build_game_state_snapshot(state)

    player_types = {
        color.name: "LLM" if _player_is_agent(sandbox, color) else "Random"
        for color in sandbox.game_engine.state.colors
    }

    bump_replay_revision(state)
    return jsonify(
        {
            "status": "started",
            "players": [str(color) for color in sandbox.game_engine.state.colors],
            "player_types": player_types,
            "mode": mode,
            "model": config.model,
            "palette": config.palette,
            "board_surface": config.board_surface,
            "realized_colors": realized_colors,
            "trade_limits": asdict(config.trade_limits),
            "communication_limits": asdict(config.communication_limits),
            "reasoning_request": dict(reasoning),
            "max_tokens": config.max_tokens,
            "max_decision_attempts": config.max_decision_attempts,
            "trace_game_id": state.live_trace_game_id,
            "trace_display_name": (trace_display_name if trace_store is not None else None),
            "trace_database": (str(trace_store.path) if trace_store is not None else None),
            "state": state_snapshot,
        }
    )


@live_game_bp.route("/api/start-game", methods=["POST"])
def start_game():
    """Start a new game and always return a JSON response."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _start_game_transaction(state)
    except Exception as exc:
        current_app.logger.exception("Unhandled live sandbox start failure")
        return (
            jsonify(
                {
                    "error": "Sandbox start failed",
                    "details": str(exc),
                }
            ),
            500,
        )


def _failed_attempt_payload(attempt):
    response = attempt.model_response
    choice = attempt.choice
    final_response = getattr(response, "content", "") if response is not None else ""
    usage = dict(getattr(response, "usage", ())) if response is not None else {}
    return {
        "validation_error": attempt.validation_error,
        "action_index": choice.action_index if choice is not None else None,
        "final_response": final_response[:4_000],
        "model": getattr(response, "model", None),
        "latency_ms": getattr(response, "latency_ms", None),
        "finish_reason": getattr(response, "finish_reason", None),
        "provider_native_finish_reason": getattr(
            response,
            "provider_native_finish_reason",
            None,
        ),
        "usage": usage,
        "reasoning_tokens": reasoning_token_count(usage),
        "native_reasoning_chars": len(
            getattr(response, "native_reasoning", "")
        ) if response is not None else 0,
        "provider_response_id": getattr(
            response,
            "provider_response_id",
            None,
        ),
        "provider_request_id": getattr(
            response,
            "provider_request_id",
            None,
        ),
    }


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
    state.last_live_step_error = None
    try:
        result = sandbox_async_runtime.run(sandbox.step())
    except PlayerResponseError as exc:
        current_app.logger.warning(
            "Live player %s returned no valid action after %s attempt(s): %s",
            exc.player,
            len(exc.attempts),
            exc.validation_error,
        )
        error_payload = {
            "error": "Model returned no valid action",
            "details": (
                f"{exc.validation_error} No gameplay action was applied. "
                "Press Step to ask the model again."
            ),
            "player": str(actor),
            "attempt_count": len(exc.attempts),
            "attempts": [
                _failed_attempt_payload(attempt)
                for attempt in exc.attempts
            ],
            "retryable": True,
        }
        state.last_live_step_error = error_payload
        broadcast_game_state(current_app.config["SOCKETIO"], state)
        return jsonify(error_payload), 422
    except Exception as exc:
        current_app.logger.exception("Live sandbox step failed")
        return (
            jsonify(
                {
                    "error": "Sandbox step failed",
                    "details": str(exc).strip() or type(exc).__name__,
                    "player": str(actor),
                }
            ),
            500,
        )
    finally:
        state.step_processing = False

    analyze_transitions(
        state,
        result.transitions,
        pre_game_state,
        sandbox.game_engine.state,
    )
    transition = result.transitions[-1]

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
    return jsonify(
        {
            "status": "ok",
            "action": format_action_for_display(
                transition.requested_action,
                include_actor=True,
            ),
            "actions": [
                format_action_for_display(item.requested_action, include_actor=True)
                for item in result.transitions
            ],
            "game_over": winner is not None,
            "winner": str(winner) if winner else None,
            "reasoning_traces": reasoning_traces,
            "trace_game_id": state.live_trace_game_id,
            "trace_step_index": trace_step_index,
            "state": state_snapshot,
        }
    )


@live_game_bp.route("/api/step", methods=["POST"])
def step_game():
    """Execute one live-game step and always return a JSON response."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _step_game_transaction(state)
    except Exception as exc:
        state.step_processing = False
        current_app.logger.exception("Unhandled live sandbox step failure")
        return (
            jsonify(
                {
                    "error": "Sandbox step failed",
                    "details": str(exc),
                }
            ),
            500,
        )


@live_game_bp.route("/api/live-traces", methods=["GET"])
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
    return jsonify(
        {
            "database": str(trace_store.path),
            "games": trace_store.list_games(limit=limit),
        }
    )


@live_game_bp.route("/api/live-traces/<game_id>", methods=["GET"])
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


@live_game_bp.route("/api/live-traces/<game_id>", methods=["PATCH"])
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
    return jsonify(
        {
            "status": "renamed",
            "game_id": game_id,
            "display_name": trace["display_name"],
        }
    )


def _load_live_trace_transaction(state, game_id):
    trace_store = getattr(state, "live_trace_store", None)
    if trace_store is None:
        return jsonify({"error": "Live trace storage is disabled"}), 503
    try:
        resume_point = trace_store.load_resume_point(game_id)
    except KeyError:
        return jsonify({"error": "Live trace not found"}), 404

    try:
        config = replace(
            _config_from_stored_payload(resume_point.config),
            reasoning=native_reasoning_request(DEFAULT_LIVE_REASONING_EFFORT),
            max_tokens=None,
            max_decision_attempts=DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
        )
        sandbox = create_live_sandbox(config, snapshot=resume_point.snapshot)
    except (OSError, TypeError, ValueError) as exc:
        return (
            jsonify(
                {
                    "error": "Saved live game cannot be restored",
                    "details": str(exc),
                }
            ),
            409,
        )

    previous_game_id = state.live_trace_game_id
    if previous_game_id is not None and previous_game_id != game_id:
        trace_store.mark_game_status(previous_game_id, status="replaced")

    _prepare_live_state(state)
    state.current_sandbox = sandbox
    state.live_inference = _live_inference_payload(config)
    winner = sandbox.game_engine.winning_color()
    state.game_running = winner is None
    prior_public_state = normalize_public_state_game_log(
        resume_point.public_state or {}
    )
    stored_game_log = prior_public_state.get("game_log", [])
    state.game_log = stored_game_log if isinstance(stored_game_log, list) else []
    state.live_trace_game_id = game_id
    trace_store.mark_game_status(
        game_id,
        status="running" if winner is None else "completed",
        winner=winner.value if winner is not None else None,
    )
    bump_replay_revision(state)
    state_snapshot = broadcast_game_state(
        current_app.config["SOCKETIO"],
        state,
    )
    return jsonify(
        {
            "status": "loaded",
            "trace_game_id": game_id,
            "trace_display_name": resume_point.display_name,
            "trace_database": str(trace_store.path),
            "loaded_step_index": resume_point.step_index,
            "mode": config.mode,
            "model": config.model,
            "reasoning_request": dict(
                validate_native_reasoning_request(config.reasoning)
            ),
            "max_tokens": config.max_tokens,
            "max_decision_attempts": config.max_decision_attempts,
            "state": state_snapshot,
        }
    )


@live_game_bp.route(
    "/api/live-traces/<game_id>/steps/<int:step_index>",
    methods=["GET"],
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
    step = checkpoint.get("step")
    if isinstance(step, dict):
        checkpoint = dict(checkpoint)
        checkpoint["step"] = dict(step)
        checkpoint["step"]["public_state"] = normalize_public_state_game_log(
            step.get("public_state")
        )
    return jsonify(checkpoint)


@live_game_bp.route("/api/live-traces/<game_id>/load", methods=["POST"])
def load_live_trace(game_id):
    """Restore the latest durable checkpoint and continue the same trace."""
    state = _get_state()
    try:
        with state.replay_mutation_lock:
            return _load_live_trace_transaction(state, game_id)
    except Exception as exc:
        state.step_processing = False
        current_app.logger.exception("Unhandled saved live-game load failure")
        return (
            jsonify(
                {
                    "error": "Saved live game load failed",
                    "details": str(exc),
                }
            ),
            500,
        )
