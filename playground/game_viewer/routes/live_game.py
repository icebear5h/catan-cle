"""Thin live-game adapter: create one sandbox and advance one full step."""

import json
import time
from dataclasses import asdict, replace
from typing import Any, Mapping

import yaml
from flask import Blueprint, current_app, jsonify, request

from cle.game_engine.json import GameEncoder
from cle.harness.board_surface import board_presentation_payload
from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.harness.reasoning import (
    native_reasoning_request,
    reasoning_token_count,
    validate_native_reasoning_request,
)
from cle.sandbox.catan import (
    PlayerResponseError,
    PostActionCommunicationCancelled,
    PostActionCommunicationError,
)
from cle.sandbox.factory import (
    DEFAULT_LIVE_MAX_DECISION_ATTEMPTS,
    DEFAULT_LIVE_REASONING_EFFORT,
    LiveSandboxConfig,
    ActivePromptConfigurationError,
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
    stamp_message_step_indexes,
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
    payload = {
        "model": config.model,
        "reasoning": validate_native_reasoning_request(config.reasoning),
        "max_tokens": config.max_tokens,
        "max_decision_attempts": config.max_decision_attempts,
    }
    if config.shared_suite is not None:
        payload["context_policy"] = "fresh_notes"
        payload["shared_suite"] = {
            "id": config.shared_suite.id,
            "version": config.shared_suite.version,
            "sha256": config.shared_suite.sha256,
        }
    return payload


def _applied_prompt_config(sandbox) -> LiveSandboxConfig | None:
    refresh = getattr(sandbox, "_refresh_players", None)
    binding = getattr(refresh, "__self__", None)
    return getattr(binding, "applied", None)


def _sync_live_inference(state, sandbox) -> None:
    if (config := _applied_prompt_config(sandbox)) is not None:
        state.live_inference = _live_inference_payload(config)


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
    """Recover gameplay composition only, never historical inference settings."""
    seed = payload.get("seed")
    mode = payload.get("mode", "random")
    return LiveSandboxConfig(
        mode=mode,
        seed=int(seed) if seed is not None else None,
        shuffle_players=bool(payload.get("shuffle_players", True)),
        palette=payload.get("palette", "random_all"),
        # Stored prompt sources/paths describe history, never active selection.
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
        selection = LiveSandboxConfig(
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
            shared_suite_path=data.get("shared_suite_path"),
            context_suite_path=data.get("context_suite_path"),
            communication_suite_path=data.get("communication_suite_path"),
        )
        config = materialize_live_prompt_suites(selection)
        sandbox = create_live_sandbox(selection)
        config = _applied_prompt_config(sandbox) or config
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
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return jsonify({"error": "Invalid live-game configuration", "message": str(exc)}), 400

    state.current_sandbox = sandbox
    state.active_live_config = selection
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


_MODEL_OUTPUT_EXCERPT_LIMIT = 1000


def _last_model_output_excerpt(attempts) -> str | None:
    """Return the most recent non-empty model output, truncated for banner display."""
    for attempt in reversed(tuple(attempts or ())):
        response = getattr(attempt, "model_response", None)
        content = getattr(response, "content", "")
        if isinstance(content, str) and content.strip():
            text = content.strip()
            if len(text) > _MODEL_OUTPUT_EXCERPT_LIMIT:
                return text[:_MODEL_OUTPUT_EXCERPT_LIMIT] + "... [truncated]"
            return text
    return None


def _failed_attempt_payload(attempt):
    response = attempt.model_response
    choice = attempt.choice
    model_request = attempt.model_request
    if not isinstance(model_request, ModelRequest):
        model_request = None
    request_payload = None
    if model_request is not None:
        request_payload = asdict(replace(model_request, board_presentation=None))
        request_payload["board_presentation"] = board_presentation_payload(
            model_request.board_presentation, include_text_content=True,
        )
    final_response = getattr(response, "content", "") if response is not None else ""
    usage = dict(getattr(response, "usage", ())) if response is not None else {}
    return {
        "context_id": attempt.context_id,
        "validation_error": attempt.validation_error,
        "action_index": getattr(choice, "action_index", None),
        "accepted": False,
        "notes_update": choice.notes_update if choice is not None else None,
        "request": (
            json.loads(json.dumps(request_payload, cls=GameEncoder))
            if model_request is not None else None
        ),
        "context_policy": model_request.context_policy if model_request is not None else None,
        "memory_revision": model_request.memory_revision if model_request is not None else None,
        "input_next_sequence": model_request.input_next_sequence if model_request is not None else None,
        "channel": model_request.channel if model_request is not None else None,
        "final_response": final_response,
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
        "native_reasoning": getattr(response, "native_reasoning", ""),
        "native_reasoning_details": list(
            getattr(response, "native_reasoning_details", ())
        ),
        "reasoning_request": dict(getattr(response, "reasoning_request", ())),
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


def _safe_failure_traces(attempts, communications, error_type):
    """Do not publish arbitrary exception bodies embedded by barrier withholding."""
    return (
        [replace(attempt, validation_error=f"Decision withheld: {error_type}") for attempt in attempts],
        [
            record if record.accepted else replace(
                record, validation_error=f"Communication withheld: {error_type}",
            )
            for record in communications
        ],
    )


def _record_live_failure(
    state, sandbox, player, error_payload, attempts, communication_attempts,
    *, validation_error=None, persistence_failed=False,
):
    """Checkpoint the exact paused boundary, including admitted silence and notes."""
    _sync_live_inference(state, sandbox)
    state.step_processing = False
    state.last_live_step_error = error_payload
    error_payload["trace_failure_id"] = None
    error_payload["checkpoint_saved"] = False
    trace_store = getattr(state, "live_trace_store", None)
    if not persistence_failed and trace_store is not None and state.live_trace_game_id is not None:
        try:
            error_payload["checkpoint_saved"] = True
            error_payload["trace_failure_id"] = trace_store.record_failure(
                state.live_trace_game_id,
                revision=sandbox.revision,
                player=player,
                validation_error=(
                    validation_error if validation_error is not None else error_payload["details"]
                ),
                attempts=attempts,
                communication_attempts=communication_attempts,
                snapshot=sandbox.snapshot(),
                public_state=build_game_state_snapshot(state),
            )
        except Exception as exc:
            current_app.logger.error("Could not persist live failure (%s)", type(exc).__name__)
            persistence_failed = True
    if persistence_failed:
        error_payload["checkpoint_saved"] = False
        error_payload["retryable"] = False
        error_payload["details"] = error_payload["details"].replace(
            " Press Step to ask the model again.", "",
        ).replace(" Press Step to retry.", "") + (
            " Current state and failure diagnostics could not be saved. "
            "Do not retry until storage is repaired; loading a saved game may lose admitted changes."
        )


async def _step_with_committed_result(sandbox):
    try:
        return await sandbox.step()
    except PostActionCommunicationCancelled as exc:
        # The thread bridge otherwise turns cancellation into an untyped Future
        # error, dropping the accepted action's request before it can be saved.
        raise PostActionCommunicationError(exc.result) from exc


def _stamp_recorded_step_messages(
    state, trace_store, trace_step_index, game_log_mark, state_snapshot,
):
    """Label the step's table-talk rows with the trace step index.

    The index only exists once the step is recorded, so speech rows are logged
    with engine-event sequences first and stamped here. The recorded public
    state is rewritten so a loaded checkpoint shows the labels the live board
    shows; the step itself is already durable, so a failed rewrite only costs
    the label.
    """
    if trace_step_index is None:
        return state_snapshot
    if not stamp_message_step_indexes(state.game_log, game_log_mark, trace_step_index):
        return state_snapshot
    state_snapshot = build_game_state_snapshot(state)
    try:
        trace_store.update_step_public_state(
            state.live_trace_game_id, trace_step_index, state_snapshot,
        )
    except Exception as exc:
        current_app.logger.warning(
            "Could not persist table-talk step labels for step %s (%s)",
            trace_step_index, type(exc).__name__,
        )
    return state_snapshot


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
    game_log_mark = len(state.game_log)
    rejected_attempt_cursor = len(sandbox.decision_trace)
    communication_cursor = len(sandbox.communication_trace)
    trace_store = getattr(state, "live_trace_store", None)
    warning = None
    communication_error_type = None

    state.step_processing = True
    state.last_live_step_error = None
    try:
        result = sandbox_async_runtime.run(_step_with_committed_result(sandbox))
    except ActivePromptConfigurationError as exc:
        error_payload = {
            "error": "Active prompt configuration is incompatible",
            "details": str(exc), "action_applied": False, "retryable": False,
            "player": actor.value, "trace_game_id": state.live_trace_game_id,
        }
        _record_live_failure(
            state, sandbox, actor, error_payload,
            sandbox.decision_trace[rejected_attempt_cursor:],
            sandbox.communication_trace[communication_cursor:],
        )
        broadcast_game_state(current_app.config["SOCKETIO"], state)
        return jsonify(error_payload), 409
    except PlayerResponseError as exc:
        current_app.logger.warning(
            "Live player %s returned no valid action after %s attempt(s): %s",
            exc.player,
            len(exc.attempts),
            exc.validation_error,
        )
        action_applied = len(sandbox.game_engine.state.actions) != len(pre_game_state.actions)
        model_excerpt = _last_model_output_excerpt(exc.attempts)
        if model_excerpt is None:
            model_part = " No final model output was returned."
        else:
            model_part = f" Model output: {model_excerpt}"
        error_payload = {
            "error": "Model returned no valid action",
            "details": (
                f"{exc.validation_error}{model_part} " + (
                    "A gameplay action was applied before this error. Do not repeat it; "
                    "inspect the current game before continuing."
                    if action_applied else
                    "No gameplay action was applied. Press Step to ask the model again."
                )
            ),
            "player": exc.player.value,
            "trace_game_id": state.live_trace_game_id,
            "attempt_count": len(exc.attempts),
            "attempts": [
                _failed_attempt_payload(attempt)
                for attempt in exc.attempts
            ],
            "action_applied": action_applied,
            "retryable": not action_applied,
        }
        _record_live_failure(
            state, sandbox, exc.player, error_payload,
            sandbox.decision_trace[rejected_attempt_cursor:],
            sandbox.communication_trace[communication_cursor:],
            validation_error=exc.validation_error,
        )
        broadcast_game_state(current_app.config["SOCKETIO"], state)
        return jsonify(error_payload), 422
    except PostActionCommunicationError as exc:
        provider_failure = exc.__cause__
        if isinstance(provider_failure, OpenRouterHTTPFailure):
            current_app.logger.warning("Post-action communication failed: %s", provider_failure)
        else:
            current_app.logger.warning(
                "Post-action communication failed (%s)", type(provider_failure).__name__,
            )
        if not isinstance(provider_failure, (OpenRouterHTTPFailure, OpenRouterTLSFailure)):
            communication_error_type = type(provider_failure).__name__
        result = exc.result
        warning = {
            "details": (
                "Game action was applied, but post-action communication failed. "
                "Auto-play stopped. Do not retry the applied action; "
                "the next Step advances the game."
            ),
            "action_applied": True,
            "retryable": False,
            "trace_game_id": state.live_trace_game_id,
        }
        if isinstance(provider_failure, ActivePromptConfigurationError):
            warning["details"] += f" {provider_failure}"
        if isinstance(provider_failure, OpenRouterHTTPFailure):
            warning["details"] += (
                f" {provider_failure} Resolve the provider rejection before the next Step."
            )
            warning["provider_status_code"] = provider_failure.status_code
            if provider_failure.provider_request_id:
                warning["provider_request_id"] = provider_failure.provider_request_id
        state.last_live_step_error = warning
    except (OpenRouterTLSFailure, OpenRouterHTTPFailure) as exc:
        http_rejection = isinstance(exc, OpenRouterHTTPFailure)
        failed_actor = next(
            (
                color for color, player in sandbox.players.items()
                if player.status().get("session_id") == exc.session_id
            ),
            None,
        )
        # Speech can advance revision without a gameplay action. Conversely,
        # a callback error after application must never invite an action retry.
        action_applied = len(sandbox.game_engine.state.actions) != len(pre_game_state.actions)
        guidance = (
            "A gameplay action was applied before this error. Do not repeat it; "
            "inspect the current game before continuing."
            if action_applied else
            "No gameplay action was applied."
        )
        if http_rejection:
            guidance += (
                " Resolve the provider rejection before continuing." if action_applied
                else " Resolve the provider rejection before retrying."
            )
        elif not action_applied:
            guidance += " Press Step to retry."
        error_payload = {
            "error": (
                "OpenRouter rejected the request" if http_rejection
                else "OpenRouter connection failed"
            ),
            "details": f"{exc} {guidance}",
            "player": failed_actor.value if failed_actor is not None else None,
            "context_id": exc.context_id,
            "transport_attempt_count": exc.attempts,
            "trace_game_id": state.live_trace_game_id,
            "trace_failure_id": None,
            "action_applied": action_applied,
            "retryable": not action_applied and not http_rejection,
        }
        if http_rejection:
            error_payload["provider_status_code"] = exc.status_code
            if exc.provider_request_id:
                error_payload["provider_request_id"] = exc.provider_request_id
        rejected_attempts = sandbox.decision_trace[rejected_attempt_cursor:]
        if rejected_attempts:
            error_payload["attempts"] = [
                _failed_attempt_payload(attempt) for attempt in rejected_attempts
            ]
        current_app.logger.warning(
            "OpenRouter %s: player=%s attempts=%d context_id=%r",
            "request rejected" if http_rejection else "TLS recovery exhausted",
            failed_actor, exc.attempts, exc.context_id,
        )
        _record_live_failure(
            state, sandbox, failed_actor or actor, error_payload, rejected_attempts,
            sandbox.communication_trace[communication_cursor:],
            validation_error=f"{exc} {guidance}" if action_applied else str(exc),
        )
        broadcast_game_state(current_app.config["SOCKETIO"], state)
        return jsonify(error_payload), 502
    finally:
        state.step_processing = False
        _sync_live_inference(state, sandbox)

    analyze_transitions(
        state,
        result.transitions,
        pre_game_state,
        sandbox.game_engine.state,
        messages=result.messages,
    )
    transition = result.transitions[-1]

    winner = result.winner
    if winner is not None:
        state.game_running = False

    communication_attempts = sandbox.communication_trace[communication_cursor:]
    if communication_error_type is not None:
        _, communication_attempts = _safe_failure_traces(
            (), communication_attempts, communication_error_type,
        )
    reasoning_traces = build_live_reasoning_traces(
        sandbox, result, communication_attempts=communication_attempts,
    )
    state_snapshot = build_game_state_snapshot(state)
    trace_step_index = None
    if trace_store is not None and state.live_trace_game_id is not None:
        try:
            trace_step_index = trace_store.record_step(
                state.live_trace_game_id,
                result=result,
                rejected_attempts=sandbox.decision_trace[rejected_attempt_cursor:],
                communication_attempts=communication_attempts,
                public_state=state_snapshot,
                snapshot=sandbox.snapshot(),
            )
        except Exception as exc:
            current_app.logger.error("Could not persist applied live step (%s)", type(exc).__name__)
            error_payload = {
                "error": "Applied game step could not be saved",
                "details": "A gameplay action was applied. Do not repeat the applied action.",
                "player": actor.value,
                "trace_game_id": state.live_trace_game_id,
                "action_applied": True,
                "retryable": False,
                "warning": warning,
            }
            _record_live_failure(
                state, sandbox, actor, error_payload, (), (), persistence_failed=True,
            )
            state_snapshot = broadcast_game_state(current_app.config["SOCKETIO"], state)
            return jsonify({**error_payload, "state": state_snapshot}), 500
    state_snapshot = _stamp_recorded_step_messages(
        state, trace_store, trace_step_index, game_log_mark, state_snapshot,
    )
    if warning is not None:
        state_snapshot = broadcast_game_state(current_app.config["SOCKETIO"], state)
    return jsonify(
        {
            "status": "ok",
            "warning": warning,
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
            "automatic_action": result.automatic_action.to_payload() if result.automatic_action else None,
            "trace_game_id": state.live_trace_game_id,
            "trace_step_index": trace_step_index,
            "state": state_snapshot,
        }
    )


@live_game_bp.route("/api/step", methods=["POST"])
def step_game():
    """Execute one live-game step and always return a JSON response."""
    state = _get_state()
    with state.replay_mutation_lock:
        sandbox = getattr(state, "current_sandbox", None)
        actor = sandbox.current_actor() if sandbox is not None and not state.replay_mode else None
        action_count = len(sandbox.game_engine.state.actions) if actor is not None else 0
        decision_cursor = len(sandbox.decision_trace) if actor is not None else 0
        communication_cursor = len(sandbox.communication_trace) if actor is not None else 0
        try:
            return _step_game_transaction(state)
        except Exception as exc:
            state.step_processing = False
            current_app.logger.error("Unhandled live sandbox step failure (%s)", type(exc).__name__)
            action_applied = actor is not None and len(sandbox.game_engine.state.actions) != action_count
            error_payload = {
                "error": "Sandbox step failed",
                "details": f"{type(exc).__name__}. " + (
                    "A gameplay action was applied before this error. Do not repeat it; "
                    "inspect the current game before continuing."
                    if action_applied else
                    "No gameplay action was applied. Inspect the failure before retrying."
                ),
                "player": actor.value if actor is not None else None,
                "trace_game_id": state.live_trace_game_id,
                "action_applied": action_applied,
                "retryable": False,
            }
            state.last_live_step_error = error_payload
            if actor is not None:
                attempts, communications = _safe_failure_traces(
                    sandbox.decision_trace[decision_cursor:],
                    sandbox.communication_trace[communication_cursor:],
                    type(exc).__name__,
                )
                if attempts:
                    error_payload["attempts"] = [_failed_attempt_payload(attempt) for attempt in attempts]
                _record_live_failure(
                    state, sandbox, actor, error_payload, attempts, communications,
                )
                try:
                    broadcast_game_state(current_app.config["SOCKETIO"], state)
                except Exception as broadcast_error:
                    current_app.logger.error(
                        "Could not broadcast live failure (%s)", type(broadcast_error).__name__,
                    )
            return jsonify(error_payload), 500


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
    trace = (
        trace_store.get_usage(game_id)
        if request.args.get("view") == "usage"
        else trace_store.get_game(game_id)
    )
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
    load_started = time.perf_counter()
    try:
        resume_point = trace_store.load_resume_point(game_id)
    except KeyError:
        return jsonify({"error": "Live trace not found"}), 404
    resume_ms = (time.perf_counter() - load_started) * 1000.0

    try:
        config = _config_from_stored_payload(resume_point.config)
        # A deliberate current path selection survives loading another game;
        # the loaded game's own paths and embedded sources never take over.
        selection = getattr(state, "active_live_config", None) or LiveSandboxConfig()
        config = replace(
            config, model=None if config.mode == "random" else resolve_live_model(selection.model),
            temperature=selection.temperature, reasoning=selection.reasoning,
            max_tokens=selection.max_tokens, max_decision_attempts=selection.max_decision_attempts,
            board_surface=selection.board_surface,
            shared_suite_path=selection.shared_suite_path,
            context_suite_path=selection.context_suite_path,
            communication_suite_path=selection.communication_suite_path,
        )
        rebuild_started = time.perf_counter()
        sandbox = create_live_sandbox(config, snapshot=resume_point.snapshot)
        rebuild_ms = (time.perf_counter() - rebuild_started) * 1000.0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
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
    state.active_live_config = config
    state.live_inference = _live_inference_payload(_applied_prompt_config(sandbox) or config)
    winner = sandbox.game_engine.winning_color()
    state.game_running = winner is None
    normalize_started = time.perf_counter()
    prior_public_state = normalize_public_state_game_log(
        resume_point.public_state or {}
    )
    stored_game_log = prior_public_state.get("game_log", [])
    state.game_log = stored_game_log if isinstance(stored_game_log, list) else []
    normalize_ms = (time.perf_counter() - normalize_started) * 1000.0
    state.live_trace_game_id = game_id
    trace_store.mark_game_status(
        game_id,
        status="running" if winner is None else "completed",
        winner=winner.value if winner is not None else None,
    )
    bump_replay_revision(state)
    snapshot_started = time.perf_counter()
    state_snapshot = broadcast_game_state(
        current_app.config["SOCKETIO"],
        state,
    )
    snapshot_ms = (time.perf_counter() - snapshot_started) * 1000.0
    total_ms = (time.perf_counter() - load_started) * 1000.0
    try:
        db_bytes = trace_store.path.stat().st_size
    except OSError:
        db_bytes = -1
    engine = getattr(sandbox, "game_engine", None)
    engine_state = getattr(engine, "state", None)
    # warning level: the dev server runs at WARNING by default, so info() is invisible.
    current_app.logger.warning(
        "saved-live-load game_id=%s loaded_step_index=%s db_bytes=%s "
        "resume_ms=%.1f rebuild_ms=%.1f normalize_ms=%.1f snapshot_ms=%.1f "
        "total_ms=%.1f actions=%s events=%s game_log=%s",
        game_id,
        resume_point.step_index,
        db_bytes,
        resume_ms,
        rebuild_ms,
        normalize_ms,
        snapshot_ms,
        total_ms,
        len(getattr(engine_state, "actions", []) or []),
        len(getattr(engine, "events", []) or []),
        len(state.game_log or []),
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
