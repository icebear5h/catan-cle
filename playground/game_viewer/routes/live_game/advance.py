"""Advancing the live sandbox by one full step, inside the mutation lock."""

from typing import cast

from flask import Response, current_app, jsonify

from cle.game_engine.public_board import JsonValue
from cle.harness.providers.openrouter import OpenRouterHTTPFailure, OpenRouterTLSFailure
from cle.sandbox.catan import (
    CatanSandbox,
    PlayerResponseError,
    PostActionCommunicationCancelled,
    PostActionCommunicationError,
)
from cle.sandbox.contracts import SandboxStepResult
from cle.sandbox.factory import (
    ActivePromptConfigurationError,
)
from cle.traces import SQLiteLiveTraceStore

from ...async_runtime import sandbox_async_runtime
from ...live.game_logging import (
    analyze_transitions,
    format_action_for_display,
)
from ...live.reasoning_trace import build_live_reasoning_traces
from ...state import ServerState
from ..websocket import broadcast_game_state, build_game_state_snapshot
from .blueprint import (
    _sync_live_inference,
)
from .failures import (
    _failed_attempt_payload,
    _last_model_output_excerpt,
    _record_live_failure,
    _safe_failure_traces,
)
from .traces import _stamp_recorded_step_messages


async def _step_with_committed_result(sandbox: CatanSandbox) -> SandboxStepResult:
    try:
        return await sandbox.step()
    except PostActionCommunicationCancelled as exc:
        # The thread bridge otherwise turns cancellation into an untyped Future
        # error, dropping the accepted action's request before it can be saved.
        raise PostActionCommunicationError(exc.result) from exc


def _step_game_transaction(state: ServerState) -> Response | tuple[Response, int]:
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
    warning: dict[str, JsonValue] | None = None
    communication_error_type = None

    state.step_processing = True
    state.last_live_step_error = None
    try:
        result = sandbox_async_runtime.run(_step_with_committed_result(sandbox))
    except ActivePromptConfigurationError as exc:
        error_payload: dict[str, JsonValue] = {
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
            warning["details"] = f'{warning["details"]} {provider_failure}'
        if isinstance(provider_failure, OpenRouterHTTPFailure):
            warning["details"] = (
                f'{warning["details"]} {provider_failure} '
                "Resolve the provider rejection before the next Step."
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
        if isinstance(exc, OpenRouterHTTPFailure) and http_rejection:
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
    store = cast(SQLiteLiveTraceStore, trace_store)
    state_snapshot = _stamp_recorded_step_messages(
        state, store, trace_step_index, game_log_mark, state_snapshot,
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
