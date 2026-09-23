"""The step endpoint: one guarded advance of the running live game."""

from typing import cast

from flask import Response, current_app, jsonify

from cle.game_engine.public_board import JsonValue
from cle.sandbox.catan import CatanSandbox

from ..websocket import broadcast_game_state
from .advance import _step_game_transaction
from .blueprint import (
    _get_state,
    live_game_bp,
)
from .failures import (
    _failed_attempt_payload,
    _record_live_failure,
    _safe_failure_traces,
)


@live_game_bp.route("/api/step", methods=["POST"])
def step_game() -> Response | tuple[Response, int]:
    """Execute one live-game step and always return a JSON response."""
    state = _get_state()
    with state.replay_mutation_lock:
        sandbox = cast(CatanSandbox, state.current_sandbox)
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
            error_payload: dict[str, JsonValue] = {
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
