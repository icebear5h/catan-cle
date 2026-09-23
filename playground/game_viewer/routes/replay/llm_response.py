"""A non-mutating model decision for the replay's current position."""

import traceback
from typing import cast

from flask import Response, current_app, jsonify, request
from httpx import HTTPError

from cle.harness.reasoning import validate_native_reasoning_request
from cle.harness.validation import (
    HarnessValidationError,
    validate_game_plan,
    validate_model_id,
)
from cle.sandbox.replay import ReplaySandbox

from ...async_runtime import sandbox_async_runtime
from ...replay.decision_preview import generate_decision_preview
from .blueprint import _get_state, replay_bp

__all__ = ["replay_llm_response"]


@replay_bp.route('/api/replay-llm-response', methods=['POST'])
def replay_llm_response() -> Response | tuple[Response, int]:
    """Generate a non-mutating LLM decision for the current replay position."""
    state = _get_state()
    if not state.replay_mode or not state.replay_data or not state.current_sandbox:
        return jsonify({"error": "No replay loaded"}), 400
    sandbox = cast(ReplaySandbox, state.current_sandbox)

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
