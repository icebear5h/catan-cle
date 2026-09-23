"""Local component-oriented Prompt Studio API.

The names the Studio tests patch (``_saving_locked``, ``_decision_preview``,
``_communication_preview``, ``save_shared_prompt_override``) are module globals
here, and every caller that must honour a patch lives in this module too.
"""

import yaml
from flask import Blueprint, Response, request
from pydantic import ValidationError

from cle.harness.prompt_store import (
    ActivePromptSuites,
    PromptSuiteConflictError,
    compile_active_suites,
    follow_latest_enabled,
    reset_shared_prompt_override,
    resolve_prompt_suites,
    save_shared_prompt_override,
    validate_shared_prompt_source,
)

from ..state import ServerState
from .prompt_studio import (
    PromptSuiteEditError,
    _communication_preview,
    _component_payload,
    _decision_preview,
    _editor_payload,
    _exact_keys,
    _latest_communication_request,
    _mapping,
    _metadata,
    _require_local_selection,
    _response,
    _saving_locked,
    _shared_expected,
    _shared_source,
    _state,
    _validation_error,
)

__all__ = [
    "PromptSuiteEditError",
    "_active_suites",
    "_communication_preview",
    "_component_payload",
    "_decision_preview",
    "_latest_communication_request",
    "_metadata",
    "_saving_locked",
    "get_prompt_suite",
    "prompt_suite_bp",
    "reset_prompt_suite",
    "save_prompt_suite",
    "validate_prompt_suite",
]

prompt_suite_bp = Blueprint("prompt_suite", __name__)


def _active_suites() -> ActivePromptSuites:
    selection = getattr(_state(), "active_live_config", None)
    if follow_latest_enabled():
        return resolve_prompt_suites(follow_latest=True)
    return resolve_prompt_suites(
        shared_path=getattr(selection, "shared_suite_path", None),
        decision_path=getattr(selection, "context_suite_path", None),
        communication_path=getattr(selection, "communication_suite_path", None),
    )


def _studio_payload(active: ActivePromptSuites, state: ServerState) -> dict[str, object]:
    decision, communication = compile_active_suites(active)
    with state.replay_mutation_lock:
        return {
            **_editor_payload(active),
            "saving_locked": _saving_locked(state),
            "preview": {
                "decision": _decision_preview(state, decision),
                "communication": _communication_preview(state, communication),
            },
        }


@prompt_suite_bp.route("/api/prompt-suite", methods=["GET"])
def get_prompt_suite() -> Response:
    try:
        return _response(_studio_payload(_active_suites(), _state()))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite cannot be loaded", "details": str(exc)},
            500,
        )


@prompt_suite_bp.route("/api/prompt-suite/validate", methods=["POST"])
def validate_prompt_suite() -> Response:
    try:
        root = _mapping(request.get_json(silent=True), "request")
        _exact_keys(root, {"shared"}, "request")
        candidate = validate_shared_prompt_source(_shared_source(root["shared"]))
        return _response(
            {
                "valid": True,
                "candidate": _studio_payload(candidate, _state()),
                "errors": [],
            }
        )
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"valid": False, "errors": [_validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["PUT"])
def save_prompt_suite() -> Response:
    state = _state()
    # The store has its own atomic lock. Publishing a new source must not wait
    # for provider I/O; players acquire it only at safe inference boundaries.
    return _save_prompt_suite_transaction(state)


def _save_prompt_suite_transaction(state: ServerState) -> Response:
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded replay before saving prompt suites"},
            409,
        )
    payload = request.get_json(silent=True)
    try:
        _require_local_selection()
        root = _mapping(payload, "request")
        _exact_keys(root, {"expected", "shared"}, "request")
        saved = save_shared_prompt_override(
            source=_shared_source(root["shared"]),
            expected_sha256=_shared_expected(root["expected"]),
        )
        return _response({"status": "saved", **_studio_payload(saved, state)})
    except PromptSuiteConflictError as exc:
        return _response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite validation failed", "errors": [_validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["DELETE"])
def reset_prompt_suite() -> Response:
    state = _state()
    return _reset_prompt_suite_transaction(state)


def _reset_prompt_suite_transaction(state: ServerState) -> Response:
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded replay before resetting prompt suites"},
            409,
        )
    try:
        _require_local_selection()
        root = _mapping(request.get_json(silent=True), "request")
        _exact_keys(root, {"expected"}, "request")
        active = reset_shared_prompt_override(expected_sha256=_shared_expected(root["expected"]))
        return _response({"status": "reset", **_studio_payload(active, state)})
    except PromptSuiteConflictError as exc:
        return _response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite reset failed", "errors": [_validation_error(exc)]},
            400,
        )
