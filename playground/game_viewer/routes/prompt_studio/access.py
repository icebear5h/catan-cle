"""Reading server state, the edit gates, and the no-store JSON response."""

import os
from collections.abc import Mapping

from flask import Response, current_app, jsonify

from cle.harness.prompt_store import PROMPT_SUITE_PIN_ENVS, PromptSuiteConflictError

from ...state import ServerState

__all__ = [
    "json_response",
    "require_local_selection",
    "saving_locked",
    "server_state",
]


def server_state() -> ServerState:
    state: ServerState = current_app.config["SERVER_STATE"]
    return state


def saving_locked(state: ServerState) -> bool:
    return bool(
        getattr(state, "replay_mode", False)
        or getattr(state, "replay_data", None) is not None
    )


def json_response(payload: Mapping[str, object], status: int = 200) -> Response:
    response = jsonify(payload)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


def require_local_selection(state: ServerState) -> None:
    selection = getattr(state, "active_live_config", None)
    if any(getattr(selection, name, None) for name in (
        "shared_suite_path", "context_suite_path", "communication_suite_path",
    )):
        raise PromptSuiteConflictError(
            "Explicit runtime prompt paths are active; edit those files or select local overrides"
        )
    if any(os.getenv(name) for name in PROMPT_SUITE_PIN_ENVS):
        raise PromptSuiteConflictError(
            "Explicit environment prompt paths are active; clear them before editing local overrides"
        )
