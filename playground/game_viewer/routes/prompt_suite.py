"""Local component-oriented Prompt Studio API.

Collaborators (the replay gate, previews, and store writes) come from
``PromptStudioDeps``; replace them per app via ``app.config[PROMPT_STUDIO_DEPS]``.
"""

import yaml
from flask import Blueprint, Response, request
from pydantic import ValidationError

from cle.harness.prompt_store import ActivePromptSuites, PromptSuiteConflictError

from .prompt_studio import (
    PromptStudioService,
    json_response,
    read_reset,
    read_save,
    read_validate,
    server_state,
    studio_deps,
    studio_payload,
    validation_error,
)

__all__ = [
    "get_prompt_suite",
    "prompt_suite_bp",
    "reset_prompt_suite",
    "save_prompt_suite",
    "validate_prompt_suite",
]

prompt_suite_bp = Blueprint("prompt_suite", __name__)


def _studio() -> PromptStudioService:
    return PromptStudioService(server_state(), studio_deps())


def _payload(studio: PromptStudioService, active: ActivePromptSuites) -> dict[str, object]:
    return studio_payload(active, studio.state, studio.deps)


@prompt_suite_bp.route("/api/prompt-suite", methods=["GET"])
def get_prompt_suite() -> Response:
    studio = _studio()
    try:
        return json_response(_payload(studio, studio.active()))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return json_response(
            {"error": "Prompt suite cannot be loaded", "details": str(exc)},
            500,
        )


@prompt_suite_bp.route("/api/prompt-suite/validate", methods=["POST"])
def validate_prompt_suite() -> Response:
    studio = _studio()
    try:
        candidate = studio.validate(read_validate(request.get_json(silent=True)))
        return json_response(
            {
                "valid": True,
                "candidate": _payload(studio, candidate),
                "errors": [],
            }
        )
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return json_response(
            {"valid": False, "errors": [validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["PUT"])
def save_prompt_suite() -> Response:
    studio = _studio()
    if studio.saving_locked():
        return json_response(
            {"error": "Clear the loaded replay before saving prompt suites"},
            409,
        )
    try:
        studio.ensure_local_selection()
        saved = studio.save(read_save(request.get_json(silent=True)))
        return json_response({"status": "saved", **_payload(studio, saved)})
    except PromptSuiteConflictError as exc:
        return json_response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return json_response(
            {"error": "Prompt suite validation failed", "errors": [validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["DELETE"])
def reset_prompt_suite() -> Response:
    studio = _studio()
    if studio.saving_locked():
        return json_response(
            {"error": "Clear the loaded replay before resetting prompt suites"},
            409,
        )
    try:
        studio.ensure_local_selection()
        active = studio.reset(read_reset(request.get_json(silent=True)))
        return json_response({"status": "reset", **_payload(studio, active)})
    except PromptSuiteConflictError as exc:
        return json_response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return json_response(
            {"error": "Prompt suite reset failed", "errors": [validation_error(exc)]},
            400,
        )
