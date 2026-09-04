"""Local component-oriented Prompt Studio API."""

from __future__ import annotations

from typing import Any, Mapping

import yaml
from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from cle.harness.board_surface import board_presentation_payload
from cle.harness.communication import (
    CommunicationSuite,
    parse_communication_suite,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import PlayerSession, PromptComponent
from cle.harness.prompt_store import (
    ActivePromptSuites,
    PromptSuiteConflictError,
    load_active_prompt_suites,
    reset_prompt_suite_overrides,
    save_prompt_suite_overrides,
    validate_prompt_suite_sources,
)
from cle.harness.suite import ContextSuite, parse_context_suite


prompt_suite_bp = Blueprint("prompt_suite", __name__)


class PromptSuiteEditError(ValueError):
    def __init__(self, component: str, message: str) -> None:
        super().__init__(message)
        self.component = component


def _state():
    return current_app.config["SERVER_STATE"]


def _saving_locked(state: Any) -> bool:
    return bool(
        getattr(state, "current_sandbox", None) is not None
        or getattr(state, "replay_mode", False)
        or getattr(state, "replay_data", None) is not None
    )


def _response(payload: Mapping[str, Any], status: int = 200):
    response = jsonify(payload)
    response.status_code = status
    response.headers["Cache-Control"] = "no-store"
    return response


def _component_payload(component: PromptComponent) -> dict[str, Any]:
    return {
        "id": component.id,
        "channel": component.channel,
        "template": component.template,
        "value": component.value,
        "rendered": component.rendered,
        "variables": dict(component.variables),
    }


def _metadata(document: Any) -> dict[str, Any]:
    return {
        "id": document.id,
        "version": document.version,
        "sha256": document.sha256,
        "overridden": document.overridden,
    }


def _editor_payload(active: ActivePromptSuites) -> dict[str, Any]:
    decision = parse_context_suite(active.decision.source)
    communication = parse_communication_suite(active.communication.source)
    return {
        "decision": {
            **_metadata(active.decision),
            "system_identity": decision.system.template,
            "component_order": [
                f"environment.{name}"
                for name in decision.context.order
                if name != "trajectory"
            ],
            "components": {
                name: decision.sections[name].template
                for name in decision.context.order
                if name != "trajectory"
            },
            "phase_guidance": dict(decision.phase_guidance),
            "response_instruction": decision.response.instruction,
        },
        "communication": {
            **_metadata(active.communication),
            "system_identity": communication.system_template,
            "component_order": [
                f"environment.{name}" for name in communication.order
            ],
            "components": {
                name: communication.sections[name].template
                for name in communication.order
            },
        },
        "variables": {
            "system.identity": ["color"],
            "environment.*": ["value"],
        },
    }


def _decision_preview(
    state: Any,
    suite: ContextSuite,
) -> dict[str, Any]:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return {"status": "no_game_context", "components": []}
    actor = sandbox.current_actor()
    player = sandbox.players.get(actor)
    session = getattr(player, "session", None)
    if session is None:
        session = PlayerSession(
            color=actor,
            session_id=f"prompt-preview:{sandbox.game_engine.id}:{actor.value}",
        )
    context = sandbox.decision_context(actor)
    model_request = ContextAssembler(suite).assemble(context, session)
    return {
        "status": "rendered",
        "actor": actor.value,
        "prompt_key": context.prompt_key,
        "components": [
            _component_payload(component)
            for component in model_request.components
        ],
        "board_presentation": board_presentation_payload(
            model_request.board_presentation,
            include_text_content=True,
        ),
    }


def _latest_communication_request(state: Any) -> Any:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return None
    for _, choice in reversed(sandbox.communication_trace):
        if choice.model_request is not None:
            return choice.model_request
    return None


def _communication_preview(
    state: Any,
    suite: CommunicationSuite,
) -> dict[str, Any]:
    model_request = _latest_communication_request(state)
    if model_request is not None and model_request.components:
        return {
            "status": "rendered",
            "components": [
                _component_payload(component)
                for component in model_request.components
            ],
        }
    if model_request is not None:
        user_messages = [
            message.content
            for message in model_request.messages
            if message.role == "user"
        ]
        return {
            "status": "legacy_combined_request",
            "components": [
                {
                    "id": "environment.legacy_combined_request",
                    "channel": "environment",
                    "template": "",
                    "value": user_messages[-1] if user_messages else "",
                    "rendered": user_messages[-1] if user_messages else "",
                    "variables": {},
                }
            ],
        }

    components = [
        {
            "id": "system.identity",
            "channel": "system",
            "template": suite.system_template,
            "value": "",
            "rendered": "",
            "variables": {"color": ""},
        }
    ]
    components.extend(
        {
            "id": f"environment.{name}",
            "channel": "environment",
            "template": suite.sections[name].template,
            "value": "",
            "rendered": "",
            "variables": (
                {"value": ""}
                if "{{ value }}" in suite.sections[name].template
                else {}
            ),
        }
        for name in suite.order
    )
    return {
        "status": "no_communication_opportunity_rendered_yet",
        "components": components,
    }


def _studio_payload(
    active: ActivePromptSuites,
    state: Any,
) -> dict[str, Any]:
    decision = parse_context_suite(active.decision.source)
    communication = parse_communication_suite(active.communication.source)
    return {
        **_editor_payload(active),
        "saving_locked": _saving_locked(state),
        "preview": {
            "decision": _decision_preview(state, decision),
            "communication": _communication_preview(state, communication),
        },
    }


def _mapping(value: Any, component: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PromptSuiteEditError(component, "must be an object")
    return value


def _string(value: Any, component: str) -> str:
    if not isinstance(value, str):
        raise PromptSuiteEditError(component, "must be a string")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    component: str,
) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise PromptSuiteEditError(
            component,
            f"must contain fixed keys; missing={missing}, extra={extra}",
        )


def _edited_sources(
    active: ActivePromptSuites,
    payload: Any,
) -> tuple[str, str]:
    root = _mapping(payload, "request")
    _exact_keys(root, {"decision", "communication"}, "request")
    decision_edit = _mapping(root["decision"], "decision")
    communication_edit = _mapping(root["communication"], "communication")
    _exact_keys(
        decision_edit,
        {
            "system_identity",
            "components",
            "phase_guidance",
            "response_instruction",
        },
        "decision",
    )
    _exact_keys(
        communication_edit,
        {"system_identity", "components"},
        "communication",
    )

    decision_data = yaml.safe_load(active.decision.source)
    communication_data = yaml.safe_load(active.communication.source)
    decision_components = _mapping(
        decision_edit["components"],
        "decision.components",
    )
    decision_guidance = _mapping(
        decision_edit["phase_guidance"],
        "decision.phase_guidance",
    )
    communication_components = _mapping(
        communication_edit["components"],
        "communication.components",
    )
    _exact_keys(
        decision_components,
        set(decision_data["sections"]),
        "decision.components",
    )
    _exact_keys(
        decision_guidance,
        set(decision_data["phase_guidance"]),
        "decision.phase_guidance",
    )
    _exact_keys(
        communication_components,
        set(communication_data["sections"]),
        "communication.components",
    )

    decision_data["system"]["template"] = _string(
        decision_edit["system_identity"],
        "system.identity",
    )
    for name in decision_data["sections"]:
        decision_data["sections"][name]["template"] = _string(
            decision_components[name],
            f"environment.{name}",
        )
    for name in decision_data["phase_guidance"]:
        decision_data["phase_guidance"][name] = _string(
            decision_guidance[name],
            f"phase_guidance.{name}",
        )
    decision_data["response"]["instruction"] = _string(
        decision_edit["response_instruction"],
        "environment.response_schema.value",
    )

    communication_data["system_template"] = _string(
        communication_edit["system_identity"],
        "communication.system.identity",
    )
    for name in communication_data["sections"]:
        communication_data["sections"][name]["template"] = _string(
            communication_components[name],
            f"communication.environment.{name}",
        )

    decision_source = yaml.safe_dump(
        decision_data,
        sort_keys=False,
        allow_unicode=True,
        width=1_000,
    )
    communication_source = yaml.safe_dump(
        communication_data,
        sort_keys=False,
        allow_unicode=True,
        width=1_000,
    )
    validate_prompt_suite_sources(decision_source, communication_source)
    return decision_source, communication_source


def _expected_hashes(payload: Any) -> tuple[str, str]:
    root = _mapping(payload, "request")
    _exact_keys(root, {"expected"}, "request")
    expected = _mapping(root["expected"], "expected")
    _exact_keys(expected, {"decision", "communication"}, "expected")
    return (
        _string(expected["decision"], "expected.decision"),
        _string(expected["communication"], "expected.communication"),
    )


def _validation_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, PromptSuiteEditError):
        return {"component": exc.component, "message": str(exc)}
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        location = ".".join(str(item) for item in first.get("loc", ()))
        return {
            "component": location or "suite",
            "message": first.get("msg", str(exc)),
        }
    return {"component": "suite", "message": str(exc)}


@prompt_suite_bp.route("/api/prompt-suite", methods=["GET"])
def get_prompt_suite():
    try:
        return _response(_studio_payload(load_active_prompt_suites(), _state()))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite cannot be loaded", "details": str(exc)},
            500,
        )


@prompt_suite_bp.route("/api/prompt-suite/validate", methods=["POST"])
def validate_prompt_suite():
    try:
        active = load_active_prompt_suites()
        decision_source, communication_source = _edited_sources(
            active,
            request.get_json(silent=True),
        )
        candidate = validate_prompt_suite_sources(
            decision_source,
            communication_source,
        )
        return _response(
            {
                "valid": True,
                "candidate": _studio_payload(candidate, _state()),
                "errors": [],
            }
        )
    except (TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"valid": False, "errors": [_validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["PUT"])
def save_prompt_suite():
    state = _state()
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded game before saving prompt suites"},
            409,
        )
    payload = request.get_json(silent=True)
    try:
        root = _mapping(payload, "request")
        _exact_keys(root, {"expected", "decision", "communication"}, "request")
        expected = _mapping(root["expected"], "expected")
        _exact_keys(expected, {"decision", "communication"}, "expected")
        expected_decision = _string(
            expected["decision"],
            "expected.decision",
        )
        expected_communication = _string(
            expected["communication"],
            "expected.communication",
        )
        edit_payload = {
            "decision": root["decision"],
            "communication": root["communication"],
        }
        active = load_active_prompt_suites()
        decision_source, communication_source = _edited_sources(
            active,
            edit_payload,
        )
        saved = save_prompt_suite_overrides(
            decision_source=decision_source,
            communication_source=communication_source,
            expected_decision_sha256=expected_decision,
            expected_communication_sha256=expected_communication,
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
def reset_prompt_suite():
    state = _state()
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded game before resetting prompt suites"},
            409,
        )
    try:
        expected_decision, expected_communication = _expected_hashes(
            request.get_json(silent=True)
        )
        active = reset_prompt_suite_overrides(
            expected_decision_sha256=expected_decision,
            expected_communication_sha256=expected_communication,
        )
        return _response({"status": "reset", **_studio_payload(active, state)})
    except PromptSuiteConflictError as exc:
        return _response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite reset failed", "errors": [_validation_error(exc)]},
            400,
        )
