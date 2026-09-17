"""Local component-oriented Prompt Studio API."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import replace
from typing import Any, Mapping

import yaml
from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from cle.harness.board_surface import board_presentation_payload
from cle.harness.communication import (
    CommunicationSuite,
    build_communication_request,
    parse_communication_suite,
)
from cle.harness.context import ContextAssembler
from cle.harness.models import PlayerSession, PromptComponent
from cle.harness.prompt_store import (
    ActivePromptSuites,
    PromptSuiteConflictError,
    resolve_prompt_suites,
    reset_prompt_suite_overrides,
    reset_shared_prompt_override,
    save_prompt_suite_overrides,
    save_shared_prompt_override,
    validate_prompt_suite_sources,
    validate_shared_prompt_source,
)
from cle.harness.shared_suite import SharedPromptSuite, parse_shared_prompt_suite
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
        getattr(state, "replay_mode", False)
        or getattr(state, "replay_data", None) is not None
    )


def _active_suites() -> ActivePromptSuites:
    selection = getattr(_state(), "active_live_config", None)
    return resolve_prompt_suites(
        shared_path=getattr(selection, "shared_suite_path", None),
        decision_path=getattr(selection, "context_suite_path", None),
        communication_path=getattr(selection, "communication_suite_path", None),
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
        "status": document.status,
        "sha256": document.sha256,
        "overridden": document.overridden,
    }


def _editor_payload(active: ActivePromptSuites) -> dict[str, Any]:
    if active.shared is not None:
        bundle = parse_shared_prompt_suite(active.shared.source)
        return {
            "mode": "shared",
            "shared": {**_metadata(active.shared), "document": bundle.model_dump(mode="json")},
        }
    decision = parse_context_suite(active.decision.source)
    communication = parse_communication_suite(active.communication.source)
    return {
        "mode": "legacy",
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
    resolved = sandbox.decision_context()
    context = resolved[0] if isinstance(resolved, tuple) else resolved
    actor = context.actor
    player = getattr(sandbox, "players", {}).get(actor)
    session = deepcopy(getattr(player, "session", None))
    if session is None:
        session = PlayerSession(
            color=actor,
            session_id=f"prompt-preview:{sandbox.game_engine.id}:{actor.value}",
        )
    if suite.context.memory_mode == "fresh_notes":
        cutoff = context.visible_through_sequence
        if cutoff is not None:
            cursor = session.action_next_sequence if session.context_policy == "fresh_notes" else 0
            messages = tuple(
                event for event in context.visible_messages if cursor <= event.sequence <= cutoff
            )
            context = replace(
                context,
                events=tuple(event for event in context.events if cursor <= event.sequence <= cutoff),
                recent_messages=messages, visible_messages=messages,
            )
    presenter = getattr(getattr(player, "_assembler", None), "board_presenter", None)
    model_request = ContextAssembler(suite, board_presenter=presenter).assemble(context, session)
    return {
        "status": "rendered",
        "provenance": "current_typed_context",
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
    for record in reversed(getattr(sandbox, "communication_trace", ())):
        choice = record.choice
        if choice.model_request is not None:
            return choice.model_request
    return None


def _communication_preview(
    state: Any,
    suite: CommunicationSuite,
    *,
    candidate: bool = False,
) -> dict[str, Any]:
    if suite.components or candidate:
        sandbox = getattr(state, "current_sandbox", None)
        policy = getattr(sandbox, "communication_policy", None)
        opportunity = None
        provenance = "current_typed_context"
        if policy is not None:
            opportunities = policy.pre_action(sandbox.game_engine)
            if opportunities:
                opportunity = opportunities[0]
            elif trace := getattr(sandbox, "communication_trace", ()):
                opportunity = replace(
                    trace[-1].opportunity,
                    visible_through_sequence=sandbox.game_engine.revision - 1,
                )
                provenance = "current_typed_context_with_latest_trigger"
        if opportunity is not None:
            context = sandbox._talk_context(opportunity)
            context = replace(context, observation=sandbox.game_engine.observe(context.player))
            player = sandbox.players.get(context.player)
            session = getattr(player, "session", None)
            notes = ""
            if suite.memory_mode == "fresh_notes":
                cursor = (
                    session.talk_next_sequence
                    if getattr(session, "context_policy", "legacy") == "fresh_notes" else 0
                )
                if session is not None:
                    notes = session.strategic_memory
                visible = sandbox.game_engine.project_events(context.player)
                messages = tuple(
                    event for event in visible
                    if cursor <= event.sequence <= context.visible_through_sequence
                    and event.event_type == "MESSAGE_SENT"
                )
                context = replace(
                    context,
                    game_events=tuple(event for event in context.game_events if event.sequence >= cursor),
                    recent_messages=messages, visible_messages=messages,
                )
            model_request = build_communication_request(
                context, "prompt-preview", suite, notes=notes,
                board_presenter=getattr(getattr(player, "_assembler", None), "board_presenter", None),
            )
            return {
                "status": "rendered", "actor": context.player.value, "provenance": provenance,
                "components": [_component_payload(item) for item in model_request.components],
                "board_presentation": board_presentation_payload(
                    model_request.board_presentation, include_text_content=True,
                ),
            }
        if suite.components:
            return {
                "status": "no_current_communication_context",
                "provenance": "authored_templates_only",
                "components": [
                    {
                        "id": f"{suite.components[name].channel}.{name}",
                        "channel": suite.components[name].channel,
                        "template": suite.components[name].template,
                        "value": "", "rendered": "", "variables": {},
                    }
                    for name in suite.order
                ],
            }
    model_request = None if candidate else _latest_communication_request(state)
    if model_request is not None and model_request.components:
        return {
            "status": "rendered",
            "provenance": "recorded_request",
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
    *,
    candidate: bool = False,
) -> dict[str, Any]:
    if active.shared is not None:
        bundle = parse_shared_prompt_suite(active.shared.source)
        decision = bundle.decision_suite()
        communication = bundle.communication_suite()
    else:
        decision = parse_context_suite(active.decision.source)
        communication = parse_communication_suite(active.communication.source)
    with state.replay_mutation_lock:
        return {
            **_editor_payload(active),
            "saving_locked": _saving_locked(state),
            "preview": {
                "decision": _decision_preview(state, decision),
                "communication": _communication_preview(state, communication, candidate=candidate),
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
    if active.shared is not None:
        raise PromptSuiteEditError("request", "Shared mode requires one complete shared document")
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


def _shared_source(payload: Any) -> str:
    document = _mapping(payload, "shared")
    _exact_keys(document, set(SharedPromptSuite.model_fields), "shared")
    # JSON strict mode accepts declared arrays but never coerces booleans/numbers/strings.
    bundle = SharedPromptSuite.model_validate_json(json.dumps(document), strict=True)
    return yaml.safe_dump(bundle.model_dump(mode="json"), sort_keys=False, width=1_000)


def _shared_expected(payload: Any) -> str:
    expected = _mapping(payload, "expected")
    _exact_keys(expected, {"shared"}, "expected")
    return _string(expected["shared"], "expected.shared")


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
        return _response(_studio_payload(_active_suites(), _state()))
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite cannot be loaded", "details": str(exc)},
            500,
        )


@prompt_suite_bp.route("/api/prompt-suite/validate", methods=["POST"])
def validate_prompt_suite():
    try:
        root = _mapping(request.get_json(silent=True), "request")
        if "shared" in root:
            _exact_keys(root, {"shared"}, "request")
            candidate = validate_shared_prompt_source(_shared_source(root["shared"]))
        else:
            active = _active_suites()
            decision_source, communication_source = _edited_sources(active, root)
            candidate = validate_prompt_suite_sources(decision_source, communication_source)
        return _response(
            {
                "valid": True,
                "candidate": _studio_payload(candidate, _state(), candidate=True),
                "errors": [],
            }
        )
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"valid": False, "errors": [_validation_error(exc)]},
            400,
        )


@prompt_suite_bp.route("/api/prompt-suite", methods=["PUT"])
def save_prompt_suite():
    state = _state()
    # The store has its own atomic lock. Publishing a new source must not wait
    # for provider I/O; players acquire it only at safe inference boundaries.
    return _save_prompt_suite_transaction(state)


def _save_prompt_suite_transaction(state: Any):
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded replay before saving prompt suites"},
            409,
        )
    payload = request.get_json(silent=True)
    try:
        _require_local_selection()
        root = _mapping(payload, "request")
        if "shared" in root:
            _exact_keys(root, {"expected", "shared"}, "request")
            saved = save_shared_prompt_override(
                source=_shared_source(root["shared"]),
                expected_sha256=_shared_expected(root["expected"]),
            )
            return _response({"status": "saved", **_studio_payload(saved, state)})
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
        active = _active_suites()
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
    return _reset_prompt_suite_transaction(state)


def _reset_prompt_suite_transaction(state: Any):
    if _saving_locked(state):
        return _response(
            {"error": "Clear the loaded replay before resetting prompt suites"},
            409,
        )
    try:
        _require_local_selection()
        root = _mapping(request.get_json(silent=True), "request")
        _exact_keys(root, {"expected"}, "request")
        if "shared" in _mapping(root["expected"], "expected"):
            active = reset_shared_prompt_override(expected_sha256=_shared_expected(root["expected"]))
            return _response({"status": "reset", **_studio_payload(active, state)})
        expected_decision, expected_communication = _expected_hashes(
            root
        )
        active = reset_prompt_suite_overrides(
            expected_decision_sha256=expected_decision,
            expected_communication_sha256=expected_communication,
            shared_default=True,
        )
        return _response({"status": "reset", **_studio_payload(active, state)})
    except PromptSuiteConflictError as exc:
        return _response({"error": str(exc)}, 409)
    except (OSError, TypeError, ValueError, ValidationError, yaml.YAMLError) as exc:
        return _response(
            {"error": "Prompt suite reset failed", "errors": [_validation_error(exc)]},
            400,
        )


def _require_local_selection() -> None:
    selection = getattr(_state(), "active_live_config", None)
    if any(getattr(selection, name, None) for name in (
        "shared_suite_path", "context_suite_path", "communication_suite_path",
    )):
        raise PromptSuiteConflictError(
            "Explicit runtime prompt paths are active; edit those files or select local overrides"
        )
    if any(os.getenv(name) for name in (
        "CATAN_SHARED_SUITE", "CATAN_CONTEXT_SUITE", "CATAN_COMMUNICATION_SUITE",
    )):
        raise PromptSuiteConflictError(
            "Explicit environment prompt paths are active; clear them before editing local overrides"
        )
