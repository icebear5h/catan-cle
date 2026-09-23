"""Rendering the current typed context through a candidate suite."""

from copy import deepcopy
from dataclasses import replace
from typing import cast

from cle.harness.board_surface import board_presentation_payload
from cle.harness.communication import CommunicationSuite, build_communication_request
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelRequest, PlayerSession
from cle.harness.suite import ContextSuite
from cle.sandbox.catan import CatanSandbox

from ...state import ServerState
from .editor import _component_payload

__all__ = [
    "_communication_preview",
    "_decision_preview",
    "_latest_communication_request",
]


def _decision_preview(
    state: ServerState,
    suite: ContextSuite,
) -> dict[str, object]:
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


def _latest_communication_request(state: ServerState) -> ModelRequest | None:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return None
    for record in reversed(getattr(sandbox, "communication_trace", ())):
        choice = record.choice
        if choice.model_request is not None:
            return cast(ModelRequest, choice.model_request)
    return None


def _communication_preview(
    state: ServerState,
    suite: CommunicationSuite,
) -> dict[str, object]:
    if suite.components:
        sandbox = cast(CatanSandbox, state.current_sandbox)
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
            session = cast(PlayerSession | None, getattr(player, "session", None))
            notes = ""
            if suite.memory_mode == "fresh_notes":
                cursor = (
                    cast(PlayerSession, session).talk_next_sequence
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
    # Non-component (pinned legacy) suites preview read-only from recorded requests.
    recorded_request = _latest_communication_request(state)
    if recorded_request is not None and recorded_request.components:
        return {
            "status": "rendered",
            "provenance": "recorded_request",
            "components": [
                _component_payload(component)
                for component in recorded_request.components
            ],
        }
    if recorded_request is not None:
        user_messages = [
            message.content
            for message in recorded_request.messages
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

    components: list[dict[str, object]] = [
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
