"""Rendering the current typed context through a candidate suite."""

from copy import deepcopy
from dataclasses import replace
from typing import TypeVar, cast

from cle.harness.board_surface import board_presentation_payload
from cle.harness.communication import CommunicationSuite, build_communication_request
from cle.harness.context import ContextAssembler
from cle.harness.models import ModelRequest, PlayerSession
from cle.harness.suite import ContextSuite
from cle.players.contracts import PlayerContext, TalkContext
from cle.sandbox.catan import CatanSandbox

from ...state import ServerState
from .editor import component_payload

__all__ = [
    "communication_preview",
    "decision_preview",
    "latest_communication_request",
]

_Context = TypeVar("_Context", PlayerContext, TalkContext)


def _slice_fresh_notes(context: _Context, cursor: int) -> _Context:
    """Keep only what a fresh-notes player has not yet seen, up to the context cutoff."""
    cutoff = context.visible_through_sequence
    if cutoff is None:
        return context
    messages = tuple(
        event for event in context.visible_messages if cursor <= event.sequence <= cutoff
    )
    if isinstance(context, TalkContext):
        return replace(
            context,
            game_events=tuple(
                event for event in context.game_events if cursor <= event.sequence <= cutoff
            ),
            recent_messages=messages, visible_messages=messages,
        )
    return replace(
        context,
        events=tuple(event for event in context.events if cursor <= event.sequence <= cutoff),
        recent_messages=messages, visible_messages=messages,
    )


def decision_preview(
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
        cursor = session.action_next_sequence if session.context_policy == "fresh_notes" else 0
        context = _slice_fresh_notes(context, cursor)
    presenter = getattr(getattr(player, "_assembler", None), "board_presenter", None)
    model_request = ContextAssembler(suite, board_presenter=presenter).assemble(context, session)
    return {
        "status": "rendered",
        "provenance": "current_typed_context",
        "actor": actor.value,
        "prompt_key": context.prompt_key,
        "components": [
            component_payload(component)
            for component in model_request.components
        ],
        "board_presentation": board_presentation_payload(
            model_request.board_presentation,
            include_text_content=True,
        ),
    }


def latest_communication_request(state: ServerState) -> ModelRequest | None:
    sandbox = getattr(state, "current_sandbox", None)
    if sandbox is None:
        return None
    for record in reversed(getattr(sandbox, "communication_trace", ())):
        choice = record.choice
        if choice.model_request is not None:
            return cast(ModelRequest, choice.model_request)
    return None


def _communication_context(state: ServerState) -> tuple[CatanSandbox, TalkContext, str] | None:
    """The live sandbox and talk context a preview renders, and how its trigger was found."""
    sandbox = getattr(state, "current_sandbox", None)
    if not isinstance(sandbox, CatanSandbox):
        return None
    opportunities = sandbox.communication_policy.pre_action(sandbox.game_engine)
    if opportunities:
        opportunity, provenance = opportunities[0], "current_typed_context"
    elif sandbox.communication_trace:
        opportunity = replace(
            sandbox.communication_trace[-1].opportunity,
            visible_through_sequence=sandbox.game_engine.revision - 1,
        )
        provenance = "current_typed_context_with_latest_trigger"
    else:
        return None
    context = sandbox.talk_context(opportunity)
    # Legacy-policy players get no observation, but every preview renders the board.
    context = replace(context, observation=sandbox.game_engine.observe(context.player))
    return sandbox, context, provenance


def _rendered_communication(
    sandbox: CatanSandbox,
    context: TalkContext,
    provenance: str,
    suite: CommunicationSuite,
) -> dict[str, object]:
    player = sandbox.players.get(context.player)
    session = cast(PlayerSession | None, getattr(player, "session", None))
    notes = ""
    if suite.memory_mode == "fresh_notes":
        cursor = 0
        if session is not None:
            notes = session.strategic_memory
            if session.context_policy == "fresh_notes":
                cursor = session.talk_next_sequence
        context = _slice_fresh_notes(context, cursor)
    model_request = build_communication_request(
        context, "prompt-preview", suite, notes=notes,
        board_presenter=getattr(getattr(player, "_assembler", None), "board_presenter", None),
    )
    return {
        "status": "rendered", "actor": context.player.value, "provenance": provenance,
        "components": [component_payload(item) for item in model_request.components],
        "board_presentation": board_presentation_payload(
            model_request.board_presentation, include_text_content=True,
        ),
    }


def _authored_templates(suite: CommunicationSuite) -> dict[str, object]:
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


def _recorded_request(recorded: ModelRequest) -> dict[str, object]:
    return {
        "status": "rendered",
        "provenance": "recorded_request",
        "components": [component_payload(component) for component in recorded.components],
    }


def _legacy_combined_request(recorded: ModelRequest) -> dict[str, object]:
    user_messages = [message.content for message in recorded.messages if message.role == "user"]
    content = user_messages[-1] if user_messages else ""
    return {
        "status": "legacy_combined_request",
        "components": [
            {
                "id": "environment.legacy_combined_request",
                "channel": "environment",
                "template": "",
                "value": content,
                "rendered": content,
                "variables": {},
            }
        ],
    }


def _legacy_templates(suite: CommunicationSuite) -> dict[str, object]:
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


def communication_preview(
    state: ServerState,
    suite: CommunicationSuite,
) -> dict[str, object]:
    if suite.components:
        found = _communication_context(state)
        if found is None:
            return _authored_templates(suite)
        return _rendered_communication(*found, suite)
    # Non-component (pinned legacy) suites preview read-only from recorded requests.
    recorded = latest_communication_request(state)
    if recorded is not None and recorded.components:
        return _recorded_request(recorded)
    if recorded is not None:
        return _legacy_combined_request(recorded)
    return _legacy_templates(suite)
