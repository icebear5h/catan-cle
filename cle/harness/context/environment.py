"""Environment-channel section values for the legacy per-section context modes."""

from __future__ import annotations

from cle.env.observation_formatter import CatanObservationFormatter
from cle.harness.action_tools import render_action_tools
from cle.harness.context.formatting import color_name, format_events
from cle.harness.models import PlayerSession
from cle.harness.reasoning import reasoning_budget_for_attempt, reasoning_budget_instruction
from cle.harness.suite import ContextSuite
from cle.players.contracts import PlayerContext


def environment_values(
    suite: ContextSuite,
    context: PlayerContext,
    session: PlayerSession,
    guidance: str,
    feedback: str | None,
) -> dict[str, str]:
    """Render the deterministic environment section values for one decision."""
    formatter = CatanObservationFormatter()
    formatted = formatter.format(
        context.observation,
        include_legal_actions=False,
        include_initial_placement_order=(
            suite.context.initial_placement_order == "both_rounds"
        ),
    )
    if suite.response.format == "json":
        legal_actions = render_action_tools(context)
        decision_request = "Choose exactly one available tool with named arguments."
    else:
        action_descriptions = (
            formatter._format_single_action(
                action,
                context.observation,
                discard_count=(
                    context.discard_count if "discard" in suite.response.tags else None
                ),
            )
            for action in context.legal_actions
        )
        legal_actions = "\n".join(
            f"{index}. {description}"
            for index, description in enumerate(action_descriptions)
        )
        decision_request = "Choose exactly one zero-based index from VALID ACTIONS."
    if feedback:
        decision_request = (
            f"{decision_request}\n\nCORRECTION FROM THE SANDBOX:\n{feedback}"
        )
    decision_request = (
        f"{decision_request}\n\n"
        f"{reasoning_budget_instruction(reasoning_budget_for_attempt(context, feedback))}"
    )
    visible_events = format_events(context.events)
    values = {
        "strategic_memory": session.strategic_memory,
        "game_events": visible_events,
        "visible_events": visible_events,
        "observation": formatted.raw_str,
        "phase_info": formatted.strategic_context,
        "board_state": formatted.board_state,
        "resources": formatted.resources,
        "opponents": formatted.opponents,
        "trade_window": formatted.trade_context,
        "phase_guidance": guidance,
        "legal_actions": legal_actions,
        "decision_request": decision_request,
        "response_schema": suite.response.instruction,
    }
    if suite.context.social_context:
        values["recent_table_talk"] = format_events(context.recent_messages)
        values["commitments"] = "\n".join(
            f"{item.id}: {color_name(item.proposer)} to "
            f"{', '.join(color_name(color) for color in item.audience)}: "
            f"{item.condition} -> {item.promise} (expires turn {item.expires_turn})"
            for item in context.active_commitments
        )
    return values
