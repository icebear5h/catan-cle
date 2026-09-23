"""System prompt, phase guidance, and prior-context assembly for the VLM player."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cle.env.observation_formatter import Observation
from cle.game_engine.models.enums import Action, ActionType

if TYPE_CHECKING:
    from cle.agents.llm_player import LLMPlayer

__all__: list[str] = []


def _consume_events(self: LLMPlayer) -> str:
    """Consume and clear event queue, return formatted string."""
    if not self.event_queue:
        return ""

    events = "\n".join(f"  - {event}" for event in self.event_queue)
    self.event_queue = []
    return f"\nRECENT EVENTS SINCE YOUR LAST TURN:\n{events}\n"


def _prompt_suite(
    self: LLMPlayer, obs: Observation, playable_actions: Sequence[Action]
) -> str:
    """Return phase/action-specific guidance to reduce model confusion."""
    action_types = {
        a.action_type for a in playable_actions if hasattr(a, "action_type")
    }

    phase = getattr(obs, "current_phase", None)
    in_initial = phase == "initial_placement"

    if not in_initial:
        return (
            "Phase: MAIN GAME\n"
            "- Prefer actions that increase VP efficiently (cities > settlements > dev cards).\n"
            "- Use trades to fix bottlenecks; avoid ending turn with a clear build available.\n"
            "- Think in sequences: trade -> build -> buy dev card -> end turn.\n"
        )

    placed = len(getattr(obs, "my_settlements", []) or [])

    if ActionType.BUILD_SETTLEMENT in action_types:
        if placed <= 0:
            return (
                "Phase: INITIAL PLACEMENT - 1st Settlement\n"
                "- You do NOT gain starting resources from the 1st settlement.\n"
                "- Prioritize high total pips, strong dice numbers (6/8 best), and resource diversity.\n"
                "- Avoid over-committing to a single resource unless a port plan is obvious.\n"
                "\n"
                "<game_plan> requirements for this phase:\n"
                "- Identify 2-3 candidate spots for your SECOND settlement and why they pair well.\n"
                "- What resource combination the 1st+2nd pair gives you.\n"
                "- Whether you lean longest road, largest army, or balanced.\n"
                "- Compare the top 2-3 nodes by pip total and resource mix.\n"
                "- Explain why your pick beats the alternatives.\n"
                "\n"
                "<turn_plan>: just state which node you chose. No elaboration needed.\n"
            )
        if placed == 1:
            return (
                "Phase: INITIAL PLACEMENT - 2nd Settlement\n"
                "- You DO gain starting resources from the 2nd settlement (1 card per adjacent non-desert tile).\n"
                "- Fill resource gaps from your 1st settlement; WHEAT/ORE for early cities/devs.\n"
                "- Consider port synergy if it matches your production mix.\n"
                "\n"
                "<game_plan> requirements for this phase:\n"
                "- Detail a concrete path to 10 VP: longest road, largest army, or balanced.\n"
                "- Which resources you need most and which tiles produce them.\n"
                "- Your first 3-4 builds after initial placement (e.g. road -> settlement -> city).\n"
                "- How your two settlements complement each other.\n"
                "- What starting resources this node gives you and what that enables turn 1.\n"
                "- Why this node over the other top candidates.\n"
                "\n"
                "<turn_plan>: just state which node you chose. No elaboration needed.\n"
            )
        return (
            "Phase: INITIAL PLACEMENT - Settlement\n"
            "- Only the 2nd settlement grants starting resources.\n"
            "- Prefer high pips and good future expansion.\n"
            "\n"
            "<turn_plan>: just state which node you chose. No elaboration needed.\n"
        )

    if ActionType.BUILD_ROAD in action_types:
        return (
            "Phase: INITIAL PLACEMENT - Road\n"
            "- Place the road to preserve future settlement spots and flexibility.\n"
            "- Prefer roads that lead to multiple viable expansion nodes.\n"
            "\n"
            "<turn_plan>: just state which road you chose. No elaboration needed.\n"
        )

    return (
        "Phase: INITIAL PLACEMENT\n"
        "- Only the 2nd settlement grants starting resources.\n"
    )


def _build_system_prompt(
    self: LLMPlayer, obs: Observation, playable_actions: Sequence[Action]
) -> str:
    """Build system prompt with role, rules, phase guidance, and format."""
    suite = self._prompt_suite(obs, playable_actions)

    vision_note = ""
    if self.use_vision:
        vision_note = "You are also shown an image of the current board. Use it for spatial reasoning.\n"

    return f"""You are an expert Settlers of Catan player playing as {self.color}.
{vision_note}
Rules:
- Every action listed is legal and affordable. Do not re-check costs.
- Catan turns are multi-step: you can trade, build, and buy in sequence before ending your turn.
- Think in sequences: "if I trade 4 WOOD for 1 ORE, then I can afford a city."
- Pip counts represent probability (5 pips = most likely). Higher pips = better production.
- Dice numbers are the roll outcomes (2-12), NOT pip counts.

{suite}

You MUST respond using these XML tags (do NOT skip any tag):
<game_plan>your overall strategy to reach 10 VP - update each turn as circumstances change. See requirements above.</game_plan>
<turn_plan>your reasoning for this specific action and what sequence comes next. See requirements above.</turn_plan>
<action>index number of the action to execute</action>"""


def _format_actions_rich(
    self: LLMPlayer, playable_actions: Sequence[Action], obs: Observation
) -> str:
    """Format actions with flat indices using the formatter's rich descriptions."""
    lines = ["VALID ACTIONS (all listed actions are legal and affordable -- do not re-check costs):"]
    for i, action in enumerate(playable_actions):
        desc = self.formatter._format_single_action(action, obs)
        lines.append(f"  {i}. {desc}")
    return "\n".join(lines)


def _build_prior_context(self: LLMPlayer) -> str:
    """Build context from persistent game plan and turn traces."""
    parts = []
    if self.strategic_notes:
        parts.append(f"Your current game plan:\n{self.strategic_notes}")
    if self.turn_traces:
        trace_lines = []
        for t in self.turn_traces:
            trace_lines.append(f"  - {t['action_desc']} (reason: {t['turn_plan']})")
        parts.append("Actions taken this turn so far:\n" + "\n".join(trace_lines))
    if not parts:
        parts.append("This is your first decision. Establish your game plan.")
    return "\n\n".join(parts)
