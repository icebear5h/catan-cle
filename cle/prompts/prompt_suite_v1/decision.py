"""V1 decision prompt assembly and response parsing."""

import re

from cle.game_engine.models.enums import Action
from cle.prompts.base import PromptContext
from cle.prompts.prompt_suite_v1.system import build_system_prompt


def build_decision_prompt(strategic_guidance: str, context: PromptContext) -> str:
    """Build complete decision prompt."""

    # Strategic notes section
    if context.strategic_notes:
        strategic_notes_section = f"""YOUR STRATEGIC NOTES (from previous turns):
{context.strategic_notes}

Update these notes based on:
- Recent events and new information
- Changes to your strategy or priorities
- Opponent threats and opportunities
- Things to remember for future turns"""
    else:
        strategic_notes_section = """YOUR STRATEGIC NOTES:
This is your first decision. Start building your strategic scratchpad:
- What's your overall game plan?
- What are your priorities?
- What should you remember about opponents?
- What are you working toward?"""

    # Build full prompt
    prompt = f"""{build_system_prompt()}

{'='*80}

{strategic_guidance}

{'='*80}

CURRENT GAME STATE:
{context.observation_text}

{'='*80}

{context.events_since_last_turn}

{'='*80}

{strategic_notes_section}

{'='*80}

DECISION FORMAT:
Respond with the following structure:

STRATEGIC_NOTES: <Your updated notes to yourself for future turns - be concise but specific>

GAME_PLAN: <What you're trying to achieve right now and in the next few turns>

REASONING: <Why you're choosing this specific action at this moment>

ACTION: <index number from 0 to {context.num_actions - 1}>

{'='*80}

VALID ACTIONS (choose index 0-{context.num_actions - 1}):
{_format_actions_list(context.valid_actions)}

{'='*80}

IMPORTANT REMINDERS:
- Dice numbers (2-12) are NOT the same as pips (probability 0-5)
- Numbers are explicitly shown as "dice=X, pips=Y" in the game state
- 6 and 8 have 5 pips (best), 2 and 12 have 1 pip (worst)
- You must choose an action index between 0 and {context.num_actions - 1}

Make your decision now."""

    return prompt


def _format_actions_list(actions: list[Action]) -> str:
    """Format actions for display in prompt."""
    lines = []
    for i, action in enumerate(actions):
        lines.append(f"{i}. {str(action)}")
    return "\n".join(lines)


def parse_response(response: str) -> dict[str, str | int | None]:
    """Parse LLM response into structured format."""
    result: dict[str, str | int | None] = {
        'strategic_notes': None,
        'game_plan': None,
        'reasoning': None,
        'action_index': None,
        'raw_response': response
    }

    # Parse sections using regex
    strategic_notes_match = re.search(
        r'STRATEGIC_NOTES:\s*(.+?)(?=GAME_PLAN:|REASONING:|ACTION:|$)',
        response,
        re.DOTALL
    )
    game_plan_match = re.search(
        r'GAME_PLAN:\s*(.+?)(?=REASONING:|ACTION:|$)',
        response,
        re.DOTALL
    )
    reasoning_match = re.search(
        r'REASONING:\s*(.+?)(?=ACTION:|$)',
        response,
        re.DOTALL
    )
    action_match = re.search(r'ACTION:\s*(\d+)', response)

    if strategic_notes_match:
        result['strategic_notes'] = strategic_notes_match.group(1).strip()
    if game_plan_match:
        result['game_plan'] = game_plan_match.group(1).strip()
    if reasoning_match:
        result['reasoning'] = reasoning_match.group(1).strip()
    if action_match:
        result['action_index'] = int(action_match.group(1))

    return result
