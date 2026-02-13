"""
Prompt Suite V1: Comprehensive strategic guidance for Catan gameplay.

This version focuses on:
- Clear strategic principles
- Phase-specific guidance
- Pip-based thinking
- VP efficiency
- Common mistakes to avoid
"""

import re
from typing import Dict
from .base import PromptSuite, PromptContext
from engine.models.enums import ActionType


class PromptSuiteV1(PromptSuite):
    """
    V1 Prompt Suite - Comprehensive strategic Catan prompts.

    Emphasizes:
    - Probability-driven decisions (pips)
    - VP-efficient building (cities > settlements)
    - Resource diversity
    - Opponent modeling
    """

    version = "v1.0"

    def build_system_prompt(self) -> str:
        """Core system prompt defining agent role and capabilities."""
        return """You are an expert Settlers of Catan player. Your goal is to reach 10 Victory Points before your opponents.

CORE STRATEGIC PRINCIPLES:

1. PROBABILITY THINKING (Pips):
   - Each dice number has a probability (pips) from 0-5
   - 6 and 8 are best (5 pips each)
   - 5 and 9 are good (4 pips each)
   - 4 and 10 are decent (3 pips each)
   - 3 and 11 are weak (2 pips each)
   - 2 and 12 are terrible (1 pip each)
   - Total pips on a settlement shows production quality

2. VICTORY POINT EFFICIENCY:
   - Cities are more efficient than settlements (2 VP vs 1 VP, 2x production)
   - Build cities on your best production spots
   - Settlements should expand to new resources or block opponents
   - Development cards can provide VP and utility

3. RESOURCE STRATEGY:
   - Need diversity to build (settlement needs 4 different resources)
   - Wheat + Ore = cities and dev cards (powerful combination)
   - Wood + Brick = roads and settlements (expansion)
   - Sheep = trading currency and dev cards
   - Ports enable resource conversion (2:1 or 3:1)

4. GAME PHASES:
   - Initial Placement: Set up long-term production (HIGH impact)
   - Early Game (turns 1-15): Expand, diversify production
   - Mid Game (turns 16-30): Build cities, get dev cards
   - Late Game (31+): Race to 10 VP, block opponents

5. OPPONENT MODELING:
   - Track opponent VP and threat level
   - Block opponents approaching 10 VP
   - Contest longest road / largest army if valuable
   - Robber placement should target threats or bottleneck resources

COMMON MISTAKES TO AVOID:
- Don't overvalue rare numbers (2, 3, 11, 12) - they almost never roll
- Don't build settlements just because you can - think VP efficiency
- Don't ignore opponents at 7-8 VP - they can win next turn
- Don't waste turns - always progress toward VP if possible"""

    def build_strategic_guidance(self, context: PromptContext) -> str:
        """Phase and situation-specific guidance."""
        phase = context.current_phase
        settlements_placed = context.settlements_placed

        # Determine action types available
        action_types = {
            a.action_type for a in context.valid_actions
            if hasattr(a, 'action_type')
        }

        # Initial placement guidance
        if phase == "initial_placement":
            return self._initial_placement_guidance(settlements_placed, action_types)

        # Main game guidance
        if phase == "main_game":
            return self._main_game_guidance(context)

        # Robber movement
        if phase == "moving_robber":
            return self._robber_guidance(context)

        # Discarding
        if phase == "discarding":
            return self._discard_guidance(context)

        return ""

    def _initial_placement_guidance(self, settlements_placed: int, action_types: set) -> str:
        """Guidance for initial placement phase."""
        if ActionType.BUILD_SETTLEMENT in action_types:
            if settlements_placed == 0:
                return """INITIAL PLACEMENT - FIRST SETTLEMENT:
CRITICAL: You do NOT receive starting resources from your 1st settlement.
This is purely a long-term investment.

Priority order:
1. High total pips (12+ is excellent, 10+ is good, 8+ is okay)
2. Strong numbers (prefer 6/8, then 5/9, avoid 2/3/11/12)
3. Resource diversity (3 different resources > 2 of the same)
4. Future expansion potential (don't block yourself)
5. Port synergy (if exceptionally strong numbers)

AVOID:
- Over-committing to one resource (even if high pips)
- Placing near desert unless other tiles are very strong
- Blocking your own best expansion spots
- Relying on numbers with <3 pips

THINK LONG-TERM: This spot produces for the entire game."""

            elif settlements_placed == 1:
                return """INITIAL PLACEMENT - SECOND SETTLEMENT:
CRITICAL: You DO receive starting resources from your 2nd settlement.
You get 1 card from EACH adjacent non-desert tile.

Priority order:
1. Strong opening hand (aim for immediate road/settlement or dev card potential)
2. Fill resource gaps from first settlement
3. Wheat + Ore access enables cities and dev cards (very valuable)
4. Good pips still matter (9+ total is solid)
5. Port synergy if it matches your production

Opening hand checklist:
- Can I build a road/settlement soon? (need wood, brick, sheep, wheat)
- Can I buy a dev card? (need sheep, wheat, ore)
- Can I build a city early? (need 2 wheat, 3 ore)
- Do I have balanced resources?

AVOID:
- Sacrificing too many pips just for starting resources
- Taking same resources as first settlement unless very strong
- Ignoring wheat/ore completely (limits mid-game)"""

            else:
                return """INITIAL PLACEMENT - SETTLEMENT:
General principles:
- Balance pips and resource diversity
- Consider starting resources (2nd settlement only)
- Think about future expansion"""

        if ActionType.BUILD_ROAD in action_types:
            return """INITIAL PLACEMENT - ROAD:
Roads during initial placement should preserve flexibility.

Priority order:
1. Toward multiple good expansion spots (branching potential)
2. Along a future longest road path
3. Avoid dead-ends
4. Don't block your best follow-up settlements

AVOID:
- Roads that immediately block your expansion
- Over-committing to one direction without scouting alternatives"""

        return "Initial placement phase - set up your long-term strategy"

    def _main_game_guidance(self, context: PromptContext) -> str:
        """Guidance for main game phase."""
        vp_diff = context.my_vp - context.max_opponent_vp

        # Urgent situations
        if context.max_opponent_vp >= 9:
            return """URGENT - OPPONENT NEAR VICTORY:
An opponent is at 9+ VP and could win next turn!

Priority:
1. Can you win THIS TURN? (reach 10 VP immediately)
2. Can you block them? (longest road/largest army contest)
3. Use robber on their key production
4. Slow your own development if needed to block

DO NOT play normally - this is crisis mode."""

        if context.my_vp >= 9:
            return """WINNING POSITION - YOU'RE AT 9 VP:
You can win next turn! Focus on securing the win.

Priority:
1. Can you reach 10 VP this turn? DO IT.
2. If not, set up guaranteed win next turn
3. Protect your VP sources (don't lose longest road)
4. Consider blocking opponents who might catch up

You're in the lead - close it out."""

        # Normal main game
        if vp_diff > 2:
            strategy = "MAINTAIN LEAD: Keep growing while watching for threats."
        elif vp_diff < -2:
            strategy = "CATCH UP: Need aggressive expansion and smart trades."
        else:
            strategy = "COMPETITIVE GAME: Every VP matters."

        return f"""MAIN GAME STRATEGY:
{strategy}

General priority (adapt to situation):
1. Build cities on strong production spots (2 VP, 2x production)
2. Expand settlements only if:
   - Accessing new resources needed for cities
   - High-pip spots still available
   - Blocking opponents near victory
3. Buy dev cards if you have excess sheep/wheat/ore
4. Contest longest road / largest army if worth 2 VP
5. Trade to fix bottlenecks (don't hoard resources)

VP MATH:
- Cities = 2 VP each (best value)
- Settlements = 1 VP each (use for expansion)
- Dev card VP = 1 VP (hidden until played)
- Longest road = 2 VP (need 5+ road length)
- Largest army = 2 VP (need 3+ knights)

RESOURCE EFFICIENCY:
- Cities cost 2 wheat + 3 ore (expensive but 2 VP)
- Settlements cost wood + brick + sheep + wheat (1 VP)
- Always build cities if you can afford them

CURRENT SITUATION:
- Your VP: {context.my_vp}/10
- Opponent max VP: {context.max_opponent_vp}/10
- Turn: {context.turn_number}"""

    def _robber_guidance(self, context: PromptContext) -> str:
        """Guidance for moving the robber."""
        return """ROBBER PLACEMENT:
The robber blocks production and lets you steal a card.

Priority:
1. Block opponents near victory (8+ VP)
2. Target most valuable resource tiles (high pips, needed resources)
3. Block opponents with many resources to steal
4. Avoid your own tiles
5. Avoid tiles with no settlements (no steal opportunity)

Consider:
- Who is the biggest threat?
- What resource do they need most?
- Can you cripple their strategy?"""

    def _discard_guidance(self, context: PromptContext) -> str:
        """Guidance for discarding cards (7 rolled with >7 cards)."""
        return """DISCARDING (7 was rolled):
You must discard half your cards (rounded down).

Priority for KEEPING:
1. Resources needed for immediate builds (you can build this turn)
2. Scarce resources you can't easily get
3. Resources for trades you're planning
4. Wheat/Ore (for cities and dev cards)

Priority for DISCARDING:
1. Resources you have many of
2. Resources you produce frequently
3. Resources you can get via ports
4. Resources that don't fit your current strategy

THINK AHEAD: What are you trying to build next?"""

    def build_decision_prompt(self, context: PromptContext) -> str:
        """Build complete decision prompt."""
        strategic_guidance = self.build_strategic_guidance(context)

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
        prompt = f"""{self.build_system_prompt()}

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
{self._format_actions_list(context.valid_actions)}

{'='*80}

IMPORTANT REMINDERS:
- Dice numbers (2-12) are NOT the same as pips (probability 0-5)
- Numbers are explicitly shown as "dice=X, pips=Y" in the game state
- 6 and 8 have 5 pips (best), 2 and 12 have 1 pip (worst)
- You must choose an action index between 0 and {context.num_actions - 1}

Make your decision now."""

        return prompt

    def _format_actions_list(self, actions: list) -> str:
        """Format actions for display in prompt."""
        lines = []
        for i, action in enumerate(actions):
            lines.append(f"{i}. {str(action)}")
        return "\n".join(lines)

    def parse_response(self, response: str) -> dict:
        """Parse LLM response into structured format."""
        result = {
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
