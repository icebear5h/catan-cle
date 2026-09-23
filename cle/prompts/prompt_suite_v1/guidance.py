"""V1 phase and situation-specific strategic guidance."""

from cle.game_engine.models.enums import ActionType
from cle.prompts.base import PromptContext


def build_strategic_guidance(context: PromptContext) -> str:
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
        return _initial_placement_guidance(settlements_placed, action_types)

    # Main game guidance
    if phase == "main_game":
        return _main_game_guidance(context)

    # Robber movement
    if phase == "moving_robber":
        return _robber_guidance(context)

    # Discarding
    if phase == "discarding":
        return _discard_guidance(context)

    return ""


def _initial_placement_guidance(
    settlements_placed: int, action_types: set[ActionType]
) -> str:
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


def _main_game_guidance(context: PromptContext) -> str:
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


def _robber_guidance(context: PromptContext) -> str:
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


def _discard_guidance(context: PromptContext) -> str:
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
