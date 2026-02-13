"""
LLM-powered player for Catan using Groq API.

This player implements the Catanatron Player interface but uses
an LLM to make decisions based on semantic text observations.
"""

import os
from typing import List
from groq import Groq
from engine.models.player import Player, Color
from engine.models.enums import Action, ActionType
from cle.env.observation_formatter import (
    CatanObservationFormatter,
    create_observation_from_state,
)


class LLMPlayer(Player):
    """
    Player that uses Groq API to make decisions.

    Converts game state to semantic text, sends to LLM, parses response
    to select an action from the valid action list.
    """

    def __init__(
        self,
        color: Color,
        model: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
    ):
        """
        Initialize LLM player.

        Args:
            color: Player color
            model: Groq model to use (default: openai/gpt-oss-120b)
            temperature: Sampling temperature (1.0 for exploration)
        """
        super().__init__(color)
        self.model = model
        self.temperature = temperature

        # Initialize Groq client (lazy - only if API key exists)
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None
        self.formatter = CatanObservationFormatter()

        # Store last decision info for debugging/visualization
        self.last_reasoning = None
        self.last_game_plan = None
        self.last_observation = None  # Full observation text

        # Persistent strategic memory - notes to self that evolve over time
        self.strategic_notes = None  # Multi-line strategic scratchpad

        # Event queue - tracks what happened since last turn
        self.event_queue = []

    def log_event(self, event_str: str):
        """Add an event to the queue (called by game server)."""
        self.event_queue.append(event_str)

    def _consume_events(self) -> str:
        """Consume and clear event queue, return formatted string."""
        if not self.event_queue:
            return ""

        events = "\n".join(f"  - {event}" for event in self.event_queue)
        self.event_queue = []  # Clear queue
        return f"\nRECENT EVENTS SINCE YOUR LAST TURN:\n{events}\n"

    def _prompt_suite(self, obs, playable_actions: List[Action]) -> str:
        """Return phase/action-specific guidance to reduce model confusion."""
        action_types = {
            a.action_type for a in playable_actions if hasattr(a, "action_type")
        }

        phase = getattr(obs, "current_phase", None)
        in_initial = phase == "initial_placement"

        if not in_initial:
            return (
                "PROMPT SUITE (main game):\n"
                "- Prefer actions that increase VP efficiently (cities, settlements, longest road/largest army).\n"
                "- Use trades to fix bottlenecks; avoid ending turn with a clear build available unless strategically necessary.\n"
                "- When evaluating production, use pips (probability) and resource diversity, not raw dice numbers.\n"
            )

        # Initial placement
        placed = len(getattr(obs, "my_settlements", []) or [])

        if ActionType.BUILD_SETTLEMENT in action_types:
            if placed <= 0:
                return (
                    "PROMPT SUITE (initial placement - 1st settlement):\n"
                    "- You do NOT gain starting resources from the 1st settlement. Optimize for long-term production.\n"
                    "- Prioritize high total pips, strong dice numbers (6/8 are best), and resource diversity.\n"
                    "- Avoid over-committing to a single resource unless a port plan is obvious.\n"
                    "- Prefer placements that keep future expansion options open (don’t self-block).\n"
                )
            if placed == 1:
                return (
                    "PROMPT SUITE (initial placement - 2nd settlement):\n"
                    "- You DO gain starting resources from the 2nd settlement: 1 card from each adjacent non-desert tile.\n"
                    "- Optimize for a strong opening hand: aim to enable early road/settlement/city/dev card plans.\n"
                    "- Fill missing resources from your first settlement; prioritize WHEAT/ORE if you want early cities/devs.\n"
                    "- Consider port synergy if it matches your production mix, but don’t sacrifice too many pips.\n"
                )
            return (
                "PROMPT SUITE (initial placement - settlement):\n"
                "- During initial placement, only the 2nd settlement grants starting resources.\n"
                "- Prefer high pips and good future expansion.\n"
            )

        if ActionType.BUILD_ROAD in action_types:
            return (
                "PROMPT SUITE (initial placement - road):\n"
                "- Place the road to preserve future settlement spots and flexibility.\n"
                "- Prefer roads that lead to multiple viable expansion nodes (branching potential).\n"
                "- Avoid roads that immediately dead-end or block your own best follow-up settlement locations.\n"
            )

        return (
            "PROMPT SUITE (initial placement):\n"
            "- Remember: only the 2nd settlement grants starting resources.\n"
        )

    def decide(self, game, playable_actions: List[Action]) -> Action:
        """
        Make a decision by asking Claude which action to take.

        Args:
            game: Current game state
            playable_actions: List of valid actions

        Returns:
            Selected action
        """
        import time

        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[{self.color}] LLM DECISION START")
        print(f"{'='*80}")
        print(f"Timestamp: {time.strftime('%H:%M:%S')}")
        print(f"Turn: {game.state.num_turns}")
        print(f"Available actions: {len(playable_actions)}")

        if not self.client:
            print(f"\n[{self.color}] No API key - falling back to random action")
            import random
            return random.choice(playable_actions)

        # Get semantic observation
        obs_start = time.time()
        obs = create_observation_from_state(game.state, self.color)
        formatted_obs = self.formatter.format(obs)
        obs_time = time.time() - obs_start
        print(f"Observation creation: {obs_time:.3f}s")

        # Store observation for debugging
        self.last_observation = formatted_obs.raw_str

        # Format actions with indices
        action_descriptions = self._format_actions(playable_actions)

        # Consume event queue
        recent_events = self._consume_events()

        suite = self._prompt_suite(obs, playable_actions)

        # Include strategic notes if they exist
        strategic_notes_section = ""
        if self.strategic_notes:
            strategic_notes_section = f"""
YOUR STRATEGIC NOTES (from previous turns):
{self.strategic_notes}

Review these notes and update them based on:
- What happened since your last turn (events above)
- New opportunities or threats you see
- Changes to your priorities or plans
- Things to remember for future turns
"""
        else:
            strategic_notes_section = """
STRATEGIC NOTES:
This is your first turn. Start building your strategic scratchpad with:
- Your overall game plan and priorities
- Specific goals or conditions to watch for
- Reminders about opponents' positions
- Ideas for future turns based on resource availability
"""

        # Build prompt
        prompt = f"""You are playing Settlers of Catan as {self.color}.

IMPORTANT CLARIFICATION ABOUT NUMBERS:
- Dice numbers are the roll outcomes (2-12), NOT the pip count.
- Pip counts are explicitly provided as pips=<0-5> for each adjacent tile.

{suite}

CURRENT GAME STATE:
{formatted_obs.raw_str}
{recent_events}
{strategic_notes_section}

VALID ACTIONS (choose by index 0-{len(playable_actions)-1}):
{action_descriptions}

IMPORTANT: You must choose an action index from 0 to {len(playable_actions)-1} (there are {len(playable_actions)} valid actions).

Format your response as:
STRATEGIC_NOTES: <updated notes to yourself - can be multi-line bullet points>
GAME_PLAN: <your overall strategy for this turn>
REASONING: <why this action makes sense now>
ACTION: <index number between 0 and {len(playable_actions)-1}>

Think strategically about:
- Building toward victory (10 VP)
- Resource production and diversity
- Blocking opponents
- Longest road / largest army opportunities"""

        # Show full observation at the top
        print(f"\n{'#'*80}")
        print(f"OBSERVATION INPUT TO LLM")
        print(f"{'#'*80}")
        print(f"\nGAME STATE:")
        print(formatted_obs.raw_str)
        if recent_events:
            print(recent_events)
        print(f"\nVALID ACTIONS ({len(playable_actions)} total):")
        print(action_descriptions)
        print(f"\n{'#'*80}")
        print(f"END OBSERVATION INPUT")
        print(f"{'#'*80}\n")

        # Call Groq
        api_start = time.time()
        print(f"Calling Groq API with model: {self.model}...")
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=8192,
            temperature=self.temperature,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        api_time = time.time() - api_start
        print(f"API call completed: {api_time:.3f}s")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")

        # Parse response
        choice_text = response.choices[0].message.content.strip()

        try:
            # Parse strategic notes, game plan, reasoning, and action
            import re

            strategic_notes_match = re.search(r'STRATEGIC_NOTES:\s*(.+?)(?=GAME_PLAN:|REASONING:|ACTION:|$)', choice_text, re.DOTALL)
            game_plan_match = re.search(r'GAME_PLAN:\s*(.+?)(?=REASONING:|ACTION:|$)', choice_text, re.DOTALL)
            reasoning_match = re.search(r'REASONING:\s*(.+?)(?=ACTION:|$)', choice_text, re.DOTALL)
            action_match = re.search(r'ACTION:\s*(\d+)', choice_text)

            # Update persistent strategic notes
            if strategic_notes_match:
                self.strategic_notes = strategic_notes_match.group(1).strip()

            self.last_game_plan = game_plan_match.group(1).strip() if game_plan_match else "No plan stated"
            self.last_reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning stated"

            # Try to extract action index
            if action_match:
                choice_idx = int(action_match.group(1))
            else:
                # Fallback to old parsing
                choice_idx = self._parse_action_choice(choice_text, len(playable_actions))

            # Bounds check - ensure index is valid
            if choice_idx < 0 or choice_idx >= len(playable_actions):
                print(f"\n{'='*80}")
                print(f"[{self.color}] LLM ERROR - INVALID INDEX")
                print(f"{'='*80}")
                print(f"Chose index: {choice_idx} (valid range: 0-{len(playable_actions)-1})")
                print(f"\nFull LLM response:")
                print(choice_text)
                print(f"\nFalling back to first valid action")
                print(f"{'='*80}\n")
                choice_idx = 0

            selected_action = playable_actions[choice_idx]

            # Calculate total time
            total_time = time.time() - start_time

            # Print full LLM output to console
            print(f"\n--- FULL LLM RESPONSE ({len(choice_text)} chars) ---")
            print(choice_text)
            print(f"--- END RESPONSE ---\n")

            print(f"\n{'='*80}")
            print(f"[{self.color}] LLM DECISION COMPLETE")
            print(f"{'='*80}")
            if self.strategic_notes:
                print(f"\nSTRATEGIC NOTES:")
                print(self.strategic_notes)
            print(f"\nGAME PLAN:")
            print(self.last_game_plan)
            print(f"\nREASONING:")
            print(self.last_reasoning)
            print(f"\nCHOSEN ACTION: {choice_idx}")
            print(f"{selected_action}")
            print(f"\nPERFORMANCE:")
            print(f"  Observation: {obs_time:.3f}s")
            print(f"  API Call: {api_time:.3f}s")
            print(f"  Total: {total_time:.3f}s")
            print(f"{'='*80}\n")

            return selected_action

        except (ValueError, IndexError) as e:
            print(f"\n[{self.color}] LLM response parsing failed: {choice_text}")
            print(f"Error: {e}")
            print(f"Falling back to first valid action")
            self.last_game_plan = "Parsing failed"
            self.last_reasoning = choice_text[:200]
            return playable_actions[0]

    def _format_actions(self, actions: List[Action]) -> str:
        """Format actions with indices for LLM to choose."""
        lines = []
        for i, action in enumerate(actions):
            # Group by action type for readability
            action_str = str(action)
            lines.append(f"{i}. {action_str}")

        return "\n".join(lines)

    def _parse_action_choice(self, text: str, num_actions: int) -> int:
        """
        Parse LLM response to extract action index.

        Args:
            text: LLM response text
            num_actions: Number of valid actions

        Returns:
            Action index (0-based)

        Raises:
            ValueError: If no valid index found
        """
        # Try to find a number in the response
        import re

        # Look for standalone number
        numbers = re.findall(r'\b\d+\b', text)

        if not numbers:
            raise ValueError(f"No number found in response: {text}")

        # Take first number
        idx = int(numbers[0])

        if idx < 0 or idx >= num_actions:
            raise ValueError(f"Index {idx} out of range [0, {num_actions-1}]")

        return idx
