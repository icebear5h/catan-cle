"""The VLM player's single decision turn, from screenshot to parsed action."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from cle.env.observation_formatter import create_observation_from_state
from cle.game_engine.game import GameEngine
from cle.game_engine.models.enums import Action, ActionType

if TYPE_CHECKING:
    from cle.agents.llm_player import LLMPlayer

__all__: list[str] = []


def decide(
    self: LLMPlayer, game: GameEngine, playable_actions: Iterable[Action]
) -> Action:
    """Make a decision by asking the VLM which action to take."""
    actions = list(playable_actions)

    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"[{self.color}] LLM DECISION START (vision={'ON' if self.use_vision else 'OFF'})")
    print(f"{'='*80}")
    print(f"Timestamp: {time.strftime('%H:%M:%S')}")
    print(f"Turn: {game.state.num_turns}")
    print(f"Available actions: {len(actions)}")

    if not self.has_vlm and not self.groq_client:
        print(f"\n[{self.color}] No API key - falling back to random action")
        return random.choice(actions)

    # Detect new turn - clear turn traces
    current_turn = game.state.num_turns
    if self.turn_number is not None and current_turn != self.turn_number:
        print(f"--- New turn {current_turn} (was {self.turn_number}) - clearing turn traces ---")
        self.turn_traces = []
    self.turn_number = current_turn

    # Capture frontend board screenshot (if vision enabled)
    image_bytes = None
    if self.use_vision:
        image_bytes = self._capture_board(game)

    # Get semantic observation
    obs_start = time.time()
    obs = create_observation_from_state(game.state, self.color)
    formatted_obs = self.formatter.format(obs)
    obs_time = time.time() - obs_start
    print(f"Observation creation: {obs_time:.3f}s")

    self.last_observation = formatted_obs.raw_str

    # Build prompts
    actions_text = self._format_actions_rich(actions, obs)
    recent_events = self._consume_events()
    prior_context = self._build_prior_context()
    system_prompt = self._build_system_prompt(obs, actions)

    user_prompt = f"""CURRENT GAME STATE:
{formatted_obs.raw_str}
{recent_events}
{prior_context}

{actions_text}

Pick the next action to execute."""

    # Debug output
    print(f"\n{'#'*80}")
    print("SYSTEM PROMPT")
    print(f"{'#'*80}")
    print(system_prompt)
    print(f"\n{'#'*80}")
    print("USER PROMPT (truncated)")
    print(f"{'#'*80}")
    print(formatted_obs.raw_str[:500])
    print(f"...\n{actions_text}")
    print(f"{'#'*80}\n")

    # Call the model
    if image_bytes and self.has_vlm:
        choice_text, api_time = self._call_vlm(system_prompt, user_prompt, image_bytes)
    elif self.groq_client:
        choice_text, api_time = self._call_groq(system_prompt, user_prompt)
    else:
        print(f"[{self.color}] No backend available - random action")
        return random.choice(actions)

    # Parse XML response
    try:
        game_plan_match = re.search(r'<game_plan>(.*?)</game_plan>', choice_text, re.DOTALL)
        turn_plan_match = re.search(r'<turn_plan>(.*?)</turn_plan>', choice_text, re.DOTALL)
        action_match = re.search(r'<action>\s*(\d+)\s*</action>', choice_text, re.DOTALL)

        if game_plan_match:
            self.strategic_notes = game_plan_match.group(1).strip()

        self.last_game_plan = game_plan_match.group(1).strip() if game_plan_match else "No plan stated"
        turn_plan = turn_plan_match.group(1).strip() if turn_plan_match else "No reasoning stated"
        self.last_reasoning = turn_plan

        if action_match:
            choice_idx = int(action_match.group(1))
        else:
            choice_idx = self._parse_action_choice(choice_text, len(actions))

        if choice_idx < 0 or choice_idx >= len(actions):
            print(f"\n[{self.color}] INVALID INDEX {choice_idx} (range 0-{len(actions)-1})")
            print(f"Full response:\n{choice_text}")
            print("Falling back to first valid action")
            choice_idx = 0

        selected_action = actions[choice_idx]

        # Track turn traces
        action_desc = self.formatter._format_single_action(selected_action, obs)
        self.turn_traces.append({
            'action_desc': action_desc,
            'turn_plan': turn_plan[:200],
            'action_idx': choice_idx,
        })

        if hasattr(selected_action, 'action_type') and selected_action.action_type == ActionType.END_TURN:
            print("--- END_TURN: clearing turn traces ---")
            self.turn_traces = []

        total_time = time.time() - start_time

        # Console output
        print(f"\n--- FULL LLM RESPONSE ({len(choice_text)} chars) ---")
        print(choice_text)
        print("--- END RESPONSE ---\n")

        print(f"\n{'='*80}")
        print(f"[{self.color}] LLM DECISION COMPLETE")
        print(f"{'='*80}")
        if self.strategic_notes:
            print("\nGAME PLAN (persistent):")
            print(self.strategic_notes)
        print("\nTURN PLAN:")
        print(self.last_reasoning)
        print(f"\nCHOSEN ACTION: {choice_idx} - {action_desc}")
        print("\nPERFORMANCE:")
        print(f"  Observation: {obs_time:.3f}s")
        print(f"  API Call: {api_time:.3f}s")
        print(f"  Total: {total_time:.3f}s")
        print(f"{'='*80}\n")

        return selected_action

    except (ValueError, IndexError) as e:
        print(f"\n[{self.color}] LLM response parsing failed: {choice_text}")
        print(f"Error: {e}")
        print("Falling back to first valid action")
        self.last_game_plan = "Parsing failed"
        self.last_reasoning = choice_text[:200]
        return actions[0]
