"""
VLM-powered player for Catan.

Uses a vision-language model (frontend screenshot + text observation) to make
strategic decisions. Falls back to text-only Groq when vision is unavailable.

Prompting approach adapted from the VLM playground:
- System prompt as separate role message with rules + format
- GAME_PLAN (persistent) + TURN_PLAN + ACTION output format
- Turn traces for multi-step awareness within a turn
- Frontend board screenshot via Playwright + rich action descriptions
"""

import io
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image
from groq import Groq
from playwright.sync_api import sync_playwright

from game_engine.models.player import Color
from cle.players.legacy import Player
from game_engine.models.enums import Action, ActionType
from cle.env.observation_formatter import (
    CatanObservationFormatter,
    create_observation_from_state,
)
from playground.openrouter_client import query_vlm, MODELS as VLM_MODELS, PROVIDERS

_SCREENSHOT_DIR = Path(__file__).resolve().parent.parent.parent / "playground" / "screenshots"

# Crop settings tuned for the frontend board (from VLM playground)
_CROP_PCT = 0.17
_VERTICAL_OFFSET_PCT = -0.017
_HORIZONTAL_OFFSET_PCT = 0.022


class _SharedBrowser:
    """Lazy singleton Playwright browser for frontend screenshots."""

    _instance = None

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._page = None
        self._ready = False

    @classmethod
    def get(cls) -> '_SharedBrowser':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def ensure_started(self, vite_url: str = "http://localhost:5173"):
        """Start browser if not already running."""
        if self._ready:
            return

        print("[screenshot] Starting Playwright (headless, sync)...")
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=True)
        self._page = self._browser.new_page(
            viewport={"width": 1200, "height": 900}
        )
        self._page.goto(vite_url)
        self._page.wait_for_selector(".app", timeout=10000)
        # Let SocketIO connect and receive initial state
        self._page.wait_for_timeout(1500)
        self._ready = True
        print("[screenshot] Browser ready.")

    def screenshot_board(self, settle_ms: int = 400) -> bytes:
        """Screenshot the .board-container and crop to the hex grid."""
        if not self._ready:
            raise RuntimeError("Browser not started")

        self._page.wait_for_timeout(settle_ms)

        board = self._page.locator(".board-container")
        board.wait_for(state="visible", timeout=5000)
        raw_png = board.screenshot(type="png")

        # Crop to center on hex grid (same settings as VLM playground)
        img = Image.open(io.BytesIO(raw_png))
        w, h = img.size
        cx = _CROP_PCT * w
        cy = _CROP_PCT * h
        vx = _HORIZONTAL_OFFSET_PCT * w
        vy = _VERTICAL_OFFSET_PCT * h
        box = (int(cx + vx), int(cy + vy), int(w - cx + vx), int(h - cy + vy))
        img = img.crop(box)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def stop(self):
        if self._page:
            self._page.close()
            self._page = None
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
        self._ready = False


class LLMPlayer(Player):
    """
    Player that uses a VLM to make decisions.

    Captures the frontend board via Playwright, builds a text observation,
    and sends both to a vision model. Falls back to Groq text-only if no VLM key.
    """

    def __init__(
        self,
        color: Color,
        vision_model: str = "glm_4_6v_novita",
        groq_model: str = "openai/gpt-oss-120b",
        temperature: float = 1.0,
    ):
        super().__init__(color)
        self.vision_model = vision_model
        self.groq_model = groq_model
        self.temperature = temperature

        # Determine which backend to use: VLM (preferred) or Groq (fallback)
        provider, model_id = VLM_MODELS[self.vision_model]
        vlm_key_env = PROVIDERS[provider]["key_env"]
        self.has_vlm = bool(os.getenv(vlm_key_env))

        groq_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=groq_key) if groq_key else None

        if not self.has_vlm and not self.groq_client:
            print(f"[{color}] WARNING: No VLM or Groq API key set. Will use random actions.")

        self.use_vision = self.has_vlm
        self.formatter = CatanObservationFormatter()

        # Store last decision info for debugging/visualization (frontend reads these)
        self.last_reasoning = None
        self.last_game_plan = None
        self.last_observation = None
        self.last_screenshot = None  # PNG bytes of latest board screenshot

        # Persistent game plan - evolves across turns (frontend reads as strategic_notes)
        self.strategic_notes = None

        # Turn trace state - tracks multi-step actions within a single turn
        self.turn_traces = []
        self.turn_number = None

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
        self.event_queue = []
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

    def _build_system_prompt(self, obs, playable_actions: List[Action]) -> str:
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

    def _format_actions_rich(self, playable_actions: List[Action], obs) -> str:
        """Format actions with flat indices using the formatter's rich descriptions."""
        lines = ["VALID ACTIONS (all listed actions are legal and affordable -- do not re-check costs):"]
        for i, action in enumerate(playable_actions):
            desc = self.formatter._format_single_action(action, obs)
            lines.append(f"  {i}. {desc}")
        return "\n".join(lines)

    def _build_prior_context(self) -> str:
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

    def _capture_board(self, game) -> Optional[bytes]:
        """Capture the frontend board via Playwright and save to disk."""
        try:
            browser = _SharedBrowser.get()
            browser.ensure_started()
            image_bytes = browser.screenshot_board()
        except Exception as e:
            print(f"[{self.color}] Screenshot failed: {e}")
            return None

        self.last_screenshot = image_bytes

        # Save to disk for verification
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        latest_path = _SCREENSHOT_DIR / "latest.png"
        latest_path.write_bytes(image_bytes)
        turn_path = _SCREENSHOT_DIR / f"turn_{game.state.num_turns}_{self.color.value}.png"
        turn_path.write_bytes(image_bytes)
        print(f"Screenshot saved: {latest_path} ({len(image_bytes)} bytes)")

        return image_bytes

    def _call_vlm(self, system_prompt: str, user_prompt: str, image_bytes: bytes) -> tuple:
        """Call VLM via OpenRouter/Novita. Returns (response_text, api_time_sec)."""
        provider, model_id = VLM_MODELS[self.vision_model]
        print(f"Calling VLM: {model_id} via {provider}...")

        result = query_vlm(
            model_id,
            image_bytes,
            user_prompt,
            system_prompt=system_prompt,
            provider=provider,
            temperature=self.temperature,
            max_tokens=8192,
        )

        api_time = result['latency_ms'] / 1000
        usage = result.get('usage', {})
        print(f"VLM call completed: {api_time:.3f}s")
        print(f"Tokens: {usage}")

        return result['content'], api_time

    def _call_groq(self, system_prompt: str, user_prompt: str) -> tuple:
        """Call Groq text-only API. Returns (response_text, api_time_sec)."""
        print(f"Calling Groq API with model: {self.groq_model}...")

        api_start = time.time()
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            max_tokens=8192,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        api_time = time.time() - api_start
        print(f"Groq call completed: {api_time:.3f}s")
        print(f"Input tokens: {response.usage.prompt_tokens}")
        print(f"Output tokens: {response.usage.completion_tokens}")

        return response.choices[0].message.content.strip(), api_time

    def decide(self, game, playable_actions: List[Action]) -> Action:
        """Make a decision by asking the VLM which action to take."""
        import random as _random

        start_time = time.time()
        print(f"\n{'='*80}")
        print(f"[{self.color}] LLM DECISION START (vision={'ON' if self.use_vision else 'OFF'})")
        print(f"{'='*80}")
        print(f"Timestamp: {time.strftime('%H:%M:%S')}")
        print(f"Turn: {game.state.num_turns}")
        print(f"Available actions: {len(playable_actions)}")

        if not self.has_vlm and not self.groq_client:
            print(f"\n[{self.color}] No API key - falling back to random action")
            return _random.choice(playable_actions)

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
        actions_text = self._format_actions_rich(playable_actions, obs)
        recent_events = self._consume_events()
        prior_context = self._build_prior_context()
        system_prompt = self._build_system_prompt(obs, playable_actions)

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
            return _random.choice(playable_actions)

        # Parse XML response
        try:
            game_plan_match = re.search(r'<game_plan>(.*?)</game_plan>', choice_text, re.DOTALL)
            turn_plan_match = re.search(r'<turn_plan>(.*?)</turn_plan>', choice_text, re.DOTALL)
            action_match = re.search(r'<action>\s*(\d+)\s*</action>', choice_text, re.DOTALL)

            if game_plan_match:
                self.strategic_notes = game_plan_match.group(1).strip()

            self.last_game_plan = game_plan_match.group(1).strip() if game_plan_match else "No plan stated"
            self.last_reasoning = turn_plan_match.group(1).strip() if turn_plan_match else "No reasoning stated"

            if action_match:
                choice_idx = int(action_match.group(1))
            else:
                choice_idx = self._parse_action_choice(choice_text, len(playable_actions))

            if choice_idx < 0 or choice_idx >= len(playable_actions):
                print(f"\n[{self.color}] INVALID INDEX {choice_idx} (range 0-{len(playable_actions)-1})")
                print(f"Full response:\n{choice_text}")
                print("Falling back to first valid action")
                choice_idx = 0

            selected_action = playable_actions[choice_idx]

            # Track turn traces
            action_desc = self.formatter._format_single_action(selected_action, obs)
            self.turn_traces.append({
                'action_desc': action_desc,
                'turn_plan': self.last_reasoning[:200],
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
            return playable_actions[0]

    def _parse_action_choice(self, text: str, num_actions: int) -> int:
        """Fallback: extract first number from LLM response as action index."""
        numbers = re.findall(r'\b\d+\b', text)

        if not numbers:
            raise ValueError(f"No number found in response: {text}")

        idx = int(numbers[0])

        if idx < 0 or idx >= num_actions:
            raise ValueError(f"Index {idx} out of range [0, {num_actions-1}]")

        return idx
