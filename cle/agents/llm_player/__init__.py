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

from __future__ import annotations

import os

from groq import Groq

from cle.agents.llm_player import backends, browser, decision, prompts
from cle.agents.llm_player.browser import _SharedBrowser as _SharedBrowser
from cle.agents.llm_player.contracts import TurnTrace
from cle.env.observation_formatter import CatanObservationFormatter
from cle.game_engine.models.player import Color
from cle.players.legacy import Player
from playground.openrouter_client import MODELS as VLM_MODELS
from playground.openrouter_client import PROVIDERS

# Extracted helpers are bound onto the class below so every method keeps
# resolving through `cle.agents.llm_player`, exactly as before the split.

__all__ = ["LLMPlayer", "TurnTrace"]


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
    ) -> None:
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
        self.last_reasoning: str | None = None
        self.last_game_plan: str | None = None
        self.last_observation: str | None = None
        self.last_screenshot: bytes | None = None  # PNG bytes of latest board screenshot

        # Persistent game plan - evolves across turns (frontend reads as strategic_notes)
        self.strategic_notes: str | None = None

        # Turn trace state - tracks multi-step actions within a single turn
        self.turn_traces: list[TurnTrace] = []
        self.turn_number: int | None = None

        # Event queue - tracks what happened since last turn
        self.event_queue: list[str] = []

    def log_event(self, event_str: str) -> None:
        """Add an event to the queue (called by game server)."""
        self.event_queue.append(event_str)

    _consume_events = prompts._consume_events
    _prompt_suite = prompts._prompt_suite
    _build_system_prompt = prompts._build_system_prompt
    _format_actions_rich = prompts._format_actions_rich
    _build_prior_context = prompts._build_prior_context

    _capture_board = browser._capture_board

    _call_vlm = backends._call_vlm
    _call_groq = backends._call_groq
    _parse_action_choice = backends._parse_action_choice

    decide = decision.decide


LLMPlayer.__module__ = "cle.agents.llm_player"
