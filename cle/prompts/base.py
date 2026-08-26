"""
Base class for prompt suites.

Defines interface for building prompts for LLM decision-making.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from game_engine.models.enums import Action


@dataclass
class PromptContext:
    """Context needed to build a prompt."""

    # Game state
    observation_text: str
    valid_actions: List[Action]
    num_actions: int

    # Phase info
    current_phase: str
    turn_number: int

    # Player state
    settlements_placed: int
    my_vp: int
    max_opponent_vp: int

    # Recent events
    events_since_last_turn: str

    # Strategic memory
    strategic_notes: Optional[str] = None


class PromptSuite(ABC):
    """
    Base class for prompt engineering suites.

    Each version of prompts should subclass this and implement
    the abstract methods.
    """

    version: str = "base"

    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build the system prompt (role + high-level instructions).

        This sets the overall behavior and persona.
        """
        pass

    @abstractmethod
    def build_strategic_guidance(self, context: PromptContext) -> str:
        """
        Build phase-specific strategic guidance.

        This adapts based on game phase and situation.
        """
        pass

    @abstractmethod
    def build_decision_prompt(self, context: PromptContext) -> str:
        """
        Build the main decision prompt.

        This is the complete prompt sent to the LLM.
        """
        pass

    @abstractmethod
    def parse_response(self, response: str) -> dict:
        """
        Parse LLM response into structured format.

        Returns:
            {
                'strategic_notes': str,
                'game_plan': str,
                'reasoning': str,
                'action_index': int
            }
        """
        pass

    def get_version_info(self) -> dict:
        """Get information about this prompt version."""
        return {
            "version": self.version,
            "class": self.__class__.__name__,
            "description": self.__doc__.strip() if self.__doc__ else ""
        }
