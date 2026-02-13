"""
Text-based action space for LLM agents.

Converts between LLM outputs (tool calls, text) and Catanatron actions.
"""

from typing import Dict, Any, List
from engine.models.enums import Action, ActionType


class ActionParser:
    """
    Parses LLM outputs into Catanatron actions.

    The LLM can output actions in multiple formats:
    1. Tool calls: build_settlement(node=3)
    2. Structured text: "Build settlement at node 3"
    3. JSON: {"action": "BUILD_SETTLEMENT", "node": 3}
    """

    def parse(self, llm_output: str) -> Action:
        """
        Parse LLM output into a Catanatron action.

        Args:
            llm_output: Text from LLM (tool call or natural language)

        Returns:
            Catanatron Action object

        Raises:
            ValueError: If action cannot be parsed
        """
        # TODO: Implement parsing logic
        # - Handle tool call format
        # - Handle natural language
        # - Handle JSON format
        # - Validate against valid actions
        raise NotImplementedError("Action parsing not yet implemented")

    def validate(self, action: Action, valid_actions: List[Action]) -> bool:
        """
        Check if parsed action is in the valid action set.

        Args:
            action: Parsed action
            valid_actions: List of valid actions from game state

        Returns:
            True if action is valid
        """
        return action in valid_actions


class ActionFormatter:
    """
    Formats Catanatron actions into semantic text for LLM context.

    This is used to show the LLM what actions are available and to
    describe opponent actions in observations.
    """

    def format_action_list(self, actions: List[Action]) -> str:
        """
        Format a list of valid actions with strategic context.

        Args:
            actions: List of valid Catanatron actions

        Returns:
            Formatted string like:
            "1. Build settlement at node 3 (wheat port: 6-wheat, 8-ore)
             2. Build road from node 3 to node 4 (extends toward ore)"
        """
        # TODO: Implement action list formatting
        # - Number each action
        # - Add strategic context (resource access, expansion direction)
        # - Group similar actions (all settlements together)
        return "TODO: Format action list"

    def format_single_action(self, action: Action, state: Any) -> str:
        """
        Format a single action with full context.

        This is used in the action log and memory updates.

        Args:
            action: Catanatron action
            state: Game state for context

        Returns:
            Semantic description like:
            "RED built settlement at node 12 (wheat port: 6-wheat, 8-ore), now 3 VP"
        """
        # TODO: Implement from action_formatter.py
        return "TODO: Format single action"
