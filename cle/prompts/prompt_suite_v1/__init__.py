"""
Prompt Suite V1: Comprehensive strategic guidance for Catan gameplay.

This version focuses on:
- Clear strategic principles
- Phase-specific guidance
- Pip-based thinking
- VP efficiency
- Common mistakes to avoid
"""

from cle.prompts.base import PromptContext, PromptSuite
from cle.prompts.prompt_suite_v1.decision import build_decision_prompt, parse_response
from cle.prompts.prompt_suite_v1.guidance import build_strategic_guidance
from cle.prompts.prompt_suite_v1.system import build_system_prompt

__all__ = ["PromptSuiteV1"]


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
        return build_system_prompt()

    def build_strategic_guidance(self, context: PromptContext) -> str:
        """Phase and situation-specific guidance."""
        return build_strategic_guidance(context)

    def build_decision_prompt(self, context: PromptContext) -> str:
        """Build complete decision prompt."""
        return build_decision_prompt(self.build_strategic_guidance(context), context)

    def parse_response(self, response: str) -> dict[str, str | int | None]:
        """Parse LLM response into structured format."""
        return parse_response(response)
