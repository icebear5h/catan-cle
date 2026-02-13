"""
Prompt engineering suite for Catan LLM agents.

Modular, versioned, testable prompt system.
"""

from .prompt_suite_v1 import PromptSuiteV1
from .base import PromptSuite

__all__ = ["PromptSuite", "PromptSuiteV1"]

# Default prompt suite version
DEFAULT_PROMPT_SUITE = PromptSuiteV1
