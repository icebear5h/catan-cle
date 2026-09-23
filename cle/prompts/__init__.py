"""
Prompt engineering suite for Catan LLM agents.

Modular, versioned, testable prompt system.
"""

from .base import PromptSuite
from .prompt_suite_v1 import PromptSuiteV1

__all__ = ["PromptSuite", "PromptSuiteV1"]

# Default prompt suite version
DEFAULT_PROMPT_SUITE = PromptSuiteV1
