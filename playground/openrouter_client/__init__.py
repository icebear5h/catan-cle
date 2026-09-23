"""Multi-provider VLM client for OpenRouter and Novita AI.

Supports image+text queries to vision models and text-only judge calls.
Each model entry specifies its provider so calls route to the right API.

Split from one module; ``playground.openrouter_client`` keeps every historical
name, including the ``httpx`` attribute that tests patch to stub transports.
"""

import httpx

from .benchmark import judge_responses, query_both_vlms
from .completions import query_text, query_vlm
from .config import (
    MODELS,
    PROVIDERS,
    ProviderConfig,
    TextResult,
    ToolLoopResult,
    VlmResult,
    _build_headers,
    _get_provider_config,
)
from .tool_loop import query_text_with_tools

__all__ = [
    "MODELS",
    "PROVIDERS",
    "ProviderConfig",
    "TextResult",
    "ToolLoopResult",
    "VlmResult",
    "_build_headers",
    "_get_provider_config",
    "httpx",
    "judge_responses",
    "query_both_vlms",
    "query_text",
    "query_text_with_tools",
    "query_vlm",
]
