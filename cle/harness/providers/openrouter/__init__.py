"""Async OpenRouter transport for fully assembled player contexts."""

from __future__ import annotations

# Re-exported so retry sleeps and client construction stay patchable here.
import asyncio as asyncio

import httpx as httpx

from cle.harness.providers.openrouter.failures import (
    OpenRouterHTTPFailure,
    OpenRouterTLSFailure,
)
from cle.harness.providers.openrouter.transport import (
    OpenRouterConfig,
    OpenRouterTransport,
)

__all__ = [
    "OpenRouterConfig",
    "OpenRouterHTTPFailure",
    "OpenRouterTLSFailure",
    "OpenRouterTransport",
]
