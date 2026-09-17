"""Shared async transport for the Cerebras OpenAI-compatible inference API."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import httpx

from cle.harness.board_surface import (
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.reasoning import (
    native_reasoning_enabled,
    validate_native_reasoning_request,
)

# Cerebras only knows none/low/medium/high; harness efforts outside that range
# clamp to the nearest supported level.
_REASONING_EFFORTS = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}
_MAX_RETRY_AFTER_SECONDS = 30.0


def cerebras_reasoning_effort(reasoning: Mapping[str, Any]) -> str:
    """Map one normalized native-reasoning request onto `reasoning_effort`."""
    if not native_reasoning_enabled(reasoning):
        return "none"
    if "max_tokens" in reasoning:
        raise ValueError(
            "Cerebras has no reasoning token budget; use reasoning.effort "
            "or reasoning.enabled=false"
        )
    return _REASONING_EFFORTS[reasoning["effort"]]


@dataclass(frozen=True, slots=True)
class CerebrasConfig:
    model: str
    base_url: str = "https://api.cerebras.ai/v1"
    temperature: float = 0.3
    max_tokens: int | None = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    reasoning: Mapping[str, Any] | None = None


class CerebrasTransport:
    """Text-only Cerebras chat completions with native reasoning evidence."""

    def __init__(
        self,
        config: CerebrasConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.reasoning_request = validate_native_reasoning_request(config.reasoning)
        self.reasoning_effort = cerebras_reasoning_effort(self.reasoning_request)
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise ValueError("CEREBRAS_API_KEY is required for CerebrasTransport")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        started_at = time.monotonic()
        payload = {
            "model": self.config.model,
            "messages": openai_messages_with_board(
                request.messages,
                request.board_presentation,
                allow_image_input=False,
            ),
            "temperature": self.config.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        if self.config.max_tokens is not None:
            # Reasoning tokens count against this cap on Cerebras.
            payload["max_completion_tokens"] = self.config.max_tokens
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(_retry_delay(exc, attempt))
        assert response is not None
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        native_reasoning_value = (
            message.get("reasoning") or message.get("reasoning_content")
        )
        native_reasoning = (
            native_reasoning_value
            if isinstance(native_reasoning_value, str)
            else json.dumps(native_reasoning_value, sort_keys=True)
            if native_reasoning_value is not None
            else ""
        )
        provider_response_id = data.get("id")
        if not isinstance(provider_response_id, str):
            provider_response_id = None
        return ModelResponse(
            content=message.get("content") or "",
            model=data.get("model", self.config.model),
            usage=tuple(usage.items()),
            latency_ms=int((time.monotonic() - started_at) * 1000),
            finish_reason=choice.get("finish_reason"),
            native_reasoning=native_reasoning,
            reasoning_request=tuple(self.reasoning_request.items()),
            provider_response_id=provider_response_id,
            provider_request_id=response.headers.get("x-request-id"),
            provider_request_payload=sanitize_provider_payload(
                payload,
                request.board_presentation,
            ),
            provider_response_payload=sanitize_provider_payload(
                data,
                request.board_presentation,
            ),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honor a rate-limit `retry-after`, else the shared exponential backoff."""
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            retry_after = float(exc.response.headers.get("retry-after", ""))
        except ValueError:
            retry_after = 0.0
        if retry_after > 0:
            return min(retry_after, _MAX_RETRY_AFTER_SECONDS)
    return 0.05 * (2**attempt)
