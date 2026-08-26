"""Async OpenRouter transport for fully assembled player contexts."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.reasoning import validate_native_reasoning_request


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    reasoning: Mapping[str, Any] | None = None
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    extra_headers: Mapping[str, str] = field(
        default_factory=lambda: {
            "HTTP-Referer": "https://github.com/catan-learning",
            "X-Title": "Catan Agent Harness",
        }
    )


class OpenRouterTransport:
    """Reuse one async client while sending complete self-contained context."""

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = config
        self.reasoning_request = validate_native_reasoning_request(config.reasoning)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-session-id": request.session_id,
            **dict(self.config.extra_headers),
        }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "reasoning": dict(self.reasoning_request),
        }

        started_at = time.monotonic()
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    self.config.endpoint,
                    headers=headers,
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
                await asyncio.sleep(0.05 * (2**attempt))
        assert response is not None
        data = response.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        native_reasoning_value = (
            message.get("reasoning") or message.get("reasoning_content")
        )
        if native_reasoning_value is None:
            native_reasoning = ""
        elif isinstance(native_reasoning_value, str):
            native_reasoning = native_reasoning_value
        else:
            native_reasoning = json.dumps(
                native_reasoning_value,
                sort_keys=True,
                separators=(",", ":"),
            )

        details_value = message.get("reasoning_details")
        if details_value is None:
            native_reasoning_details = ()
        elif isinstance(details_value, (list, tuple)):
            native_reasoning_details = tuple(details_value)
        else:
            native_reasoning_details = (details_value,)

        provider_response_id = data.get("id")
        if not isinstance(provider_response_id, str):
            provider_response_id = None
        provider_request_id = (
            response.headers.get("x-request-id")
            or response.headers.get("x-openrouter-request-id")
        )
        provider_native_finish_reason = choice.get("native_finish_reason")
        if not isinstance(provider_native_finish_reason, str):
            provider_native_finish_reason = None

        return ModelResponse(
            content=message.get("content") or "",
            model=data.get("model", self.config.model),
            usage=tuple(usage.items()),
            latency_ms=int((time.monotonic() - started_at) * 1000),
            finish_reason=choice.get("finish_reason"),
            native_reasoning=native_reasoning,
            native_reasoning_details=native_reasoning_details,
            reasoning_request=tuple(self.reasoning_request.items()),
            provider_response_id=provider_response_id,
            provider_request_id=provider_request_id,
            provider_native_finish_reason=provider_native_finish_reason,
            provider_request_payload=payload,
            provider_response_payload=data,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()
