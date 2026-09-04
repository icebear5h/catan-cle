"""Shared async transport for a vLLM OpenAI-compatible server."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from cle.harness.board_surface import (
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse


@dataclass(frozen=True, slots=True)
class VLLMConfig:
    model: str
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    temperature: float = 0.3
    max_tokens: int | None = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    allow_image_input: bool = False


class VLLMTransport:
    """Let vLLM continuously batch requests from many sandbox coroutines."""

    def __init__(self, config: VLLMConfig, *, client: Any | None = None) -> None:
        self.config = config
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        started_at = time.monotonic()
        payload = {
            "model": self.config.model,
            "messages": openai_messages_with_board(
                request.messages,
                request.board_presentation,
                allow_image_input=self.config.allow_image_input,
            ),
            "temperature": self.config.temperature,
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens
        response = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.config.base_url.rstrip('/')}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                        "x-session-id": request.session_id,
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
                await asyncio.sleep(0.05 * (2**attempt))
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
        native_details_value = message.get("reasoning_details")
        native_reasoning_details = (
            tuple(native_details_value)
            if isinstance(native_details_value, (list, tuple))
            else (native_details_value,)
            if native_details_value is not None
            else ()
        )
        provider_response_id = data.get("id")
        if not isinstance(provider_response_id, str):
            provider_response_id = None
        provider_request_id = response.headers.get("x-request-id")
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
            provider_response_id=provider_response_id,
            provider_request_id=provider_request_id,
            provider_native_finish_reason=provider_native_finish_reason,
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
