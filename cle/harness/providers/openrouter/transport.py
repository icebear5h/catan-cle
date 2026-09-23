"""Async OpenRouter transport for fully assembled player contexts."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from cle.harness.board_surface import (
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.providers.openrouter.failures import (
    OpenRouterHTTPFailure,
    OpenRouterTLSFailure,
    _classify_tls_error,
)
from cle.harness.reasoning import validate_native_reasoning_request
from cle.players.data import JsonValue

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float = 0.3
    max_tokens: int | None = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    reasoning: Mapping[str, JsonValue] | None = None
    allow_image_input: bool = False
    endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    extra_headers: Mapping[str, str] = field(
        default_factory=lambda: {
            "HTTP-Referer": "https://github.com/catan-learning",
            "X-Title": "Catan Agent Harness",
        }
    )


class OpenRouterTransport:
    """Reuse one async client while sending complete self-contained context.

    TLS recovery uses isolated per-attempt clients only when we own the client.
    Borrowed clients are reused, never cloned or closed; their pools cannot be
    refreshed here. Retrying a lost response can duplicate billed inference.
    """

    def __init__(
        self,
        config: OpenRouterConfig,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config
        self.reasoning_request = validate_native_reasoning_request(config.reasoning)
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        self.api_key = resolved_key
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=config.timeout_seconds)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "x-session-id": request.session_id,
            **dict(self.config.extra_headers),
        }
        # A per-decision override from the frozen context wins; otherwise the
        # transport-level default applies. The effective request is recorded
        # on the response for trace evidence.
        reasoning = validate_native_reasoning_request(
            dict(request.reasoning_request)
            if request.reasoning_request is not None
            else self.reasoning_request
        )
        if request.reasoning_request is not None:
            # Policy-owned request: its cap wins, and None means uncapped.
            max_tokens = request.max_tokens
        else:
            max_tokens = self.config.max_tokens
        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": openai_messages_with_board(
                request.messages,
                request.board_presentation,
                allow_image_input=self.config.allow_image_input,
            ),
            "temperature": self.config.temperature,
            "reasoning": dict(reasoning),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        started_at = time.monotonic()
        response = None
        tls_retries = 0
        for attempt in range(self.config.max_retries + 1):
            retry_client = (
                httpx.AsyncClient(timeout=self.config.timeout_seconds)
                if self._owns_client and tls_retries
                else None
            )
            try:
                response = await (retry_client or self.client).post(
                    self.config.endpoint,
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                break
            except Exception as exc:
                tls_kind = _classify_tls_error(exc)
                if tls_kind == "SSLV3_ALERT_BAD_RECORD_MAC":
                    if attempt == self.config.max_retries:
                        raise OpenRouterTLSFailure(
                            request, model=self.config.model, attempts=attempt + 1
                        ) from exc
                    delay = 2 ** min(tls_retries, 3)
                    tls_retries += 1
                    logger.warning(
                        "OpenRouter retry classifier=%s attempt=%d context_id=%r",
                        tls_kind,
                        attempt + 1,
                        request.decision_id,
                    )
                elif tls_kind is not None:
                    raise
                elif isinstance(exc, (httpx.TransportError, httpx.HTTPStatusError)):
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403:
                        raise OpenRouterHTTPFailure(
                            request,
                            model=self.config.model,
                            attempts=attempt + 1,
                            response=exc.response,
                            redactions=(self.api_key,),
                        ) from exc
                    retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                        exc.response.status_code == 429
                        or exc.response.status_code >= 500
                    )
                    if not retryable or attempt == self.config.max_retries:
                        raise
                    delay = 0.05 * (2**attempt)
                else:
                    raise
            finally:
                # Outside retry catching: a cleanup failure must not replay a POST.
                if retry_client is not None:
                    await retry_client.aclose()
            await asyncio.sleep(delay)
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
        native_reasoning_details: tuple[JsonValue, ...]
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
            reasoning_request=tuple(reasoning.items()),
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
