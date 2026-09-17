"""Async OpenRouter transport for fully assembled player contexts."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import quote, quote_plus, unquote

import httpx

from cle.harness.board_surface import (
    openai_messages_with_board,
    sanitize_provider_payload,
)
from cle.harness.models import ModelRequest, ModelResponse
from cle.harness.reasoning import validate_native_reasoning_request


logger = logging.getLogger(__name__)


def _classify_tls_error(exc: BaseException) -> str | None:
    """Inspect the active exception chain; any other SSL error fails closed."""
    classification = None
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            reason = getattr(current, "reason", None)
            if isinstance(current, ssl.SSLCertVerificationError) or not (
                reason == "SSLV3_ALERT_BAD_RECORD_MAC"
                or (
                    reason is None
                    and "[SSL: SSLV3_ALERT_BAD_RECORD_MAC]" in str(current)
                )
            ):
                return "other_tls"
            classification = "SSLV3_ALERT_BAD_RECORD_MAC"
        # Explicit causes supersede incidental context, just as in a traceback.
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return classification


class OpenRouterTLSFailure(RuntimeError):
    """Safe public summary; the original exception is retained only as a cause."""

    def __init__(
        self, request: ModelRequest, *, model: str, attempts: int
    ) -> None:
        self.context_id = request.decision_id
        self.session_id = request.session_id
        self.model = model
        self.attempts = attempts
        super().__init__(
            "OpenRouter TLS connection failed (SSLV3_ALERT_BAD_RECORD_MAC) "
            f"after {attempts} transport attempt(s)."
        )


class OpenRouterHTTPFailure(RuntimeError):
    """Bounded provider diagnostics, not a general-purpose free-text privacy filter."""

    def __init__(
        self,
        request: ModelRequest,
        *,
        model: str,
        attempts: int,
        response: httpx.Response,
        redactions: tuple[str, ...] = (),
    ) -> None:
        self.context_id = request.decision_id
        self.session_id = request.session_id
        self.model = model
        self.attempts = attempts
        self.status_code = response.status_code
        self.provider_message: str | None = None
        self.provider_request_id: str | None = None

        secrets = set(redactions)
        try:
            credential_headers = response.request.headers.multi_items()
        except RuntimeError:  # Direct callers may supply a response without a request.
            credential_headers = ()
        for name, value in credential_headers:
            if re.search(r"auth|key|token|secret|cookie|credential", name, re.I):
                secrets.add(value)
                secrets.update(part.strip("\"'") for part in re.split(r"[=;,\s]+", value))
        echoes = {message.content for message in request.messages}
        if request.board_presentation is not None and request.board_presentation.kind == "text":
            echoes.add(request.board_presentation.content)
        echoes.update(line.strip() for text in tuple(echoes) for line in text.splitlines())

        def sanitize(value: Any, *, limit: int, identifier: bool = False) -> str | None:
            if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                return None
            if identifier and (
                len(value) > limit or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value)
            ):
                return None
            # Check echoes before replacing secrets; never truncate before redaction.
            for texts, is_echo in ((echoes, True), (secrets, False)):
                for text in sorted(texts, key=len, reverse=True):
                    if not text or len(text) > len(value):
                        continue
                    variants = {text, html.escape(text), quote(text, safe=""), quote_plus(text)}
                    for ascii_only in (False, True):
                        escaped = json.dumps(text, ensure_ascii=ascii_only)[1:-1]
                        variants.update((escaped, escaped.replace("/", "\\/")))
                        variants.add(json.dumps(escaped)[1:-1])
                    for variant in sorted(variants, key=len, reverse=True):
                        if variant in value:
                            if is_echo or identifier:
                                return None
                            value = value.replace(variant, "[REDACTED]")
            decoded = value
            for _ in range(2):
                decoded = unquote(html.unescape(decoded))
            decoded = " ".join(decoded.split())
            if any(text and text in decoded for text in secrets | echoes):
                return None
            # Suppress obvious payloads, including escaped ones, rather than trying
            # to salvage arbitrary raw/metadata/flagged-input strings as diagnostics.
            if re.search(
                r"[{}<>]|\[\s*[\[\"']|data:|\\(?:u[0-9a-f]{4}|[nrtbf])|%[0-9a-f]{2}|"
                r"\b(?:messages|content|prompt|input|flagged_input|request|body|payload|headers|"
                r"authorization|cookie|api[_-]?key|metadata|raw)"
                r"[\s\"'\\]*[:=]",
                decoded,
                re.I,
            ):
                return None
            value = " ".join(value.split())
            if not value or not value.isprintable():
                return None
            return value if len(value) <= limit else value[:limit - 3] + "..."

        # httpx has already buffered the response. Limit parsing, not the wire read.
        try:
            data = response.json() if len(response.content) <= 16 * 1024 else None
        except (ValueError, RecursionError, httpx.ResponseNotRead):
            data = None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            self.provider_message = sanitize(data["error"].get("message"), limit=512)
        for name in ("x-request-id", "x-openrouter-request-id"):
            self.provider_request_id = sanitize(
                response.headers.get(name), limit=128, identifier=True
            )
            if self.provider_request_id is not None:
                break
        summary = (
            f"OpenRouter HTTP{self.status_code} failed after {attempts} transport attempt(s): "
            + (self.provider_message or "no usable structured error.message reason returned.")
        )
        if self.provider_request_id is not None:
            summary += f" (request ID: {self.provider_request_id})"
        super().__init__(summary)


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str
    temperature: float = 0.3
    max_tokens: int | None = 2048
    timeout_seconds: float = 120.0
    max_retries: int = 2
    reasoning: Mapping[str, Any] | None = None
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
            "messages": openai_messages_with_board(
                request.messages,
                request.board_presentation,
                allow_image_input=self.config.allow_image_input,
            ),
            "temperature": self.config.temperature,
            "reasoning": dict(self.reasoning_request),
        }
        if self.config.max_tokens is not None:
            payload["max_tokens"] = self.config.max_tokens

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
