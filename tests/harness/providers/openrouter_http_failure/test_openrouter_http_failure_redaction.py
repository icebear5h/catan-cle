"""Credential redaction and fail-closed parsing."""
import html
import json
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock, Mock
from urllib.parse import quote, quote_plus

import httpx
import pytest

from cle.harness.models import ModelMessage, ModelRequest
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterHTTPFailure,
    OpenRouterTransport,
)

from .support import _failure


@pytest.mark.asyncio
async def test_actual_credentials_redacted_in_reason_and_request_id(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    credentials = (
        "private-api-key",
        "overridden-bearer",
        "proxy-credential",
        "custom-api-key",
        "cookie-session-secret",
        "cookie-other-secret",
        "custom-token",
        "client-default-secret",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer overridden-bearer"
        assert request.headers["x-client-secret"] == "client-default-secret"
        return httpx.Response(
            403,
            headers={"x-request-id": "req-overridden-bearer"},
            json={"error": {"message": "Access denied for " + " / ".join(credentials)}},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"X-Client-Secret": "client-default-secret"},
    ) as client:
        transport = OpenRouterTransport(
            OpenRouterConfig(
                model="test/model",
                extra_headers={
                    "Authorization": "Bearer overridden-bearer",
                    "Proxy-Authorization": "Basic proxy-credential",
                    "X-API-Key": "custom-api-key",
                    "Cookie": 'session=cookie-session-secret; other="cookie-other-secret"',
                    "X-Auth-Token": "custom-token",
                },
            ),
            api_key="private-api-key",
            client=client,
        )
        with pytest.raises(OpenRouterHTTPFailure) as caught:
            await transport.complete(model_request)
    error: Any = caught.value
    assert error.provider_message.startswith("Access denied for [REDACTED]")
    assert error.provider_request_id is None
    for credential in credentials:
        for public in (str(error), repr(error), repr(error.args), repr(vars(error)), caplog.text):
            assert credential not in public
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoding",
    ["literal", "json", "json-unicode", "double-json", "json-slashes", "html", "url", "form"],
)
@pytest.mark.parametrize("kind", ["credential", "echo", "echo-line"])
async def test_known_secrets_and_request_echo_escaped_variants(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, encoding: str, kind: str
) -> None:
    secret = 'private "quoted"/value & \u00e9\nsecond line'
    if kind == "credential":
        source = secret
    else:
        model_request = replace(model_request, messages=(ModelMessage("user", secret),))
        source = secret if kind == "echo" else secret.splitlines()[0]
    encoded = source
    if encoding in ("json", "json-unicode", "double-json", "json-slashes"):
        encoded = json.dumps(source, ensure_ascii=encoding != "json-unicode")[1:-1]
        if encoding == "double-json":
            encoded = json.dumps(encoded)[1:-1]
        elif encoding == "json-slashes":
            encoded = encoded.replace("/", "\\/")
    elif encoding == "html":
        encoded = html.escape(source)
    elif encoding == "url":
        encoded = quote(source, safe="")
    elif encoding == "form":
        encoded = quote_plus(source)

    response = httpx.Response(403, json={"error": {"message": "Denied for " + encoded}})
    if kind == "credential":
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: response)
        ) as client:
            response = await client.post("https://openrouter.test/", json={"messages": []})
        error = OpenRouterHTTPFailure(
            model_request, model="test/model", attempts=1, response=response, redactions=(secret,)
        )
    else:
        error = await _failure(monkeypatch, model_request, response)
    assert error.provider_message == ("Denied for [REDACTED]" if kind == "credential" else None)
    for public in (str(error), repr(error), repr(error.args), repr(vars(error))):
        assert source not in public
        assert encoded not in public
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded",
    [
        "private%2dapi%2dkey",
        "private%252dapi%252dkey",
        "private&#45;api&#45;key",
        "private&amp;#45;api&amp;#45;key",
        "privat\\u0065-api-key",
    ],
)
async def test_noncanonical_escaped_credentials_fail_closed(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, encoded: str
) -> None:
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(403, json={"error": {"message": "Denied for " + encoded}}),
    )
    assert error.provider_message is None
    assert encoded not in str(error)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", ["private system\nrules", "private%20system%0Arules"])
async def test_whitespace_normalization_cannot_reveal_known_echo(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, echo: str
) -> None:
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(403, json={"error": {"message": "Denied for " + echo}}),
    )
    assert error.provider_message is None
    assert "private system rules" not in str(error)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        'Rejected: {"messages":[{"role":"user","content":"unknown-private-text"}]}',
        'Rejected: \\"messages\\": [\\"unknown-private-text\\"]',
        "flagged_input: unknown-private-text",
        "Prompt=unknown-private-text",
        "raw: unknown-private-text",
        "metadata: unknown-private-text",
        "payload: unknown-private-text",
        "request: unknown-private-text",
        "Request body: unknown-private-text",
        "Authorization: Bearer unknown-private-text",
        "Cookie: session=unknown-private-text",
        '["unknown-private-text"]',
        "<html>unknown-private-text</html>",
        "&lt;html&gt;unknown-private-text&lt;/html&gt;",
        "%7B%22content%22%3A%22unknown-private-text%22%7D",
        "%257B%2522content%2522%253A%2522unknown-private-text%2522%257D",
        "data:image/png;base64,unknown-private-text",
        "Denied\x00unknown-private-text",
        "Denied\ud800unknown-private-text",
    ],
)
async def test_obvious_payload_and_malformed_text_fail_closed(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, message: str
) -> None:
    # JSON-encode explicitly to exercise escaped invalid Unicode, not httpx's encoder.
    response = httpx.Response(403, content=json.dumps({"error": {"message": message}}).encode())
    error = await _failure(monkeypatch, model_request, response)
    assert error.provider_message is None
    assert "unknown-private-text" not in str(error)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_message_redacts_before_truncation(monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock) -> None:
    message = "Denied. " + "x" * 498 + "private-api-key" + "x" * 300
    error: Any = await _failure(
        monkeypatch, model_request, httpx.Response(403, json={"error": {"message": message}})
    )
    assert len(error.provider_message) == 512
    assert error.provider_message.endswith("...")
    assert "private" not in str(error)
    assert len(str(error)) < 650
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_id",
    [
        "x" * 129,
        "private-api-key",
        "private system rules",
        "id\nforged",
        "id\n",
        '{"raw":"secret"}',
    ],
)
async def test_unsafe_request_ids_suppressed(
    monkeypatch: pytest.MonkeyPatch,
    model_request: ModelRequest,
    retry_sleep: AsyncMock,
    request_id: str,
) -> None:
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(
            403, headers={"x-request-id": request_id}, json={"error": {"message": "Denied."}}
        ),
    )
    assert error.provider_message == "Denied."
    assert error.provider_request_id is None
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_oversized_body_not_parsed(monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock) -> None:
    response = httpx.Response(403, json={"error": {"message": "Denied."}, "raw": "x" * 16384})
    parse = Mock(side_effect=AssertionError("Oversized body must not be parsed"))
    monkeypatch.setattr(response, "json", parse)
    error = await _failure(monkeypatch, model_request, response)
    assert error.provider_message is None
    parse.assert_not_called()
    retry_sleep.assert_not_awaited()
