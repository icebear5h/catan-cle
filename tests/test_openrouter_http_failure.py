import html
import json
import ssl
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, call
from urllib.parse import quote, quote_plus

import httpx
import pytest

from cle.harness.models import ModelMessage, ModelRequest
from cle.harness.providers import openrouter
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterHTTPFailure,
    OpenRouterTLSFailure,
    OpenRouterTransport,
)


@pytest.fixture
def model_request():
    return ModelRequest(
        decision_id="decision-403",
        session_id="game-1:RED",
        messages=(
            ModelMessage("system", "private system rules"),
            ModelMessage("user", 'private state: "wood/ore"\nprivate second line'),
        ),
    )


@pytest.fixture(autouse=True)
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(openrouter.asyncio, "sleep", sleep)
    return sleep


async def _failure(monkeypatch, model_request, response, **config):
    seen = []

    def handler(request):
        seen.append(request)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        close = AsyncMock(wraps=client.aclose)
        monkeypatch.setattr(client, "aclose", close)
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model", **config), api_key="private-api-key", client=client
        )
        with pytest.raises(OpenRouterHTTPFailure) as caught:
            await transport.complete(model_request)
        await transport.aclose()
        close.assert_not_awaited()
        assert not client.is_closed
    error = caught.value
    assert len(seen) == 1
    assert seen[0].method == "POST"
    assert type(error.__cause__) is httpx.HTTPStatusError
    assert error.__cause__.response is response
    assert error.__cause__.request is seen[0]
    assert "HTTP403" in str(error)
    assert error.args == (str(error),)
    return error


@pytest.mark.asyncio
async def test_structured_reason_public_contract_and_request_id(
    monkeypatch, model_request, retry_sleep
):
    response = httpx.Response(
        403,
        headers={"x-request-id": "req-403", "x-openrouter-request-id": "secondary-id"},
        json={"error": {"message": "This model is not available in your region.", "code": 403}},
    )
    error = await _failure(monkeypatch, model_request, response)

    assert isinstance(error, RuntimeError)
    assert not isinstance(error, OpenRouterTLSFailure)
    assert vars(error) == {
        "context_id": "decision-403",
        "session_id": "game-1:RED",
        "model": "test/model",
        "attempts": 1,
        "status_code": 403,
        "provider_message": "This model is not available in your region.",
        "provider_request_id": "req-403",
    }
    assert str(error) == (
        "OpenRouter HTTP403 failed after 1 transport attempt(s): "
        "This model is not available in your region. (request ID: req-403)"
    )
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        b"",
        b"Forbidden: private-api-key",
        b"<html>private-api-key private system rules</html>",
        b'{"error":{"message":"private-api-key"}',
        b'{"error":{"message":"denied"}} trailing',
        b'{"error":{"message":"\xff"}}',
        b"null",
        b'[{"error":{"message":"do not expose"}}]',
        b'{"message":"do not expose"}',
        b'{"error":"do not expose"}',
        b'{"error":[{"message":"do not expose"}]}',
        b'{"error":{"message":{"text":"do not expose"}}}',
        b'{"error":{"message":["do not expose"]}}',
        b'{"error":{"message":403}}',
        b'{"error":{"message":true}}',
        b'{"error":{"message":null}}',
        b'{"error":{"message":"   "}}',
        b'{"error":{"metadata":{"error":{"message":"do not expose"}}}}',
        b'{"error":{"message":"denied","metadata":"' + b"x" * 16384 + b'"}}',
        b'{"error":{"message":"' + b"x" * 2049 + b'"}}',
        b'{"error":{"message":' + b"[" * 7000 + b"]" * 7000 + b"}}",
    ],
    ids=[
        "empty",
        "plaintext",
        "html",
        "malformed",
        "trailing-data",
        "invalid-encoding",
        "null",
        "array",
        "no-error",
        "string-error",
        "array-error",
        "object-message",
        "array-message",
        "number-message",
        "boolean-message",
        "null-message",
        "blank-message",
        "nested-metadata-message",
        "oversized-body",
        "oversized-field",
        "deeply-nested-message",
    ],
)
async def test_no_usable_structured_reason(monkeypatch, model_request, retry_sleep, body):
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(403, content=body, headers={"x-openrouter-request-id": "req-fallback"}),
    )
    assert error.provider_message is None
    assert error.provider_request_id == "req-fallback"
    assert "no usable structured error.message reason" in str(error)
    assert "req-fallback" in str(error)
    for secret in ("private-api-key", "private system rules", "do not expose"):
        assert secret not in str(error)
        assert secret not in repr(error)
        assert secret not in repr(error.args)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_message_and_allowlisted_request_headers_are_exposed(
    monkeypatch, model_request, retry_sleep
):
    hidden = "unrelated-sensitive-provider-data"
    error = await _failure(
        monkeypatch,
        model_request,
        httpx.Response(
            403,
            headers={"set-cookie": hidden, "x-debug": hidden},
            json={
                "error": {
                    "message": "Access denied by provider policy.",
                    "metadata": {"raw": hidden, "flagged_input": hidden, "request_id": hidden},
                    "raw": hidden,
                    "flagged_input": hidden,
                    "request_id": hidden,
                },
                "message": hidden,
                "request_id": hidden,
            },
        ),
    )
    assert error.provider_message == "Access denied by provider policy."
    assert error.provider_request_id is None
    assert hidden not in str(error)
    assert hidden not in repr(error)
    assert hidden not in repr(vars(error))
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_credentials_redacted_in_reason_and_request_id(
    monkeypatch, model_request, retry_sleep, caplog
):
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

    def handler(request):
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
    error = caught.value
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
    monkeypatch, model_request, retry_sleep, encoding, kind
):
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
    monkeypatch, model_request, retry_sleep, encoded
):
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
    monkeypatch, model_request, retry_sleep, echo
):
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
    monkeypatch, model_request, retry_sleep, message
):
    # JSON-encode explicitly to exercise escaped invalid Unicode, not httpx's encoder.
    response = httpx.Response(403, content=json.dumps({"error": {"message": message}}).encode())
    error = await _failure(monkeypatch, model_request, response)
    assert error.provider_message is None
    assert "unknown-private-text" not in str(error)
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_bounded_message_redacts_before_truncation(monkeypatch, model_request, retry_sleep):
    message = "Denied. " + "x" * 498 + "private-api-key" + "x" * 300
    error = await _failure(
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
async def test_unsafe_request_ids_suppressed(monkeypatch, model_request, retry_sleep, request_id):
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
async def test_oversized_body_not_parsed(monkeypatch, model_request, retry_sleep):
    response = httpx.Response(403, json={"error": {"message": "Denied."}, "raw": "x" * 16384})
    parse = Mock(side_effect=AssertionError("Oversized body must not be parsed"))
    monkeypatch.setattr(response, "json", parse)
    error = await _failure(monkeypatch, model_request, response)
    assert error.provider_message is None
    parse.assert_not_called()
    retry_sleep.assert_not_awaited()


@pytest.mark.parametrize("streaming", [False, True])
def test_direct_constructor_without_attached_request(model_request, streaming):
    body = b'{"error":{"message":"Denied for explicit-secret"}}'
    response = httpx.Response(
        403,
        headers={"x-request-id": "req-direct"},
        **({"stream": httpx.ByteStream(body)} if streaming else {"content": body}),
    )
    error = OpenRouterHTTPFailure(
        model_request,
        model="test/model",
        attempts=3,
        response=response,
        redactions=("explicit-secret",),
    )
    assert error.provider_message == (None if streaming else "Denied for [REDACTED]")
    assert error.provider_request_id == "req-direct"
    assert error.attempts == 3
    assert "explicit-secret" not in repr(error)
    assert not any(hasattr(error, name) for name in ("request", "response", "headers", "body"))


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("exhausted_budget", [False, True])
@pytest.mark.parametrize(
    "outcomes,delays",
    [
        ([403], []),
        ([503, 403], [0.05]),
        (["tls", 403], [1]),
        ([503, "tls", 403], [0.05, 1]),
        (["tls", 503, 403], [1, 0.1]),
    ],
)
async def test_attempt_count_no_403_retry_and_unchanged_client_cleanup(
    monkeypatch, model_request, retry_sleep, owned, exhausted_budget, outcomes, delays
):
    seen = []
    clients = []
    async_client = httpx.AsyncClient

    def handler(request):
        outcome = outcomes[len(seen)]
        seen.append(request)
        if outcome == "tls":
            raise ssl.SSLError(ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record")
        return httpx.Response(outcome, json={"error": {"message": "Access denied."}})

    def create_client(**kwargs):
        client = async_client(transport=httpx.MockTransport(handler), **kwargs)
        monkeypatch.setattr(client, "aclose", AsyncMock(wraps=client.aclose))
        clients.append(client)
        return client

    borrowed = None if owned else create_client()
    factory = Mock(side_effect=create_client)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", factory)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model",
            max_retries=len(outcomes) - 1 if exhausted_budget else 5,
            timeout_seconds=7.5,
        ),
        api_key="private-api-key",
        client=borrowed,
    )
    try:
        with pytest.raises(OpenRouterHTTPFailure) as caught:
            await transport.complete(model_request)
        assert caught.value.attempts == len(outcomes)
        assert caught.value.__cause__.response.status_code == 403
        assert len(seen) == len(outcomes)
        assert all(request.method == "POST" for request in seen)
        assert all(request.content == seen[0].content for request in seen)
        assert all(request.headers == seen[0].headers for request in seen)
        assert retry_sleep.await_args_list == [call(delay) for delay in delays]
        recovery_attempts = len(outcomes) - outcomes.index("tls") - 1 if "tls" in outcomes else 0
        assert factory.call_args_list == (
            [call(timeout=7.5)] * (1 + recovery_attempts) if owned else []
        )
        assert transport.client is clients[0]
        assert not clients[0].is_closed
        clients[0].aclose.assert_not_awaited()
        for client in clients[1:]:
            assert client.is_closed
            client.aclose.assert_awaited_once_with()
        await transport.aclose()
        if owned:
            clients[0].aclose.assert_awaited_once_with()
        else:
            clients[0].aclose.assert_not_awaited()
    finally:
        if not clients[0].is_closed:
            await clients[0].aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,max_retries,attempts", [(400, 5, 1), (401, 5, 1), (404, 5, 1), (429, 1, 2), (503, 1, 2)]
)
async def test_other_http_status_errors_unchanged(
    model_request, retry_sleep, status, max_retries, attempts
):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(status, json={"error": {"message": "Provider reason."}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model", max_retries=max_retries),
            api_key="private-api-key",
            client=client,
        )
        with pytest.raises(httpx.HTTPStatusError) as caught:
            await transport.complete(model_request)
    assert type(caught.value) is httpx.HTTPStatusError
    assert caught.value.response.status_code == status
    assert len(seen) == attempts
    assert retry_sleep.await_args_list == [call(0.05)] * (attempts - 1)
