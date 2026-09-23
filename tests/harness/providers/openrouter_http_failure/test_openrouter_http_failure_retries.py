"""Attempt counts, cleanup, and status passthrough."""
import ssl
from typing import Any
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest

from cle.harness.models import ModelRequest
from cle.harness.providers import openrouter
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterHTTPFailure,
    OpenRouterTransport,
)


@pytest.mark.parametrize("streaming", [False, True])
def test_direct_constructor_without_attached_request(model_request: ModelRequest, streaming: bool) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
    model_request: ModelRequest,
    retry_sleep: AsyncMock,
    owned: bool,
    exhausted_budget: bool,
    outcomes: list[int | str],
    delays: list[float],
) -> None:
    seen: list[httpx.Request] = []
    clients: list[Any] = []
    async_client = httpx.AsyncClient

    def handler(request: httpx.Request) -> httpx.Response:
        outcome: Any = outcomes[len(seen)]
        seen.append(request)
        if outcome == "tls":
            raise ssl.SSLError(ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record")
        return httpx.Response(outcome, json={"error": {"message": "Access denied."}})

    def create_client(**kwargs: object) -> httpx.AsyncClient:
        client: Any = async_client(transport=httpx.MockTransport(handler), **kwargs)
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
    model_request: ModelRequest, retry_sleep: AsyncMock, status: int, max_retries: int, attempts: int
) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
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
