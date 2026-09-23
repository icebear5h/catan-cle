"""Borrowed and owned client recovery on TLS alerts."""

import logging
import ssl
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest

from cle.harness.models import ModelRequest
from cle.harness.providers import openrouter
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterTransport,
)

from .support import _alert, _owned_transport, _success


@pytest.mark.asyncio
async def test_borrowed_alert_retries_same_client_payload_and_session(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, caplog: pytest.LogCaptureFixture
) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            raise _alert()
        return _success()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        close = AsyncMock(wraps=client.aclose)
        monkeypatch.setattr(client, "aclose", close)
        factory = Mock(side_effect=AssertionError("borrowed clients cannot be cloned"))
        monkeypatch.setattr(openrouter.httpx, "AsyncClient", factory)
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model"), api_key="private-api-key", client=client
        )
        with caplog.at_level(logging.WARNING, logger=openrouter.__name__):
            response = await transport.complete(model_request)
        await transport.aclose()

        assert response.content == "<action>0</action>"
        assert transport.client is client
        assert not client.is_closed
        close.assert_not_awaited()
        factory.assert_not_called()

    assert len(seen) == 2
    assert seen[0].content == seen[1].content
    assert seen[0].headers == seen[1].headers
    assert seen[1].headers["x-session-id"] == model_request.session_id
    assert retry_sleep.await_args_list == [call(1)]
    assert caplog.messages == [
        "OpenRouter retry classifier=SSLV3_ALERT_BAD_RECORD_MAC "
        "attempt=1 context_id='decision-2'"
    ]
    assert caplog.records[0].exc_info is None


@pytest.mark.asyncio
async def test_owned_recovery_uses_fresh_clients_with_default_verification(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock
) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) < 3:
            raise _alert()
        assert clients[1].is_closed
        return _success()

    transport, clients, factory = _owned_transport(monkeypatch, handler)
    try:
        response = await transport.complete(model_request)
        assert response.content == "<action>0</action>"
        assert len(clients) == len({id(client) for client in clients}) == 3
        # No verify override, clone arguments, or changes to the shared client.
        assert factory.call_args_list == [call(timeout=7.5)] * 3
        assert transport.client is clients[0]
        assert not clients[0].is_closed
        clients[0].aclose.assert_not_awaited()
        for client in clients[1:]:
            assert client.is_closed
            client.aclose.assert_awaited_once_with()
        assert all(request.content == seen[0].content for request in seen)
        assert all(request.headers == seen[0].headers for request in seen)
        assert retry_sleep.await_args_list == [call(1), call(2)]
    finally:
        await transport.aclose()
    clients[0].aclose.assert_awaited_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["__cause__", "__context__"])
@pytest.mark.parametrize("wrapper_type", [httpx.ConnectError, RuntimeError])
@pytest.mark.parametrize("structured_reason", [False, True])
async def test_wrapped_alert_and_cyclic_chain_retry(
    model_request: ModelRequest,
    retry_sleep: AsyncMock,
    link: str,
    wrapper_type: type[BaseException],
    structured_reason: bool,
) -> None:
    alert = _alert()
    if structured_reason:
        alert = ssl.SSLError(ssl.SSL_ERROR_SSL, "no reason in message")
        alert.reason = "SSLV3_ALERT_BAD_RECORD_MAC"
    middle = OSError("intermediate network wrapper")
    outer = wrapper_type("outer wrapper")
    setattr(outer, link, middle)
    setattr(middle, link, alert)
    setattr(alert, link, outer)
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise outer
        return _success()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model"), api_key="test-key", client=client
        )
        assert (await transport.complete(model_request)).content == "<action>0</action>"
    assert attempts == 2
    assert retry_sleep.await_args_list == [call(1)]
