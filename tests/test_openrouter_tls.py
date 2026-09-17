import asyncio
import logging
import ssl
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, call

import httpx
import pytest

from cle.harness.models import ModelMessage, ModelRequest
from cle.harness.providers import openrouter
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterTLSFailure,
    OpenRouterTransport,
)


@pytest.fixture
def model_request():
    return ModelRequest(
        decision_id="decision-2",
        session_id="game-1:RED",
        messages=(
            ModelMessage("system", "private system prompt"),
            ModelMessage("user", "private player state"),
        ),
    )


@pytest.fixture(autouse=True)
def retry_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(openrouter.asyncio, "sleep", sleep)
    return sleep


def _alert():
    return ssl.SSLError(
        ssl.SSL_ERROR_SSL,
        "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] sslv3 alert bad record mac (_ssl.c:1234)",
    )


def _success():
    return httpx.Response(
        200, json={"choices": [{"message": {"content": "<action>0</action>"}}]}
    )


def _owned_transport(monkeypatch, handler, *, max_retries=2):
    clients = []
    async_client = httpx.AsyncClient

    def create_client(**kwargs):
        client = async_client(transport=httpx.MockTransport(handler), **kwargs)
        monkeypatch.setattr(client, "aclose", AsyncMock(wraps=client.aclose))
        clients.append(client)
        return client

    factory = Mock(side_effect=create_client)
    monkeypatch.setattr(openrouter.httpx, "AsyncClient", factory)
    transport = OpenRouterTransport(
        OpenRouterConfig(
            model="test/model", timeout_seconds=7.5, max_retries=max_retries
        ),
        api_key="private-api-key",
    )
    return transport, clients, factory


@pytest.mark.asyncio
async def test_borrowed_alert_retries_same_client_payload_and_session(
    monkeypatch, model_request, retry_sleep, caplog
):
    seen = []

    def handler(request):
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
    monkeypatch, model_request, retry_sleep
):
    seen = []

    def handler(request):
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
@pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
async def test_tls_exhaustion_budget_backoff_and_safe_error(
    monkeypatch, model_request, retry_sleep, caplog, max_retries
):
    errors = []
    secret = "private-api-key private player state secret-header-value"

    def handler(request):
        error = httpx.ReadError(secret, request=request)
        error.__cause__ = _alert()
        errors.append(error)
        raise error

    transport, clients, factory = _owned_transport(
        monkeypatch, handler, max_retries=max_retries
    )
    try:
        with pytest.raises(OpenRouterTLSFailure) as caught:
            await transport.complete(model_request)
        error = caught.value
        assert isinstance(error, RuntimeError)
        assert vars(error) == {
            "context_id": model_request.decision_id,
            "session_id": model_request.session_id,
            "model": transport.config.model,
            "attempts": max_retries + 1,
        }
        assert str(error) == (
            "OpenRouter TLS connection failed (SSLV3_ALERT_BAD_RECORD_MAC) "
            f"after {max_retries + 1} transport attempt(s)."
        )
        assert error.args == (str(error),)
        assert error.__cause__ is errors[-1]
        assert len(errors) == factory.call_count == max_retries + 1
        assert retry_sleep.await_args_list == [
            call(delay) for delay in [1, 2, 4, 8, 8][:max_retries]
        ]
        assert not clients[0].is_closed
        clients[0].aclose.assert_not_awaited()
        for client in clients[1:]:
            assert client.is_closed
            client.aclose.assert_awaited_once_with()
        for value in (secret, "private-api-key", "private player state", "secret-header-value"):
            assert value not in str(error)
            assert value not in repr(error)
            assert value not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["__cause__", "__context__"])
@pytest.mark.parametrize("wrapper_type", [httpx.ConnectError, RuntimeError])
@pytest.mark.parametrize("structured_reason", [False, True])
async def test_wrapped_alert_and_cyclic_chain_retry(
    model_request, retry_sleep, link, wrapper_type, structured_reason
):
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

    def handler(request):
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


@pytest.mark.asyncio
@pytest.mark.parametrize("link", [None, "__cause__", "__context__"])
@pytest.mark.parametrize(
    "kind",
    ["certificate", "protocol", "eof", "local_bad_record", "conflicting_reason"],
)
async def test_other_ssl_errors_fail_closed_even_when_wrapped(
    model_request, retry_sleep, link, kind
):
    if kind == "certificate":
        # Even an alert-looking message must not override the certificate type.
        error = ssl.SSLCertVerificationError(ssl.SSL_ERROR_SSL, str(_alert()))
    elif kind == "eof":
        error = ssl.SSLEOFError(ssl.SSL_ERROR_EOF, "EOF occurred in violation of protocol")
    elif kind == "conflicting_reason":
        error = _alert()
        error.reason = "CERTIFICATE_VERIFY_FAILED"
    else:
        reason = (
            "TLSV1_ALERT_PROTOCOL_VERSION" if kind == "protocol"
            else "DECRYPTION_FAILED_OR_BAD_RECORD_MAC"
        )
        error = ssl.SSLError(ssl.SSL_ERROR_SSL, f"[SSL: {reason}] unrelated TLS failure")
    # A deeper alert cannot make an unrelated SSL error retryable.
    error.__cause__ = _alert()
    if link is not None:
        wrapper = httpx.ConnectError("wrapper")
        setattr(wrapper, link, error)
        error = wrapper
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model"), api_key="test-key", client=client
        )
        with pytest.raises(type(error)) as caught:
            await transport.complete(model_request)
    assert caught.value is error
    assert attempts == 1
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message_only", "suppressed_context", "explicit_cause"])
async def test_irrelevant_alerts_do_not_change_unrelated_failure_handling(
    model_request, retry_sleep, kind
):
    error = RuntimeError(str(_alert()))
    if kind != "message_only":
        error.__context__ = _alert()
        if kind == "explicit_cause":
            error.__cause__ = ValueError("actual cause")
        else:
            error.__suppress_context__ = True

    def handler(request):
        raise error

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        transport = OpenRouterTransport(
            OpenRouterConfig(model="test/model"), api_key="test-key", client=client
        )
        with pytest.raises(RuntimeError) as caught:
            await transport.complete(model_request)
    assert caught.value is error
    retry_sleep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcomes,max_retries,error_type,delays",
    [
        ([503, "tls", 200], 2, None, [0.05, 1]),
        (["tls", 503, 200], 2, None, [1, 0.1]),
        ([503, "tls"], 1, OpenRouterTLSFailure, [0.05]),
        (["tls", 503], 1, httpx.HTTPStatusError, [1]),
        (["tls", "network"], 1, httpx.ReadError, [1]),
        (["tls", 400], 2, httpx.HTTPStatusError, [1]),
        ([503, 429, 200], 2, None, [0.05, 0.1]),
        (["network", 200], 2, None, [0.05]),
        ([400], 2, httpx.HTTPStatusError, []),
    ],
)
async def test_http_and_tls_share_one_budget_and_preserve_legacy_failures(
    monkeypatch, model_request, retry_sleep, outcomes, max_retries, error_type, delays
):
    seen = []
    network_error = httpx.ReadError("legacy network failure")

    def handler(request):
        outcome = outcomes[len(seen)]
        seen.append(request)
        if outcome == "tls":
            raise _alert()
        if outcome == "network":
            raise network_error
        if outcome == 200:
            return _success()
        return httpx.Response(outcome)

    transport, clients, factory = _owned_transport(
        monkeypatch, handler, max_retries=max_retries
    )
    try:
        if error_type is None:
            assert (await transport.complete(model_request)).content == "<action>0</action>"
        else:
            with pytest.raises(error_type) as caught:
                await transport.complete(model_request)
            if outcomes[-1] == "network":
                assert caught.value is network_error
            if error_type is OpenRouterTLSFailure:
                assert caught.value.attempts == len(outcomes)
        assert len(seen) == len(outcomes)
        assert retry_sleep.await_args_list == [call(delay) for delay in delays]
        recovery_attempts = (
            len(outcomes) - outcomes.index("tls") - 1 if "tls" in outcomes else 0
        )
        assert factory.call_count == 1 + recovery_attempts
        assert not clients[0].is_closed
        assert all(client.is_closed for client in clients[1:])
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_recovery_does_not_disrupt_concurrent_shared_client_caller(
    monkeypatch, model_request
):
    held = asyncio.Event()
    release = asyncio.Event()
    recovery_attempts = 0

    async def handler(request):
        nonlocal recovery_attempts
        if request.headers["x-session-id"] == "game-1:BLUE":
            held.set()
            await release.wait()
            assert not clients[0].is_closed
        else:
            recovery_attempts += 1
            if recovery_attempts == 1:
                raise _alert()
        return _success()

    transport, clients, factory = _owned_transport(monkeypatch, handler)
    sibling = asyncio.create_task(
        transport.complete(replace(model_request, session_id="game-1:BLUE"))
    )
    try:
        await asyncio.wait_for(held.wait(), timeout=1)
        response = await asyncio.wait_for(transport.complete(model_request), timeout=1)
        assert response.content == "<action>0</action>"
        assert not sibling.done()
        assert transport.client is clients[0]
        clients[0].aclose.assert_not_awaited()
        assert clients[1].is_closed
        release.set()
        assert (await asyncio.wait_for(sibling, timeout=1)).content == "<action>0</action>"
        # Recovery state is per call, not a persistent replacement of self.client.
        await transport.complete(model_request)
        assert factory.call_count == 2
    finally:
        release.set()
        sibling.cancel()
        await asyncio.gather(sibling, return_exceptions=True)
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["post", "backoff"])
async def test_cancellation_during_recovery_cleans_up_and_propagates(
    monkeypatch, model_request, retry_sleep, phase
):
    entered = asyncio.Event()
    blocked = asyncio.Event()
    attempts = 0

    async def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 2 and phase == "post":
            entered.set()
            await blocked.wait()
        raise _alert()

    async def sleep(delay):
        if attempts == 2:
            entered.set()
            await blocked.wait()

    retry_sleep.side_effect = sleep
    transport, clients, factory = _owned_transport(monkeypatch, handler, max_retries=3)
    task = asyncio.create_task(transport.complete(model_request))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert attempts == factory.call_count == 2
        assert not clients[0].is_closed
        clients[0].aclose.assert_not_awaited()
        assert clients[1].is_closed
        clients[1].aclose.assert_awaited_once_with()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("tls_cleanup_error", [False, True])
async def test_cleanup_failure_cannot_retry_a_successful_post(
    monkeypatch, model_request, retry_sleep, tls_cleanup_error
):
    attempts = 0
    cleanup_error = _alert() if tls_cleanup_error else httpx.ReadError("cleanup failed")

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise _alert()
        clients[-1].aclose.side_effect = cleanup_error
        return _success()

    transport, clients, factory = _owned_transport(monkeypatch, handler, max_retries=3)
    try:
        with pytest.raises(type(cleanup_error)) as caught:
            await transport.complete(model_request)
        assert caught.value is cleanup_error
        assert attempts == factory.call_count == 2
        assert retry_sleep.await_args_list == [call(1)]
        clients[0].aclose.assert_not_awaited()
        clients[1].aclose.assert_awaited_once_with()
    finally:
        for client in clients[1:]:
            client.aclose.side_effect = None
            await client.aclose()
        await transport.aclose()
