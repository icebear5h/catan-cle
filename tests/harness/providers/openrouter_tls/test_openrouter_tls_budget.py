"""Shared retry budget, backoff, and unrelated failures."""
import ssl
from typing import Any
from unittest.mock import AsyncMock, call

import httpx
import pytest

from cle.harness.models import ModelRequest
from cle.harness.providers.openrouter import (
    OpenRouterConfig,
    OpenRouterTLSFailure,
    OpenRouterTransport,
)

from .support import _alert, _owned_transport, _success


@pytest.mark.asyncio
@pytest.mark.parametrize("max_retries", [0, 1, 2, 5])
async def test_tls_exhaustion_budget_backoff_and_safe_error(
    monkeypatch: pytest.MonkeyPatch, model_request: ModelRequest, retry_sleep: AsyncMock, caplog: pytest.LogCaptureFixture, max_retries: int
) -> None:
    errors = []
    secret = "private-api-key private player state secret-header-value"

    def handler(request: httpx.Request) -> None:
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
@pytest.mark.parametrize("link", [None, "__cause__", "__context__"])
@pytest.mark.parametrize(
    "kind",
    ["certificate", "protocol", "eof", "local_bad_record", "conflicting_reason"],
)
async def test_other_ssl_errors_fail_closed_even_when_wrapped(
    model_request: ModelRequest, retry_sleep: AsyncMock, link: str | None, kind: str
) -> None:
    if kind == "certificate":
        # Even an alert-looking message must not override the certificate type.
        error = ssl.SSLCertVerificationError(ssl.SSL_ERROR_SSL, str(_alert()))
    elif kind == "eof":
        error = ssl.SSLEOFError(ssl.SSL_ERROR_EOF, "EOF occurred in violation of protocol")
    elif kind == "conflicting_reason":
        error = _alert()
        error.reason = "CERTIFICATE_VERIFY_FAILED"
    else:
        reason: Any = (
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

    def handler(request: httpx.Request) -> None:
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
    model_request: ModelRequest, retry_sleep: AsyncMock, kind: str
) -> None:
    error = RuntimeError(str(_alert()))
    if kind != "message_only":
        error.__context__ = _alert()
        if kind == "explicit_cause":
            error.__cause__ = ValueError("actual cause")
        else:
            error.__suppress_context__ = True

    def handler(request: httpx.Request) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
    model_request: ModelRequest,
    retry_sleep: AsyncMock,
    outcomes: list[int | str],
    max_retries: int,
    error_type: type[BaseException] | None,
    delays: list[float],
) -> None:
    seen = []
    network_error = httpx.ReadError("legacy network failure")

    def handler(request: httpx.Request) -> httpx.Response:
        outcome: Any = outcomes[len(seen)]
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
